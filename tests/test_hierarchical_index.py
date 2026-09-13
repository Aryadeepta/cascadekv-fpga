from itertools import pairwise

import pytest
import torch

from cascadekv.hierarchical_index import HierarchicalIndex, SearchAccounting, accounting_ratios
from experiments.qwen_hierarchical_search import score_metrics


def index(count: int = 37) -> HierarchicalIndex:
    torch.manual_seed(4)
    return HierarchicalIndex(torch.randn(count, 128), torch.randperm(128))


def test_radix_tree_ranges_relationships_and_leaf_ids() -> None:
    tree = index(257)
    assert [tree.nodes[node].start for node in tree.levels[0]] == list(range(257))
    assert tree.nodes[tree.root_id].start == 0
    assert tree.nodes[tree.root_id].end == 257
    for node in tree.nodes:
        if node.children:
            children = [tree.nodes[child] for child in node.children]
            assert len(children) <= 16
            assert children[0].start == node.start and children[-1].end == node.end
            assert all(child.parent == node.id for child in children)
            assert all(left.end == right.start for left, right in pairwise(children))


def test_bounds_are_valid_and_prefix_box_is_explicitly_not_assumed_safe() -> None:
    tree = index()
    queries = torch.randn(3, 128)
    assert tree.validate_bounds(queries, summary="full_box_bound", dimensions=128) == 0
    for width in (16, 32, 64):
        assert tree.validate_bounds(queries, dimensions=width) == 0


def test_best_first_matches_dense_topk_on_synthetic_data() -> None:
    tree = index(53)
    query = torch.randn(128)
    result = tree.best_first_search(query, k=8)
    expected = (tree.keys @ query).topk(8).indices
    assert torch.equal(result.ids, expected)


def test_beam_is_deterministic_and_adaptive_is_exact() -> None:
    tree = index(81)
    query = torch.randn(128)
    first = tree.beam_search(query, beam_width=8, schedule={0: 16, 1: 32, 2: 64})
    second = tree.beam_search(query, beam_width=8, schedule={0: 16, 1: 32, 2: 64})
    assert torch.equal(first.ids, second.ids)
    adaptive = tree.best_first_search(query, k=8, adaptive=True, adaptive_policy="descend_immediately")
    assert torch.equal(adaptive.ids, (tree.keys @ query).topk(8).indices)
    assert adaptive.accounting.refinements_16_to_32 == 0


def test_adaptive_descends_with_progressive_width_without_full_summary() -> None:
    tree = index(81)
    result = tree.best_first_search(
        torch.randn(128), k=8, adaptive=True, adaptive_policy="descend_immediately"
    )
    # Level-one nodes are expanded after their 32-D bounds; none needed a
    # 128-D summary fetch to permit descent.
    expanded = [node.id for node in tree.nodes if node.children]
    assert any(result.accounting.fetched_summary_width.get(node, 0) < 128 for node in expanded)
    assert result.accounting.refinements_16_to_32 == 0
    assert result.accounting.refinements_32_to_64 == 0


def test_ambiguity_policy_refines_only_after_an_exact_threshold_exists() -> None:
    tree = index(81)
    result = tree.best_first_search(
        torch.randn(128),
        k=1,
        adaptive=True,
        adaptive_policy="ambiguity_refine",
        ambiguity_margin=float("inf"),
    )
    # Before the first exact leaf, adaptive mode descends.  Once a threshold
    # exists, the intentionally broad margin permits refinement.
    assert result.accounting.leaf_tokens_evaluated >= 1
    assert result.accounting.refinements_16_to_32 + result.accounting.refinements_32_to_64 > 0


def test_progressive_metadata_charges_each_coordinate_once() -> None:
    accounting = SearchAccounting()
    for width in (16, 32, 64, 128):
        accounting.add_node(11, 1, width, residual=True)
    # min/max FP32 coordinates: 2 values * 128 dimensions * 4 bytes.
    # Three prefix residual scalars are separately stored metadata.
    assert accounting.metadata_bytes_read == 2 * 128 * 4 + 3 * 4
    assert accounting.unique_metadata_bytes == accounting.metadata_bytes_read
    assert accounting.repeated_metadata_bytes == 0
    assert accounting.fetched_summary_width[11] == 128


def test_top32_metrics_reject_an_insufficient_candidate_set() -> None:
    full = torch.arange(40, dtype=torch.float32)
    with pytest.raises(ValueError, match="at least 32"):
        score_metrics(torch.arange(8), torch.arange(8), full)


def test_accounting_formulas_and_degenerate_sizes() -> None:
    for count in (1, 15, 16, 17):
        tree = index(count)
        assert tree.nodes[tree.root_id].end == count
        result = tree.best_first_search(torch.randn(128), k=1)
        ratios = accounting_ratios(result.accounting, count)
        assert ratios["estimated_total_routing_bytes"] == (
            ratios["metadata_bytes_read"] + ratios["leaf_k_bytes_read"]
        )
        assert 0 < ratios["fraction_leaf_k_read"] <= 1
