from itertools import pairwise

import pytest
import torch

from cascadekv.hierarchical_index import (
    HierarchicalIndex,
    SearchAccounting,
    accounting_ratios,
    bottom_up_p1_summaries,
    bottom_up_radius,
    radix_amortized_summary_updates,
    radix_completion_levels,
    radix_internal_node_count,
    support_index_storage,
)
from cascadekv.quantize import quantize_symmetric
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


def test_support_sets_are_constituent_deterministic_and_conservative() -> None:
    keys = torch.randn(43, 128)
    order = torch.randperm(128)
    first = HierarchicalIndex(keys, order, support_set_sizes=(1, 2, 4, 8, 16))
    second = HierarchicalIndex(keys, order, support_set_sizes=(1, 2, 4, 8, 16))
    queries = torch.randn(3, 128)
    for p in (1, 2, 4, 8, 16):
        for node in first.nodes:
            ids = first.support_ids(node.id, p)
            assert ids.numel() == min(p, node.end - node.start)
            assert all(node.start <= token < node.end for token in ids.tolist())
            assert torch.equal(ids, second.support_ids(node.id, p))
        for summary in ("support_set_global", "support_set_per_prototype"):
            for width in (16, 32, 64, 128):
                assert first.validate_bounds(queries, summary=summary, dimensions=width, representatives=p) == 0


def test_support_progressive_accounting_and_search_order_are_deterministic() -> None:
    tree = HierarchicalIndex(torch.randn(81, 128), torch.randperm(128), support_set_sizes=(4,))
    accounting = SearchAccounting()
    for width in (16, 32, 64, 128):
        accounting.add_support_node(11, 1, width, 4, per_prototype=False, tail=width < 128)
    assert accounting.metadata_bytes_read == 4 * 128 * 4 + (2 + 2 + 2 + 1) * 4
    assert accounting.projected_support_metadata_bytes == 4 * 8 * 10 + (2 + 2 + 2 + 1) * 4
    query = torch.randn(128)
    first = tree.best_first_search(query, k=8, summary="support_set_global", representatives=4)
    second = tree.best_first_search(query, k=8, summary="support_set_global", representatives=4)
    assert torch.equal(first.ids, second.ids)
    assert torch.equal(first.ids, (tree.keys @ query).topk(8).indices)


def test_farthest_point_support_selects_separated_outlier() -> None:
    keys = torch.zeros(17, 128)
    keys[-1, 0] = 100.0
    tree = HierarchicalIndex(keys, support_set_sizes=(2,))
    assert 16 in tree.support_ids(tree.root_id, 2).tolist()


def test_quantized_support_is_the_frozen_k4_encoding_and_bounds_original_keys() -> None:
    keys = torch.randn(33, 128)
    tree = HierarchicalIndex(keys, support_set_sizes=(2,))
    node = tree.root_id
    expected = quantize_symmetric(tree.ordered_keys[tree.support_ids(node, 2)], 4, 16)
    actual = tree.quantized_support(node, 2)
    assert torch.equal(actual.values, expected.values)
    assert torch.equal(actual.scales, expected.scales)
    queries = torch.randn(4, 128)
    assert tree.validate_bounds(queries, summary="support_set_k4_per_prototype", representatives=2) == 0
    assert tree.validate_bounds(queries, summary="support_set_q8k4_per_prototype", representatives=2) == 0


def test_q8_error_cushion_handles_adversarial_query_error() -> None:
    keys = torch.zeros(16, 128)
    keys[:, 0] = torch.linspace(-3, 3, 16)
    tree = HierarchicalIndex(keys, support_set_sizes=(1,))
    query = torch.zeros(128)
    query[0] = 0.01337
    assert tree.validate_bounds(query.unsqueeze(0), summary="support_set_q8k4_global", representatives=1) == 0


def test_storage_count_and_bottom_up_triangle_bound() -> None:
    assert radix_internal_node_count(1) == 0
    assert radix_internal_node_count(16) == 1
    assert radix_internal_node_count(17) == 3
    storage = support_index_storage(4096, 2)
    assert storage["internal_node_count"] == 273
    assert storage["total_index_bytes"] == storage["support_vector_bytes"] + storage["radius_metadata_bytes"] + storage["kmax_metadata_bytes"]
    children = torch.tensor([[0.0, 0.0], [4.0, 0.0]])
    radii = torch.tensor([1.0, 2.0])
    parent = torch.tensor([1.0, 0.0])
    assert bottom_up_radius(parent, children, radii) == pytest.approx(5.0)


def test_bottom_up_p1_balls_cover_all_descendant_keys_and_have_inflation() -> None:
    tree = HierarchicalIndex(torch.randn(257, 128), support_set_sizes=(1,))
    summaries = bottom_up_p1_summaries(tree)
    radii = {node_id: radius for node_id, (_, radius) in summaries.items()}
    for node in tree.nodes:
        prototype, radius = summaries[node.id]
        distances = (tree.ordered_keys[node.start : node.end] - prototype).norm(dim=1)
        assert distances.max() <= radius + 1e-5
        if node.children:
            child_radii = torch.stack([radii[child] for child in node.children])
            assert radii[node.id] >= child_radii.max()
        else:
            assert radii[node.id] == 0


def test_incremental_radix_completion_schedule_and_amortized_work() -> None:
    assert radix_completion_levels(15) == ()
    assert radix_completion_levels(16) == (1,)
    assert radix_completion_levels(256) == (1, 2)
    assert radix_completion_levels(4096) == (1, 2, 3)
    assert radix_completion_levels(65536) == (1, 2, 3, 4)
    assert radix_amortized_summary_updates() == pytest.approx(1 / 15)
