import torch

from cascadekv.hadamard import random_coordinate_indices, signed_hadamard
from cascadekv.scorer import (
    attention_mass_recall,
    full_dot_scores,
    gqa_kv_head_for_query,
    has_nontrivial_top_k,
    pearson_correlation,
    progressive_dot_scores,
    relative_attention_mass_recall,
    spearman_correlation,
    top_k_recall,
)
from experiments.qwen_partial_dot import add_metrics, new_record


def test_full_dimension_progressive_score_is_exact() -> None:
    q, k = torch.randn(2, 128), torch.randn(2, 7, 128)
    q_rotated, k_rotated = signed_hadamard(q, seed=4), signed_hadamard(k, seed=4)
    assert torch.allclose(
        progressive_dot_scores(q_rotated, k_rotated)[128], full_dot_scores(q_rotated, k_rotated)
    )


def test_ranking_metrics_have_sane_ranges_and_exact_recall() -> None:
    full = torch.tensor([[0.2, 3.0, -1.0, 1.5]])
    assert top_k_recall(full, full, 2) == 1.0
    expected_mass = torch.softmax(full, -1).topk(2, -1).values.sum().item()
    assert attention_mass_recall(full, full, 2) == expected_mass
    assert relative_attention_mass_recall(full, full, 2) == 1.0
    assert 0.0 <= top_k_recall(torch.flip(full, [-1]), full, 2) <= 1.0
    assert 0.0 <= attention_mass_recall(torch.flip(full, [-1]), full, 2) <= 1.0
    assert 0.0 <= relative_attention_mass_recall(torch.flip(full, [-1]), full, 2) <= 1.0
    assert -1.0 <= pearson_correlation(torch.flip(full, [-1]), full) <= 1.0
    assert -1.0 <= spearman_correlation(torch.flip(full, [-1]), full) <= 1.0


def test_top_k_eligibility_excludes_trivial_cases() -> None:
    assert not has_nontrivial_top_k(31, 8)
    assert has_nontrivial_top_k(32, 8)
    assert not has_nontrivial_top_k(127, 32)
    assert has_nontrivial_top_k(128, 32)


def test_experiment_metrics_do_not_count_trivial_top_k_samples() -> None:
    record = new_record()
    scores = torch.arange(31, dtype=torch.float32)
    add_metrics(record, scores, scores)
    assert len(record["pearson"]) == 1
    assert "top_8_recall" not in record

    add_metrics(record, torch.arange(32, dtype=torch.float32), torch.arange(32, dtype=torch.float32))
    assert record["top_8_recall"] == [1.0]


def test_random_coordinate_subsets_are_deterministic() -> None:
    assert torch.equal(random_coordinate_indices(128, 32, seed=7), random_coordinate_indices(128, 32, seed=7))
    assert not torch.equal(random_coordinate_indices(128, 32, seed=7), random_coordinate_indices(128, 32, seed=8))
    assert torch.equal(random_coordinate_indices(128, 128, seed=7), torch.arange(128))


def test_gqa_query_head_mapping() -> None:
    assert [gqa_kv_head_for_query(head, 16, 4) for head in range(16)] == [
        0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3
    ]
