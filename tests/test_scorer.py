import torch

from cascadekv.hadamard import signed_hadamard
from cascadekv.scorer import (
    attention_mass_recall,
    full_dot_scores,
    pearson_correlation,
    progressive_dot_scores,
    spearman_correlation,
    top_k_recall,
)


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
    assert 0.0 <= top_k_recall(torch.flip(full, [-1]), full, 2) <= 1.0
    assert 0.0 <= attention_mass_recall(torch.flip(full, [-1]), full, 2) <= 1.0
    assert -1.0 <= pearson_correlation(torch.flip(full, [-1]), full) <= 1.0
    assert -1.0 <= spearman_correlation(torch.flip(full, [-1]), full) <= 1.0
