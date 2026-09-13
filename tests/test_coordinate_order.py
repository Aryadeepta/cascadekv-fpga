import torch

from cascadekv.coordinate_order import (
    contribution_energy,
    covariance_with_full_score,
    greedy_block_ordering,
    validate_permutation,
)
from cascadekv.scorer import (
    full_dot_scores,
    normalized_score_mse,
    ordered_progressive_dot_scores,
)
from experiments.qwen_coordinate_order import calibration_contributions


def test_all_ordering_strategies_are_deterministic_permutations() -> None:
    contributions = torch.randn(40, 128)
    for strategy in (contribution_energy, covariance_with_full_score, greedy_block_ordering):
        first, second = strategy(contributions), strategy(contributions)
        validate_permutation(first, 128)
        assert torch.equal(first, second)


def test_ordered_full_dimension_is_exact_and_prefixes_are_nested() -> None:
    q, k = torch.randn(128), torch.randn(19, 128)
    order = torch.randperm(128)
    scores = ordered_progressive_dot_scores(q, k, order)
    assert torch.allclose(scores[128], full_dot_scores(q, k))
    ordered_contributions = q[order] * k[:, order]
    assert torch.allclose(scores[16], ordered_contributions[:, :16].sum(-1) * 8)
    assert torch.allclose(scores[32], ordered_contributions[:, :32].sum(-1) * 4)


def test_cumulative_scoring_agrees_with_direct_indexed_scoring() -> None:
    q, k, order = torch.randn(3, 128), torch.randn(3, 11, 128), torch.randperm(128)
    scores = ordered_progressive_dot_scores(q, k, order, unbiased=False)
    for dimension in (16, 32, 64, 128):
        direct = (q[..., order[:dimension]].unsqueeze(-2) * k[..., order[:dimension]]).sum(-1)
        # Cumulative hardware-style accumulation and Torch's reduction tree
        # differ by a few float32 ulps around near-zero cancellation.
        assert torch.allclose(scores[dimension], direct, rtol=1e-5, atol=1e-6)


def test_normalized_score_mse_for_exact_scores_is_zero() -> None:
    scores = torch.randn(4, 17)
    assert normalized_score_mse(scores, scores) == 0.0


def test_calibration_builder_uses_only_supplied_tensors() -> None:
    q = torch.ones(128)
    keys = torch.full((2, 128), 3.0)
    withheld_q = torch.full((128,), 99.0)
    withheld_keys = torch.full((2, 128), 99.0)
    contributions = calibration_contributions([(q, keys)])
    assert contributions.shape == (2, 128)
    assert torch.equal(contributions, torch.full((2, 128), 3.0))
    assert not torch.equal(contributions, calibration_contributions([(withheld_q, withheld_keys)]))
