import pytest
import torch

from cascadekv.fixed_point import (
    FixedFormat,
    fixed_accumulate,
    fixed_multiply,
    float_to_fixed,
    integer_dot_times_scale,
)


def test_float_conversion_rounding_sign_zero_and_saturation() -> None:
    converted = float_to_fixed(torch.tensor([-9.0, -1.25, -0.125, 0.0, 0.125, 1.25, 9.0]), FixedFormat(8, 4))
    assert converted.codes.tolist() == [-128, -20, -2, 0, 2, 20, 127]
    assert converted.clipping_rate == pytest.approx(2 / 7)


def test_fixed_multiply_matches_hand_computed_example() -> None:
    fmt = FixedFormat(8, 4)
    left, right = float_to_fixed(torch.tensor([1.5]), fmt), float_to_fixed(torch.tensor([-2.0]), fmt)
    assert fixed_multiply(left, right, fmt).codes.tolist() == [-48]


def test_integer_dot_and_wide_accumulation() -> None:
    scale = float_to_fixed(torch.tensor([0.25, -0.5]), FixedFormat(12, 4))
    contribution = integer_dot_times_scale(torch.tensor([100, 100]), scale, FixedFormat(24, 8))
    accumulated = fixed_accumulate(
        type(contribution)(contribution.codes.reshape(1, 2), contribution.format, contribution.clipping_rate),
        FixedFormat(24, 8),
    )
    assert contribution.dequantize().tolist() == [25.0, -50.0]
    assert accumulated.codes.tolist() == [[1638400, -1638400]]


def test_candidate_evaluator_reports_clipping_and_numerical_error() -> None:
    from experiments.qwen_fixed_point_range import evaluate_candidate, quantized_groups

    torch.manual_seed(3)
    q, keys, order = torch.randn(128), torch.randn(32, 128), torch.randperm(128)
    dots, scales, _, _ = quantized_groups(q, keys, order)
    result = evaluate_candidate(
        FixedFormat(16, 4), FixedFormat(24, 8), [(dots, scales)]
    )
    assert result["clipping_saturation_rate"]["scale_product"] == 0.0
    row = result["metrics_against_floating_scale_q8_k4"]["128"]
    assert row["mean_absolute_numerical_error"] >= 0.0
    assert 0.0 <= row["top_8_recall"] <= 1.0
