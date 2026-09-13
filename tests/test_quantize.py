import torch

from cascadekv.quantize import (
    quantization_diagnostics,
    quantize_symmetric,
    quantized_ordered_progressive_scores,
)
from experiments.qwen_quantized_ordered_sweep import storage_accounting, validate_metric_schema


def test_symmetric_int4_and_int8_ranges_and_determinism() -> None:
    source = torch.tensor([-100.0, -1.2, 0.0, 0.7, 100.0] * 32)
    int4_a, int4_b = quantize_symmetric(source, 4, 16), quantize_symmetric(source, 4, 16)
    int8 = quantize_symmetric(source, 8, 16)
    assert int(int4_a.values.min()) >= -7 and int(int4_a.values.max()) <= 7
    assert int(int8.values.min()) >= -127 and int(int8.values.max()) <= 127
    assert torch.equal(int4_a.values, int4_b.values)
    assert torch.equal(int4_a.scales, int4_b.scales)


def test_zero_and_dequantization_are_safe_and_approximate() -> None:
    zero = quantize_symmetric(torch.zeros(128), 4, 16)
    assert torch.equal(zero.values, torch.zeros(128, dtype=torch.int8))
    assert torch.equal(zero.dequantize(), torch.zeros(128))
    source = torch.linspace(-2, 2, 128)
    recovered = quantize_symmetric(source, 8, 128).dequantize()
    assert torch.max((source - recovered).abs()) < 0.02


def test_integer_accumulation_matches_hand_computed_scaled_dot() -> None:
    q, keys = torch.tensor([1.0, -2.0]), torch.tensor([[3.0, 4.0], [-1.0, 2.0]])
    scores, q_codes, k_codes = quantized_ordered_progressive_scores(
        q, keys, [0, 1], q_bits=4, k_bits=4, group_size=2, dimensions=(2,), unbiased=False
    )
    integer = (q_codes.values.to(torch.int32) * k_codes.values[0].to(torch.int32)).sum(dtype=torch.int32)
    expected = integer.float() * q_codes.scales[0] * k_codes.scales[0, 0]
    assert scores[2][0] == expected
    assert scores[2].dtype == torch.float32


def test_cumulative_groups_equal_direct_quantized_prefixes_and_nested_dimensions() -> None:
    q, keys, order = torch.randn(128), torch.randn(5, 128), torch.randperm(128)
    scores, q_codes, k_codes = quantized_ordered_progressive_scores(
        q, keys, order, q_bits=8, k_bits=4, group_size=16, unbiased=False
    )
    for dimension in (16, 32, 64, 128):
        direct = torch.zeros(keys.shape[0])
        for start in range(0, dimension, 16):
            group = start // 16
            dot = (q_codes.values[start : start + 16].to(torch.int32) * k_codes.values[:, start : start + 16].to(torch.int32)).sum(1, dtype=torch.int32)
            direct += dot.float() * q_codes.scales[group] * k_codes.scales[:, group]
        assert torch.allclose(scores[dimension], direct)


def test_fp32_ordered_128_score_remains_exact() -> None:
    from cascadekv.scorer import full_dot_scores, ordered_progressive_dot_scores

    q, keys, order = torch.randn(128), torch.randn(4, 128), torch.randperm(128)
    assert torch.allclose(ordered_progressive_dot_scores(q, keys, order)[128], full_dot_scores(q, keys))


def test_diagnostics_and_packed_storage_accounting() -> None:
    source = torch.randn(128)
    diagnostic = quantization_diagnostics(source, quantize_symmetric(source, 4, 16))
    assert diagnostic.clipping_rate == 0.0
    accounting = storage_accounting()["int4"]
    assert accounting["128"]["total_bytes"] == 66
    assert accounting["32"]["total_bytes"] == 72
    assert accounting["16"]["total_bytes"] == 80
    assert accounting["16"]["progressive_read_bytes"]["16"]["total_bytes"] == 10


def test_quantized_result_schema_requires_all_requested_metrics() -> None:
    names = {
        "pearson", "spearman", "top_8_recall", "top_32_recall", "attention_mass_top_8",
        "attention_mass_top_32", "relative_attention_mass_top_8", "relative_attention_mass_top_32",
        "normalized_score_mse", "delta_relmass_top_8", "delta_relmass_top_32", "delta_recall_top_8",
        "delta_recall_top_32",
    }
    row = {name: 0.0 for name in names}
    validate_metric_schema({"q8_k4": {"16": {"16": row}}})
