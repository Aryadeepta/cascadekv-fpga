import torch

from cascadekv.fixed_point import (
    SCORE_MAX,
    SCORE_MIN,
    factorized_scale_product_codes,
    progressive_dot_rtl_update,
    q8k4_rtl_error_cushion,
    q8k4_rtl_score_codes,
    scale_product_rtl_codes,
)
from cascadekv.quantize import quantize_symmetric


def test_scale_product_and_progressive_update_are_integer_rtl_formulas() -> None:
    q = torch.tensor([1, 65535, 1, 32768], dtype=torch.int64)
    k = torch.tensor([4095, 65535, 4096, 512], dtype=torch.int64)
    assert factorized_scale_product_codes(q, k)[0].tolist() == [0, 32767, 1, 2048]
    assert scale_product_rtl_codes(torch.tensor([0.0]), torch.tensor([1.0])).tolist() == [0]
    lanes_q = torch.tensor([[127] * 16, [-128] * 16], dtype=torch.int8)
    lanes_k = torch.tensor([[7] * 16, [7] * 16], dtype=torch.int8)
    dot, contribution, score = progressive_dot_rtl_update(lanes_q, lanes_k, torch.tensor([4096, 4096]), torch.tensor([SCORE_MAX, SCORE_MIN]))
    assert dot.tolist() == [14224, -14336]
    assert contribution.tolist() == [SCORE_MAX, SCORE_MIN]
    assert score.tolist() == [SCORE_MAX, SCORE_MIN]


def test_eight_group_rtl_score_and_cushion_cover_dequantized_dot() -> None:
    generator = torch.Generator().manual_seed(33)
    q = quantize_symmetric(torch.randn(128, generator=generator), 8, 16)
    k = quantize_symmetric(torch.randn(5, 128, generator=generator), 4, 16)
    codes = q8k4_rtl_score_codes(q, k)
    assert codes.dtype == torch.int64 and codes.shape == (5,)
    real = q.dequantize() @ k.dequantize().T
    rtl = codes.float() / 8192
    assert torch.all(real <= rtl + q8k4_rtl_error_cushion(q, k) + 1e-5)
