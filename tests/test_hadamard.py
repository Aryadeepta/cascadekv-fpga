import torch

from cascadekv.hadamard import inverse_signed_hadamard, signed_hadamard


def test_shape_and_norm_preservation() -> None:
    x = torch.randn(2, 3, 128)
    transformed = signed_hadamard(x, seed=7)
    assert transformed.shape == x.shape
    assert torch.allclose(transformed.norm(dim=-1), x.norm(dim=-1), atol=1e-5, rtol=1e-5)


def test_signs_are_deterministic() -> None:
    x = torch.randn(4, 128)
    assert torch.equal(signed_hadamard(x, seed=9), signed_hadamard(x, seed=9))
    assert not torch.equal(signed_hadamard(x, seed=9), signed_hadamard(x, seed=10))


def test_shared_rotation_preserves_dot_products_and_inverse() -> None:
    x, y = torch.randn(5, 128), torch.randn(5, 128)
    rotated_x, rotated_y = signed_hadamard(x, seed=3), signed_hadamard(y, seed=3)
    assert torch.allclose((x * y).sum(-1), (rotated_x * rotated_y).sum(-1), atol=2e-5, rtol=1e-5)
    assert torch.allclose(inverse_signed_hadamard(rotated_x, seed=3), x, atol=2e-5, rtol=1e-5)
