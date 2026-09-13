"""Small, pure-PyTorch randomized Hadamard transforms."""

from __future__ import annotations

import math

import torch


def _require_power_of_two(size: int) -> None:
    if size <= 0 or size & (size - 1):
        raise ValueError(f"the final dimension must be a positive power of two, got {size}")


def fast_walsh_hadamard(x: torch.Tensor) -> torch.Tensor:
    """Apply an orthonormally normalized FWHT along the final dimension.

    This is a butterfly implementation and deliberately never materializes a
    dense Hadamard matrix.  The normalized transform is self-inverse.
    """
    size = x.shape[-1]
    _require_power_of_two(size)
    result = x
    width = 1
    while width < size:
        result = result.reshape(*x.shape[:-1], -1, 2, width)
        left, right = result.unbind(dim=-2)
        result = torch.stack((left + right, left - right), dim=-2)
        width *= 2
    return result.reshape_as(x) / math.sqrt(size)


def random_signs(
    size: int,
    *,
    seed: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a reproducible Rademacher (+1/-1) diagonal for a given seed."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    signs = torch.randint(0, 2, (size,), generator=generator, dtype=torch.int64)
    return (signs.mul(2).sub(1)).to(device=device, dtype=dtype)


def signed_hadamard(x: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Apply ``H(Dx)`` where D is a deterministic random-sign diagonal.

    The same seed must be used for queries and keys to preserve their dot
    products under the rotation.
    """
    signs = random_signs(x.shape[-1], seed=seed, device=x.device, dtype=x.dtype)
    return fast_walsh_hadamard(x * signs)


def inverse_signed_hadamard(x: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Invert :func:`signed_hadamard` (the inverse is ``D(Hx)``)."""
    signs = random_signs(x.shape[-1], seed=seed, device=x.device, dtype=x.dtype)
    return fast_walsh_hadamard(x) * signs
