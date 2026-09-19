"""Model-independent mechanical contracts for CASCADEKV capture and evaluation.

This module intentionally contains no routing parameters or action schedules.  It
is safe to use with synthetic models; real-model capture is a separate host
operation which must supply a validated provenance record.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import re
from typing import Any, Literal, Mapping, Sequence

import torch


Family = Literal["qwen3", "phi3"]


@dataclass(frozen=True)
class ModelGeometry:
    model_family: Family
    hidden_size: int
    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    context_length: int
    sampled_layers: tuple[int, ...]
    sampled_positions: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.model_family not in ("qwen3", "phi3"):
            raise ValueError("unsupported model family")
        if min(self.hidden_size, self.num_layers, self.num_q_heads, self.num_kv_heads, self.head_dim, self.context_length) < 1:
            raise ValueError("geometry dimensions must be positive")
        if self.num_q_heads % self.num_kv_heads:
            raise ValueError("Q heads must divide evenly into KV heads")
        if self.hidden_size != self.num_q_heads * self.head_dim:
            raise ValueError("hidden_size must equal num_q_heads * head_dim")
        if not self.sampled_layers or not self.sampled_positions:
            raise ValueError("sampled layers and positions are required")
        if len(set(self.sampled_layers)) != len(self.sampled_layers) or any(x < 0 or x >= self.num_layers for x in self.sampled_layers):
            raise ValueError("sampled layers must be unique valid layer indices")
        if len(set(self.sampled_positions)) != len(self.sampled_positions) or any(x < 0 or x >= self.context_length for x in self.sampled_positions):
            raise ValueError("sampled positions must be unique positions in context")

    def kv_head_for_query(self, q_head: int) -> int:
        if not 0 <= q_head < self.num_q_heads:
            raise ValueError("query head outside geometry")
        # GQA/MQA is a geometric relationship, not a family-name shortcut.
        # Keeping this derivation here makes the public mapping safe for any
        # supported head ratio (including Phi's MHA ratio of one).
        q_per_kv = self.num_q_heads // self.num_kv_heads
        kv_head = q_head // q_per_kv
        if not 0 <= kv_head < self.num_kv_heads:  # defensive against future edits
            raise ValueError("derived KV head outside geometry")
        return kv_head

    @property
    def q_per_kv(self) -> int:
        return self.num_q_heads // self.num_kv_heads

    @property
    def schedule_cell_count(self) -> int:
        return len(self.sampled_layers) * self.num_kv_heads


QWEN_GEOMETRY = ModelGeometry("qwen3", 2048, 28, 16, 8, 128, 4096, (0, 7, 14, 21, 27), (2047, 3071, 4095))
PHI35_GEOMETRY = ModelGeometry("phi3", 3072, 32, 32, 32, 96, 8192, (0, 8, 16, 24, 31), (4095, 6143, 8191))


@dataclass(frozen=True)
class TrafficAccounting:
    head_dim: int
    fp16_key_bytes: int
    fp16_value_bytes: int
    k4_groups: int
    k4_key_bytes: int


def traffic_accounting(head_dim: int) -> TrafficAccounting:
    if head_dim < 16 or head_dim % 16:
        raise ValueError("K4/group16 requires head_dim divisible by 16")
    groups = head_dim // 16
    # Repository packing: eight 4-bit codes (8 B) plus an FP16 scale (2 B).
    return TrafficAccounting(head_dim, head_dim * 2, head_dim * 2, groups, groups * 10)


def k4_group_partitions(head_dim: int) -> tuple[tuple[int, ...], ...]:
    traffic_accounting(head_dim)
    return tuple(tuple(range(i, i + 16)) for i in range(0, head_dim, 16))


def g4_partitions(head_dim: int) -> tuple[tuple[int, ...], ...]:
    if head_dim < 4 or head_dim % 4:
        raise ValueError("G4 requires a dimension divisible by four")
    width = head_dim // 4
    return tuple(tuple(range(i, i + width)) for i in range(0, head_dim, width))


def validate_static_schedule(geometry: ModelGeometry, cells: Sequence[tuple[int, int]]) -> None:
    expected = {(layer, kv) for layer in geometry.sampled_layers for kv in range(geometry.num_kv_heads)}
    actual = list(cells)
    if len(actual) != len(set(actual)):
        raise ValueError("schedule has duplicate cells")
    if set(actual) != expected:
        missing, extra = expected - set(actual), set(actual) - expected
        raise ValueError(f"schedule cells mismatch (missing={sorted(missing)}, extra={sorted(extra)})")


@dataclass(frozen=True)
class RoutingProfile:
    """Frozen family-specific routing choices, intentionally distinct from geometry."""
    family: Family
    profile_id: str
    lambdas: Mapping[int, float]
    reserve_policy: Mapping[str, int | float | str]


def require_routing_profile(geometry: ModelGeometry, profile: RoutingProfile | None) -> RoutingProfile:
    if profile is None:
        raise ValueError(f"{geometry.model_family} evaluation requires an explicitly frozen routing profile")
    if profile.family != geometry.model_family or not isinstance(profile.profile_id, str) or not profile.profile_id.strip():
        raise ValueError("routing profile family/id mismatch")
    expected_layers = set(geometry.sampled_layers)
    if set(profile.lambdas) != expected_layers:
        raise ValueError("routing profile must cover exactly every sampled layer")
    for layer, value in profile.lambdas.items():
        if not isinstance(layer, int) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError("routing profile lambdas must be finite nonnegative numbers")
    policy = profile.reserve_policy
    # Schema v1 deliberately describes existing frozen sink/local mechanics.
    # A future family cannot accidentally become 'frozen' by passing {}.
    if policy.get("schema") != "cascadekv-reserve-v1":
        raise ValueError("routing profile reserve policy requires recognized schema")
    # Only the frozen absolute-token convention has an executable definition.
    # ``fraction_of_context`` is intentionally rejected until a future protocol
    # defines its rounding and interaction with the candidate budget.
    if policy.get("context_behavior") != "absolute_tokens":
        raise ValueError("routing profile reserve policy requires the defined absolute_tokens behavior")
    required = {"sink", "local"}
    if not required <= set(policy):
        raise ValueError("routing profile reserve policy is incomplete")
    for key, value in policy.items():
        if key in {"schema", "context_behavior"}:
            continue
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError("routing profile reserve values must be finite nonnegative numbers")
    return profile


# Explicit fail-closed marker.  It is intentionally not a RoutingProfile and
# cannot be supplied to an evaluator as though Phase B had frozen science.
PHI35_ROUTING_PROFILE_UNFROZEN = "PHI35_ROUTING_PROFILE_UNFROZEN"


class CaptureAdapter:
    family: Family
    identity: str
    def capture(self, module: torch.nn.Module, hidden: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class Qwen3CaptureAdapter(CaptureAdapter):
    family, identity = "qwen3", "qwen3-post-rope-qkv-v1"
    def capture(self, module: torch.nn.Module, hidden: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
        shape = (*hidden.shape[:-1], -1, module.head_dim)
        q = module.q_norm(module.q_proj(hidden).view(shape)).transpose(1, 2)
        k = module.k_norm(module.k_proj(hidden).view(shape)).transpose(1, 2)
        v = module.v_proj(hidden).view(shape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        return q, k, v


class Phi3CaptureAdapter(CaptureAdapter):
    family, identity = "phi3", "phi3-post-rope-qkv-v1"
    def capture(self, module: torch.nn.Module, hidden: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from transformers.models.phi3.modeling_phi3 import apply_rotary_pos_emb
        shape = (*hidden.shape[:-1], -1, module.head_dim)
        packed = module.qkv_proj(hidden)
        q_width = module.config.num_attention_heads * module.head_dim
        k_width = module.num_key_value_heads * module.head_dim
        q = packed[..., :q_width].view(shape).transpose(1, 2)
        k = packed[..., q_width:q_width + k_width].view(shape).transpose(1, 2)
        v = packed[..., q_width + k_width:].view(shape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        return q, k, v


def capture_target_layer_post_rope_qkv_low_memory(model: torch.nn.Module, input_ids: torch.Tensor, layer: int, *, adapter: CaptureAdapter) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Capture only one target attention boundary, copying it to CPU immediately.

    The model's own forward executes predecessor semantics; the hook observes
    the exact attention input boundary rather than recreating family internals.
    """
    if layer < 0 or layer >= len(model.layers):
        raise ValueError("target layer outside model")
    captured: dict[str, torch.Tensor] = {}
    attention = model.layers[layer].self_attn
    def hook(module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        pos = kwargs.get("position_embeddings")
        if hidden is None or pos is None:
            raise RuntimeError("installed Transformers attention hook boundary changed")
        q, k, v = adapter.capture(module, hidden, pos)
        # Capture computation is retained as FP32 CPU tensors for exact
        # adapter validation.  Serialization is a separate, explicit step.
        captured.update(q=q.detach().float().cpu(), k=k.detach().float().cpu(), v=v.detach().float().cpu())
    handle = attention.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.inference_mode():
            model(input_ids=input_ids)
    finally:
        handle.remove()
    if set(captured) != {"q", "k", "v"}:
        raise RuntimeError("capture hook did not execute")
    return captured["q"], captured["k"], captured["v"]


def serialize_qkv_for_cache(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, storage_dtype: str = "float16") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert validated capture tensors to the authoritative cache dtype."""
    dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if storage_dtype not in dtypes:
        raise ValueError("unsupported serialized cache dtype")
    dtype = dtypes[storage_dtype]
    if any(not torch.is_floating_point(x) for x in (q, k, v)):
        raise ValueError("Q/K/V cache tensors must be floating point")
    return tuple(x.detach().to(device="cpu", dtype=dtype).contiguous() for x in (q, k, v))  # type: ignore[return-value]


@dataclass(frozen=True)
class CacheProvenance:
    model_name: str; revision: str; family: Family; config_sha256: str; tokenizer_revision: str
    geometry: ModelGeometry; context: int; source_identity: Mapping[str, Any] | str; source_proof: Mapping[str, Any] | str
    layer: int; q_shape: tuple[int, ...]; k_shape: tuple[int, ...]; v_shape: tuple[int, ...]
    dtype: str; attention_scaling: float; capture_adapter: str

    def validate(self) -> None:
        if not self.model_name.strip() or not self.revision.strip() or not self.tokenizer_revision.strip():
            raise ValueError("cache provenance requires model and tokenizer identity")
        if self.family != self.geometry.model_family:
            raise ValueError("cache provenance family does not match geometry")
        if not re.fullmatch(r"[0-9a-f]{64}", self.config_sha256):
            raise ValueError("cache provenance config digest must be 64 lowercase hex characters")
        if len(self.q_shape) != 4 or self.context != self.geometry.context_length:
            raise ValueError("invalid cache provenance")
        if self.layer < 0 or self.layer >= self.geometry.num_layers or not self.source_identity or not self.source_proof:
            raise ValueError("cache provenance lacks source/layer binding")
        if not isinstance(self.source_identity, Mapping) or not isinstance(self.source_proof, Mapping):
            raise ValueError("cache provenance requires structured source identity and proof")
        expected_q = (1, self.geometry.num_q_heads, self.context, self.geometry.head_dim)
        expected_k = (1, self.geometry.num_kv_heads, self.context, self.geometry.head_dim)
        if self.q_shape != expected_q or self.k_shape != expected_k or self.v_shape != expected_k:
            raise ValueError("cache shapes do not match geometry")
        if self.dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("cache provenance dtype is not explicit/supported")
        if not math.isfinite(self.attention_scaling) or self.attention_scaling <= 0 or not self.capture_adapter.strip():
            raise ValueError("cache provenance lacks capture semantics")

    def canonical_digest(self) -> str:
        # repr() is not an archival format: it is sensitive to implementation
        # details and mapping order.  This wire representation is stable.
        payload = asdict(self)
        return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()
