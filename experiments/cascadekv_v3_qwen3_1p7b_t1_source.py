"""Canonical mechanical source resolution for the Qwen3-1.7B T1 protocol.

This module is intentionally data/tokenizer-only.  It has no model, tensor,
attention, routing, or CascadeKV imports.  Production resolution is index-only;
only preparation selection may enumerate candidate indices.
"""
from __future__ import annotations

from typing import Any, Callable

MODEL = "Qwen/Qwen3-1.7B"
TOKENIZER_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
MINIMUM = 4096
IDENTITY_KIND = "dataset_index"
SPECS = {
    "narrative": {"dataset": "emozilla/pg19", "config": None, "split": "train", "revision": "c021754c8e01c5b1cc83a1f549c1f97fbbb756b8", "field": "text"},
    "report": {"dataset": "ccdv/govreport-summarization", "config": None, "split": "train", "revision": "4e21184e01ae8017e2c036e180fe5e541fef60a0", "field": "report"},
    "qa": {"dataset": "zai-org/LongBench", "config": "narrativeqa", "split": "test", "revision": "75b6d5bffbcaa2cf4da85a9fa99939b13ee5b00b", "field": "context"},
}

def load_tokenizer() -> Any:
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL, revision=TOKENIZER_REVISION)

def load_dataset_for_spec(spec: dict[str, Any]) -> Any:
    """Load exactly the pinned dataset/config/split/revision, without a fallback."""
    from datasets import load_dataset
    if spec["dataset"] == "zai-org/LongBench":
        # The pinned revision is a legacy dataset-script repository.  Resolve
        # the one config/split file directly, rather than accepting the script
        # loader's mutable/default behaviour.
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=spec["dataset"], repo_type="dataset",
                               revision=spec["revision"], filename="narrativeqa/test-00000-of-00001.parquet")
        return load_dataset("parquet", data_files={spec["split"]: path}, split=spec["split"])
    return load_dataset(spec["dataset"], spec["config"], split=spec["split"], revision=spec["revision"])

def token_length_at_least(tokenizer: Any, text: str, minimum: int = MINIMUM) -> tuple[bool, int]:
    """The sole bounded-length predicate; second return is capped at ``minimum``."""
    if minimum <= 0:
        raise ValueError("minimum must be positive")
    ids = tokenizer(text, add_special_tokens=True, truncation=True, max_length=minimum).input_ids
    bounded = len(ids)
    return bounded >= minimum, bounded

def _spec_for_entry(entry: dict[str, Any]) -> dict[str, Any]:
    family = entry.get("family")
    if family not in SPECS:
        raise ValueError("unsupported frozen dataset family")
    spec = SPECS[family]
    for key in ("dataset", "config", "split", "revision", "field"):
        if entry.get(key) != spec[key]:
            raise ValueError(f"frozen source {key} mismatch")
    if entry.get("identity_kind") != IDENTITY_KIND or not isinstance(entry.get("dataset_index"), int) or entry["dataset_index"] < 0:
        raise ValueError("source identity must be a non-negative dataset_index")
    return spec

def mechanical_proof(entry: dict[str, Any], *, tokenizer: Any | None = None,
                     dataset_loader: Callable[[dict[str, Any]], Any] = load_dataset_for_spec) -> tuple[str, dict[str, Any]]:
    """Resolve precisely one frozen row and return text plus mechanical proof.

    This deliberately does not consult a row's ``id`` field and does not scan
    neighbouring rows.
    """
    spec = _spec_for_entry(entry)
    dataset = dataset_loader(spec)
    index = entry["dataset_index"]
    try:
        row = dataset[index]
    except (IndexError, KeyError) as exc:
        raise RuntimeError("frozen dataset_index unavailable") from exc
    exists = spec["field"] in row
    value = row.get(spec["field"])
    is_string = isinstance(value, str)
    # Keep the identity binding in the proof itself.  This makes the
    # selection proof and the later one-row production re-proof directly
    # comparable without relying on surrounding manifest structure.
    proof = {"dataset": spec["dataset"], "config": spec["config"], "split": spec["split"], "revision": spec["revision"], "family": entry["family"], "identity_kind": IDENTITY_KIND, "dataset_index": index, "field": spec["field"], "field_exists": exists, "is_string": is_string, "character_count": len(value) if is_string else None, "tokenizer_model": MODEL, "tokenizer_revision": TOKENIZER_REVISION, "minimum": MINIMUM, "at_least_4096": False, "bounded_frozen_tokenizer_length": None}
    if not exists or not is_string:
        return "" if not is_string else value, proof
    ok, bounded = token_length_at_least(tokenizer or load_tokenizer(), value, MINIMUM)
    proof["at_least_4096"] = ok
    proof["bounded_frozen_tokenizer_length"] = bounded
    return value, proof

# Future capture must call this name; it is deliberately the same function
# used for the manifest's second, production-resolution proof.
resolve_frozen_source = mechanical_proof

def select_first_unused(family: str, unavailable: set[int], *, start: int,
                        tokenizer: Any | None = None,
                        dataset_loader: Callable[[dict[str, Any]], Any] = load_dataset_for_spec) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Preparation-only ascending index selection using the production predicate."""
    if family not in SPECS:
        raise ValueError("unsupported frozen dataset family")
    spec = SPECS[family]; dataset = dataset_loader(spec); tok = tokenizer or load_tokenizer(); skips = []
    for index in range(start, len(dataset)):
        if index in unavailable:
            continue
        entry = {"family": family, **spec, "identity_kind": IDENTITY_KIND, "dataset_index": index}
        _, proof = mechanical_proof(entry, tokenizer=tok, dataset_loader=lambda _: dataset)
        if proof["field_exists"] and proof["is_string"] and proof["at_least_4096"]:
            return entry, proof, skips
        reason = "missing required field" if not proof["field_exists"] else "required field is not a Python string" if not proof["is_string"] else f"bounded frozen tokenizer length {proof['bounded_frozen_tokenizer_length']} < 4096"
        skips.append({"dataset_index": index, "reason": reason, "proof": proof})
    raise RuntimeError(f"no eligible unused {family} source")
