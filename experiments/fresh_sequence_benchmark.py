#!/usr/bin/env python3
"""Frozen CascadeKV-v1 fresh-text benchmark (capture/evaluate/merge shards).

This program has no calibration or search code.  Every routing parameter is
read from ``configs/cascadekv_v1.json`` and rejected if it is not v1.
"""
from __future__ import annotations

import argparse, copy, hashlib, heapq, json, math, statistics, sys, time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cascadekv.adaptive_lifting import StreamingLiftingForest, reconstruct_pair
from cascadekv.evaluation_core import (
    exact_attention_reference as core_attention_reference,
    exact_rerank_and_output as core_output_metrics,
    select_flat_k4 as core_quantized_flat,
    traffic as core_traffic,
)
from cascadekv.quantize import quantize_symmetric
from experiments.grouped_variance_gate import candidate_budget, group_energy, reserve_ids
from experiments.streaming_grouped_variance_gate import forest_route

MODEL = "Qwen/Qwen3-0.6B"
LAYERS = (0, 7, 14, 21, 27)
NARRATIVEQA_PARQUET_REVISION = "75b6d5bffbcaa2cf4da85a9fa99939b13ee5b00b"
NARRATIVEQA_PARQUET_FILE = "narrativeqa/test-00000-of-00001.parquet"
SOURCES = {
    # No fallback is intentional: an unavailable requested dataset fails loudly.
    "narrative": {"kind": "hf", "dataset": "emozilla/pg19", "config": None, "split": "train", "index": 0, "text_field": "text", "category": "long-form narrative/book-style", "hub_repository_used": "emozilla/pg19", "canonical_dataset_family": "DeepMind PG-19 (Parquet mirror)"},
    "report": {"kind": "hf", "dataset": "ccdv/govreport-summarization", "config": None, "split": "train", "index": 0, "text_field": "report", "category": "report/document-style", "hub_repository_used": "ccdv/govreport-summarization", "canonical_dataset_family": "GovReport"},
    "qa": {"kind": "hf", "dataset": "zai-org/LongBench", "config": "narrativeqa", "split": "test", "index": 0, "text_field": "context", "loader": "direct_parquet", "parquet_prefix": "narrativeqa/", "revision": NARRATIVEQA_PARQUET_REVISION, "category": "QA/information-dense long-context", "hub_repository_used": "zai-org/LongBench", "canonical_dataset_family": "LongBench / NarrativeQA"},
}
HIGHER_BETTER = ("top8_recall", "relative_exact_attention_mass", "cosine_similarity")
LOWER_BETTER = ("relative_l2_error", "absolute_l2_error")
BASELINE_METHODS = ("dense_exact", "flat_q8k4", "cascadekv_v1", "cascadekv_mean_only")
Q_HEADS = 16
KV_HEADS = 8
FRACTION_CONFIG_KEYS = {0.05: "0.05", 0.10: "0.10"}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fraction_config_key(fraction: float) -> str:
    """Return the explicit frozen-config key for a supported operating point."""
    for operating_point, key in FRACTION_CONFIG_KEYS.items():
        if math.isclose(fraction, operating_point):
            return key
    raise ValueError(f"unsupported frozen configuration fraction: {fraction!r}")


def load_frozen(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    required = {"architecture_version": "cascadekv-v1", "atom_size": 8, "builder": "binary_counter", "variance_groups": 4}
    if any(data.get(k) != v for k, v in required.items()):
        raise ValueError("not the immutable CascadeKV-v1 configuration")
    expected_fraction_keys = set(FRACTION_CONFIG_KEYS.values())
    if not expected_fraction_keys <= set(data.get("reserve_schedule", {})):
        raise ValueError("v1 reserve schedule is incomplete")
    lambdas = data.get("qwen3_0_6b", {}).get("lambdas", {})
    if not lambdas:
        raise ValueError("v1 Qwen lambda configuration is missing")
    for layer, layer_lambdas in lambdas.items():
        if not expected_fraction_keys <= set(layer_lambdas):
            raise ValueError(f"v1 Qwen lambda configuration is incomplete for layer {layer}")
    return data


def frozen_routing_config(config: dict[str, Any], layer: int, fraction: float) -> tuple[dict[str, Any], float]:
    """Read the schedule and lambda used by one frozen evaluation operating point."""
    key = fraction_config_key(fraction)
    return config["reserve_schedule"][key], config["qwen3_0_6b"]["lambdas"][str(layer)][key]


def token_length_at_least(tokenizer: Any, text: str, required_tokens: int) -> tuple[bool, int]:
    """Bounded length probe; its result is never a document token count."""
    if required_tokens <= 0:
        raise ValueError("required_tokens must be positive")
    ids = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=required_tokens,
    ).input_ids
    observed = len(ids)
    return observed >= required_tokens, observed


def direct_parquet_files(spec: dict[str, Any], dataset_revision: str | None) -> list[str]:
    """Return config-scoped Parquet files from one immutable dataset commit."""
    if not dataset_revision:
        raise RuntimeError(f"direct Parquet loader for {spec['dataset']} requires a pinned dataset revision")
    from huggingface_hub import HfApi
    prefix = spec["parquet_prefix"]
    paths = sorted(
        path for path in HfApi().list_repo_files(
            repo_id=spec["dataset"], repo_type="dataset", revision=dataset_revision,
        )
        if path.startswith(prefix) and path.endswith(".parquet")
    )
    if not paths:
        raise RuntimeError(
            f"no Parquet files under {prefix!r} at {spec['dataset']}@{dataset_revision}"
        )
    if (spec["dataset"], spec.get("config"), dataset_revision) == (
        "zai-org/LongBench", "narrativeqa", NARRATIVEQA_PARQUET_REVISION,
    ) and NARRATIVEQA_PARQUET_FILE not in paths:
        raise RuntimeError(
            f"required NarrativeQA Parquet file {NARRATIVEQA_PARQUET_FILE!r} is absent at "
            f"{spec['dataset']}@{dataset_revision}"
        )
    return paths


def streaming_dataset(spec: dict[str, Any], dataset_revision: str | None) -> tuple[Any, list[str] | None]:
    """Open a source in streaming mode, bypassing legacy scripts when requested."""
    from datasets import load_dataset
    if spec.get("loader") == "direct_parquet":
        paths = direct_parquet_files(spec, dataset_revision)
        uris = [f"hf://datasets/{spec['dataset']}@{dataset_revision}/{path}" for path in paths]
        return load_dataset("parquet", data_files={spec["split"]: uris}, split=spec["split"], streaming=True), paths
    return load_dataset(spec["dataset"], spec["config"], split=spec["split"], streaming=True,
                        revision=dataset_revision), None


def resolve_text(sequence: str, local_jsonl: Path | None, *, required_tokens: int | None = None,
                 tokenizer: Any | None = None, dataset_revision: str | None = None) -> tuple[str, dict[str, Any]]:
    if local_jsonl:
        for line_number, line in enumerate(local_jsonl.read_text().splitlines()):
            row = json.loads(line)
            if str(row.get("id")) == sequence:
                if not isinstance(row.get("text"), str): raise ValueError("local JSONL row needs string text")
                result = {"kind": "local_jsonl", "path": str(local_jsonl), "line": line_number, "id": row["id"], "example_index": line_number, "example_identifier": row["id"], "category": row.get("category")}
                if required_tokens is not None:
                    if tokenizer is None:
                        raise ValueError("tokenizer is required when selecting by token length")
                    accepted, observed = token_length_at_least(tokenizer, row["text"], required_tokens)
                    result["token_length_probe"] = {"minimum_required": required_tokens,
                                                    "verified_at_least_minimum": accepted,
                                                    "probe_length": observed,
                                                    "probe_is_capped_at_minimum": True}
                    if not accepted:
                        raise RuntimeError(f"local JSONL row {sequence!r} does not satisfy {required_tokens} tokenizer tokens")
                return row["text"], result
        raise KeyError(f"sequence {sequence!r} absent from {local_jsonl}")
    if sequence not in SOURCES: raise KeyError(f"unknown built-in source {sequence!r}; choices: {sorted(SOURCES)}")
    spec = SOURCES[sequence]
    dataset, parquet_files = streaming_dataset(spec, dataset_revision)
    # The configured index is preferred.  If it is too short, selection is
    # deterministic and depends only on tokenizer length, never routing data.
    for index, row in enumerate(dataset):
        text = row.get(spec["text_field"])
        if not isinstance(text, str):
            continue
        if required_tokens is not None:
            if tokenizer is None:
                raise ValueError("tokenizer is required when selecting by token length")
            acceptable, observed = token_length_at_least(tokenizer, text, required_tokens)
            probe = {"minimum_required": required_tokens,
                     "verified_at_least_minimum": acceptable,
                     "probe_length": observed,
                     "probe_is_capped_at_minimum": True}
        else:
            acceptable, probe = True, None
        if index == int(spec["index"]) and (required_tokens is None or acceptable):
            result = {**spec, "example_identifier": row.get("id", index), "example_index": index, "requested_sequence": sequence}
            if parquet_files is not None: result["parquet_files"] = parquet_files
            if probe is not None: result["token_length_probe"] = probe
            return text, result
        if index >= int(spec["index"]) and acceptable:
            result = {**spec, "example_identifier": row.get("id", index), "example_index": index, "requested_sequence": sequence}
            if parquet_files is not None: result["parquet_files"] = parquet_files
            if probe is not None: result["token_length_probe"] = probe
            return text, result
    raise RuntimeError(f"no streaming {spec['dataset']} example satisfies {required_tokens} tokenizer tokens")


def hf_revision(kind: str, repo: str, revision: str = "main") -> tuple[str | None, dict[str, Any]]:
    """Resolve a mutable Hub revision to a commit.  Deliberately fails on lookup errors."""
    from huggingface_hub import HfApi
    info = HfApi().model_info(repo, revision=revision) if kind == "model" else HfApi().dataset_info(repo, revision=revision)
    metadata = {k: getattr(info, k, None) for k in ("sha", "id", "lastModified", "tags")}
    if hasattr(metadata["lastModified"], "isoformat"):
        metadata["lastModified"] = metadata["lastModified"].isoformat()
    return getattr(info, "sha", None), metadata


def manifest_path(args: argparse.Namespace) -> Path:
    return args.manifest


def build_manifest(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    from transformers import AutoTokenizer
    model_sha, model_info = hf_revision("model", MODEL, args.model_revision)
    if not model_sha: raise RuntimeError("unable to resolve immutable Hugging Face model commit")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=model_sha)
    # The ordinary 4K preflight deliberately freezes sources capable of the
    # planned 8K extrapolation.  Thus enabling --run-8k later cannot select a
    # different example after 4K results have been observed.
    required = {"narrative": [4096, 8192], "report": [4096, 8192], "qa": [4096]}
    sources: dict[str, Any] = {}
    for sequence, lengths in required.items():
        need = max(lengths)
        spec = SOURCES.get(sequence, {})
        requested_dataset_revision = spec.get("revision", "main")
        dataset_sha, dataset_info = (None, {}) if args.local_jsonl else hf_revision(
            "dataset", spec["dataset"], requested_dataset_revision,
        )
        if not args.local_jsonl and not dataset_sha: raise RuntimeError(f"unable to resolve dataset commit for {sequence}")
        if not args.local_jsonl and sequence == "qa" and dataset_sha != NARRATIVEQA_PARQUET_REVISION:
            raise RuntimeError(
                "NarrativeQA must resolve to the immutable Parquet-conversion commit "
                f"{NARRATIVEQA_PARQUET_REVISION}; got {dataset_sha!r}"
            )
        text, selected = resolve_text(sequence, args.local_jsonl, required_tokens=need, tokenizer=tokenizer, dataset_revision=dataset_sha)
        probe = selected.get("token_length_probe")
        if probe is None or not probe["verified_at_least_minimum"]:
            raise RuntimeError(f"source selection for {sequence} did not provide a successful bounded token-length probe")
        sources[sequence] = {"source": selected, "requested_dataset_revision": requested_dataset_revision,
                             "resolved_dataset_revision": dataset_sha, "dataset_revision": dataset_sha, "dataset_metadata": dataset_info,
                             "character_count": len(text), "token_length_probe": probe,
                             "required_context_lengths": lengths}
    return {"schema_version": 3, "architecture_config": {"path": str(args.config), "sha256": file_sha256(args.config), "architecture_version": config["architecture_version"]},
            "model": {"name": MODEL, "requested_revision": args.model_revision, "resolved_commit_sha": model_sha},
            "tokenizer": {"name_or_path": tokenizer.name_or_path, "revision": model_sha, "vocab_size": tokenizer.vocab_size},
            "model_metadata": model_info, "sources": sources}


def preflight(args: argparse.Namespace, config: dict[str, Any]) -> None:
    candidate = build_manifest(args, config)
    path = manifest_path(args)
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"existing benchmark manifest is incomplete or invalid: {path}") from exc
        if existing != candidate: raise ValueError("existing benchmark manifest differs; refusing to mix revisions or examples")
    else:
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(candidate, indent=2) + "\n")
    print(json.dumps({"status": "preflight_completed", "manifest": str(path), "sources": candidate["sources"]}, indent=2))


def load_manifest(args: argparse.Namespace, config: dict[str, Any], sequence: str, length: int) -> dict[str, Any]:
    path = manifest_path(args)
    if not path.exists(): raise RuntimeError("benchmark manifest absent; run --preflight before capture")
    manifest = json.loads(path.read_text())
    if manifest["architecture_config"]["sha256"] != file_sha256(args.config): raise ValueError("manifest architecture config hash mismatch")
    if sequence not in manifest["sources"] or length not in manifest["sources"][sequence]["required_context_lengths"]: raise ValueError("sequence/context absent from frozen manifest")
    return manifest


def cache_path(root: Path, sequence: str, length: int, layer: int) -> Path:
    return root / f"{sequence}_L{length}_layer{layer}_qkv.pt"


def validate_cache_artifact(path: Path, manifest: Path, sequence: str, length: int, layer: int) -> tuple[bool, str | None]:
    """Validate a capture cache against the exact frozen manifest."""
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        meta = loaded["metadata"]
        if meta["layer"] != layer or meta["token_count"] != length:
            return False, "cache layer or context identity mismatch"
        if meta.get("manifest_sha256") != file_sha256(manifest):
            return False, "cache manifest hash mismatch"
        if meta.get("sequence_source") is None:
            return False, "cache source metadata absent"
        if not all(name in loaded for name in ("query", "key", "value")):
            return False, "cache Q/K/V tensors absent"
    except Exception as exc:
        return False, str(exc)
    return True, None


def expected_shard_identities(sequence: str, length: int, layer: int, *, include_no_reserve: bool = False) -> set[tuple[int, int, int, float, str]]:
    positions = (2047, 3071, 4095) if length == 4096 else (4095, 6143, 8191)
    methods = BASELINE_METHODS + (("cascadekv_v1_no_reserve",) if include_no_reserve else ())
    return {(position, q_head, q_head // (Q_HEADS // KV_HEADS), fraction, method)
            for position in positions for q_head in range(Q_HEADS)
            for fraction in (.05, .10) for method in methods}


def validate_shard_artifact(path: Path, sequence: str, length: int, layer: int, *, include_no_reserve: bool = False,
                            manifest: Path | None = None) -> tuple[bool, str | None]:
    """Validate one evaluation shard using the benchmark's complete row schema."""
    try:
        payload = json.loads(path.read_text())
        rows = payload["rows"]
        if not isinstance(rows, list) or not rows:
            return False, "shard rows are empty"
        actual = set()
        for row in rows:
            if row["sequence"] != sequence or row["context_length"] != length or row["layer"] != layer:
                return False, "shard sequence/context/layer identity mismatch"
            actual.add((row["position"], row["q_head"], row["kv_head"], row["fraction"], row["method"]))
        expected = expected_shard_identities(sequence, length, layer, include_no_reserve=include_no_reserve)
        if len(rows) != len(expected) or actual != expected:
            return False, f"shard row identities/cardinality mismatch: got {len(rows)}, expected {len(expected)}"
        if manifest is not None and payload.get("cache_metadata", {}).get("manifest_sha256") != file_sha256(manifest):
            return False, "shard cache manifest hash mismatch"
    except Exception as exc:
        return False, str(exc)
    return True, None


def capture(args: argparse.Namespace, config: dict[str, Any]) -> None:
    from transformers import AutoModel, AutoTokenizer
    from experiments.qwen_partial_dot import capture_target_layer_post_rope_qkv_low_memory
    manifest = load_manifest(args, config, args.sequence, args.context_length)
    source = manifest["sources"][args.sequence]["source"]
    text, resolved = resolve_text(args.sequence, args.local_jsonl, required_tokens=args.context_length,
                                  tokenizer=AutoTokenizer.from_pretrained(MODEL, revision=manifest["model"]["resolved_commit_sha"]),
                                  dataset_revision=manifest["sources"][args.sequence]["dataset_revision"])
    if resolved["example_index"] != source["example_index"] or resolved["example_identifier"] != source["example_identifier"]:
        raise ValueError("source no longer resolves to frozen manifest example")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=manifest["model"]["resolved_commit_sha"])
    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.context_length).input_ids
    if ids.shape[1] != args.context_length:
        raise RuntimeError(f"requested {args.context_length} tokens, source supplied only {ids.shape[1]}; choose a different explicit example")
    model = AutoModel.from_pretrained(MODEL, revision=manifest["model"]["resolved_commit_sha"], dtype=torch.float16, low_cpu_mem_usage=True).eval()
    q, k, v = capture_target_layer_post_rope_qkv_low_memory(model, ids, args.layer)
    out = cache_path(args.cache_dir, args.sequence, args.context_length, args.layer); out.parent.mkdir(parents=True, exist_ok=True)
    scaling = float(model.layers[args.layer].self_attn.scaling)
    expected = 1.0 / math.sqrt(q.shape[-1])
    if not math.isclose(scaling, expected, rel_tol=1e-6, abs_tol=1e-8): raise AssertionError("Qwen3 attention.scaling differs from expected 1/sqrt(head_dim)")
    metadata = {"model_name": MODEL, "model_revision": manifest["model"]["resolved_commit_sha"], "manifest_sha256": file_sha256(manifest_path(args)), "sequence_source": source, "token_count": int(ids.shape[1]), "layer": args.layer, "attention_scaling": scaling, "tensor_shapes": {"q": list(q.shape), "k": list(k.shape), "v": list(v.shape)}, "storage_dtype": "float16", "tokenizer": {"name_or_path": tokenizer.name_or_path, "revision": manifest["model"]["resolved_commit_sha"], "vocab_size": tokenizer.vocab_size, "model_max_length": tokenizer.model_max_length}, "capture_convention": "Q/K post-q_norm,k_norm,RoPE; V projected pre-attention; separate fresh benchmark cache"}
    torch.save({"metadata": metadata, "query": q.half(), "key": k.half(), "value": v.half()}, out)
    print(json.dumps({"status": "completed", "cache": str(out), "metadata": metadata}, indent=2))


def mean_route(forest: StreamingLiftingForest, q: torch.Tensor, budget: int, reserve: set[int]) -> dict[str, Any]:
    """Frozen explanatory ablation: same forest, but exactly no variance term."""
    chosen, heap = set(reserve), []
    for root in forest.roots or []:
        mean = quantize_symmetric(forest.nodes[root].mean, 4, 16).dequantize()
        heapq.heappush(heap, (-float(q.float() @ mean), root, mean))
    details = expanded = 0
    while heap and len(chosen) < budget:
        _, idx, mean = heapq.heappop(heap); node = forest.nodes[idx]
        if node.is_leaf:
            for token in range(node.start, node.end):
                if token not in chosen: chosen.add(token)
                if len(chosen) == budget: break
        else:
            left, right = reconstruct_pair(mean, quantize_symmetric(node.detail, 4, 16).dequantize(), node.alpha)
            details += 1; expanded += 1
            heapq.heappush(heap, (-float(q.float() @ left), node.left, left)); heapq.heappush(heap, (-float(q.float() @ right), node.right, right))
    assert len(chosen) == budget
    return {"ids": chosen, "active_root_reads": len(forest.roots or []), "detail_reads": details, "expanded_internal_nodes": expanded, "variance_scalar_reads": 0, "variance_scalar_products": 0, "variance_scalar_adds": 0}


def traffic(route: dict[str, Any], n: int, budget: int, *, variance: bool, flat: bool = False, head_dim: int = 128) -> dict[str, float]:
    """Compatibility wrapper for the shared geometry-driven accounting core."""
    return core_traffic(route, n, budget, head_dim, variance=variance, flat=flat)


def attention_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scaling: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dense exact score/softmax/output reference, timed once per query."""
    return core_attention_reference(q, k, v, scaling)


def output_metrics(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, ids: set[int], reference: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None, scaling: float | None = None) -> dict[str, float]:
    return core_output_metrics(q, k, v, ids, reference=reference, scaling=scaling)


def quantized_flat(q: torch.Tensor, k: torch.Tensor, budget: int) -> set[int]:
    return core_quantized_flat(q, k, budget)


def evaluate(args: argparse.Namespace, config: dict[str, Any]) -> None:
    manifest = load_manifest(args, config, args.sequence, args.context_length)
    loaded = torch.load(cache_path(args.cache_dir, args.sequence, args.context_length, args.layer), map_location="cpu", weights_only=False)
    meta = loaded["metadata"]
    if meta["token_count"] != args.context_length: raise ValueError("cache context mismatch")
    if meta.get("manifest_sha256") != file_sha256(manifest_path(args)): raise ValueError("cache manifest mismatch")
    if meta.get("model_revision") != manifest["model"]["resolved_commit_sha"]: raise ValueError("cache model revision mismatch")
    scaling = float(meta["attention_scaling"])
    q_all, k_all, v_all = loaded["query"][0].float(), loaded["key"][0].float(), loaded["value"][0].float()
    positions = (2047, 3071, 4095) if args.context_length == 4096 else (4095, 6143, 8191)
    qpk = q_all.shape[0] // k_all.shape[0]; forests: dict[tuple[int, int], StreamingLiftingForest] = {}
    for kv in range(k_all.shape[0]):
        f = StreamingLiftingForest(mode=config["builder"], atom_size=config["atom_size"], window=4, group_counts=(1, 4))
        for start in range(0, args.context_length, 8):
            f.append_atom(k_all[kv, start:start + 8]); end = start + 7
            if end in positions: forests[end, kv] = copy.deepcopy(f)
    rows: list[dict[str, Any]] = []
    for pos in positions:
        n = pos + 1
        for qh in range(q_all.shape[0]):
            kv = qh // qpk; q, k, v = q_all[qh, pos], k_all[kv, :n], v_all[kv, :n]
            dense_start = time.perf_counter(); reference = attention_reference(q, k, v, scaling); dense_seconds = time.perf_counter() - dense_start
            for fraction in (0.05, 0.10):
                budget = candidate_budget(n, fraction); schedule, lam = frozen_routing_config(config, args.layer, fraction)
                reserve = reserve_ids(n, schedule["sink"], schedule["local"], budget)
                methods: dict[str, tuple[set[int], dict[str, float]]] = {}
                all_ids = set(range(n)); methods["dense_exact"] = (all_ids, {"total_k_bytes": float(n * 256), "selected_v_fp16_bytes": float(n * 256), "total_k_vs_dense_fp16_k": 1.})
                t = time.perf_counter(); flat = quantized_flat(q, k, budget); flat_time = time.perf_counter() - t
                methods["flat_q8k4"] = (flat, {**traffic({"active_root_reads": 0, "detail_reads": 0, "expanded_internal_nodes": 0, "variance_scalar_reads": 0}, n, budget, variance=False, flat=True), "python_cpu_seconds": flat_time})
                f = forests[pos, kv]
                t = time.perf_counter(); cascade = forest_route(f, q, budget, 4, lam, reserve); routing_time = time.perf_counter() - t
                methods["cascadekv_v1"] = (cascade["ids"], {**traffic(cascade, n, budget, variance=True), "python_cpu_routing_seconds": routing_time})
                t = time.perf_counter(); mean = mean_route(f, q, budget, reserve); mean_time = time.perf_counter() - t
                methods["cascadekv_mean_only"] = (mean["ids"], {**traffic(mean, n, budget, variance=False), "python_cpu_routing_seconds": mean_time})
                if args.include_no_reserve:
                    nr = forest_route(f, q, budget, 4, lam, set())
                    methods["cascadekv_v1_no_reserve"] = (nr["ids"], traffic(nr, n, budget, variance=True))
                for method, (ids, tdata) in methods.items():
                    start = time.perf_counter(); metrics = output_metrics(q, k, v, ids, reference, scaling); rerank = time.perf_counter() - start
                    rows.append({"sequence": args.sequence, "context_length": args.context_length, "layer": args.layer, "position": pos, "q_head": qh, "kv_head": kv, "fraction": fraction, "candidate_budget": budget, "method": method, "metrics": metrics, "traffic": {**tdata, "python_cpu_dense_score_seconds": dense_seconds, "python_cpu_authoritative_rerank_seconds": rerank if method != "dense_exact" else 0.0}})
    out = args.shard_dir / f"{args.sequence}_L{args.context_length}_layer{args.layer}.json"; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"experiment": "fresh_sequence_frozen_cascadekv_v1", "frozen_config": str(args.config), "cache_metadata": meta, "rows": rows}, indent=2) + "\n")
    print(json.dumps({"status": "completed", "shard": str(out), "rows": len(rows)}, indent=2))


def summarize_higher_better(values: list[float]) -> dict[str, float]:
    values = sorted(values)
    return {"mean": float(statistics.mean(values)), "median": float(statistics.median(values)), "p5": float(values[max(0, math.ceil(.05 * len(values)) - 1)]), "worst": float(values[0])}


def summarize_lower_better(values: list[float]) -> dict[str, float]:
    values = sorted(values)
    return {"mean": float(statistics.mean(values)), "median": float(statistics.median(values)), "p95": float(values[min(len(values) - 1, math.ceil(.95 * len(values)) - 1)]), "worst": float(values[-1])}


def summary_for_metric(metric: str, values: list[float]) -> dict[str, float]:
    return summarize_higher_better(values) if metric in HIGHER_BETTER else summarize_lower_better(values)


def development_reference() -> dict[str, Any]:
    """Use exact source results, never display-rounded development numbers."""
    return json.loads(Path("results/streaming_budget_schedule_gate.json").read_text())


def merge(args: argparse.Namespace, config: dict[str, Any]) -> None:
    expected = [(seq, length, layer) for seq in ("narrative", "report", "qa") for length in (4096,) for layer in LAYERS]
    if args.run_8k: expected += [(seq, 8192, layer) for seq in ("narrative", "report") for layer in LAYERS]
    expected_names = {f"{s}_L{n}_layer{l}.json" for s, n, l in expected}
    shard_paths = [p for p in sorted(args.shard_dir.glob("*_L*_layer*.json")) if p.name in expected_names]
    valid_paths, invalid_shards = [], []
    for path in shard_paths:
        stem = path.stem
        sequence, context, layer_text = stem.split("_L", 1)[0], stem.split("_L", 1)[1].split("_layer", 1)[0], stem.rsplit("_layer", 1)[1]
        manifest = getattr(args, "manifest", None)
        ok, reason = validate_shard_artifact(path, sequence, int(context), int(layer_text), include_no_reserve=getattr(args, "include_no_reserve", False), manifest=manifest if manifest and manifest.exists() else None)
        if ok: valid_paths.append(path)
        else: invalid_shards.append({"name": path.name, "reason": reason})
    shards = [json.loads(p.read_text()) for p in valid_paths]
    rows = [r for s in shards for r in s["rows"]]
    grouped: dict[tuple[str, int, float, str], list[dict[str, Any]]] = {}
    for row in rows: grouped.setdefault((row["sequence"], row["context_length"], row["fraction"], row["method"]), []).append(row)
    per_sequence = {}
    for key, members in grouped.items():
        per_sequence["|".join(map(str, key))] = {m: summary_for_metric(m, [x["metrics"][m] for x in members]) for m in HIGHER_BETTER + LOWER_BETTER} | {"traffic": {m: summarize_lower_better([x["traffic"].get(m, 0.) for x in members]) for m in ("routing_index_bytes", "candidate_sketch_bytes", "authoritative_k_bytes", "total_k_bytes", "selected_v_fp16_bytes", "total_k_vs_dense_fp16_k", "total_k_vs_flat_q8k4_plus_rerank")}, "python_cpu_prototype_seconds": {m: summarize_lower_better([x["traffic"].get(m, 0.) for x in members]) for m in ("python_cpu_dense_score_seconds", "python_cpu_seconds", "python_cpu_routing_seconds", "python_cpu_authoritative_rerank_seconds")}}
    pooled = {}
    pooled_groups: dict[tuple[int, float, str], list[dict[str, Any]]] = {}
    for row in rows: pooled_groups.setdefault((row["context_length"], row["fraction"], row["method"]), []).append(row)
    for key, members in pooled_groups.items():
        sequence_means = {seq: statistics.mean(x["metrics"]["top8_recall"] for x in members if x["sequence"] == seq) for seq in sorted({x["sequence"] for x in members})}
        pooled["|".join(map(str, key))] = {m: summary_for_metric(m, [x["metrics"][m] for x in members]) for m in HIGHER_BETTER + LOWER_BETTER} | {"worst_sequence_by_mean_top8_recall": min(sequence_means, key=sequence_means.get), "worst_sequence_mean_top8_recall": min(sequence_means.values())}
    robustness: dict[str, Any] = {}
    cascades = [r for r in rows if r["method"] == "cascadekv_v1"]
    for context in (4096, 8192):
        for fraction in (.05, .10):
            members = [r for r in cascades if r["context_length"] == context and r["fraction"] == fraction]
            if not members: continue
            key = f"L{context}|{fraction}"
            robustness[key] = {"worst_sequence": min(members, key=lambda r: r["metrics"]["top8_recall"])["sequence"], "worst_layer": min(members, key=lambda r: r["metrics"]["top8_recall"])["layer"], "worst_position": min(members, key=lambda r: r["metrics"]["top8_recall"])["position"], "worst_q_head": min(members, key=lambda r: r["metrics"]["top8_recall"])["q_head"], "worst_kv_head": min(members, key=lambda r: r["metrics"]["top8_recall"])["kv_head"], "lowest_top8_recall": min(r["metrics"]["top8_recall"] for r in members), "lowest_retained_attention_mass": min(r["metrics"]["relative_exact_attention_mass"] for r in members), "lowest_cosine_similarity": min(r["metrics"]["cosine_similarity"] for r in members), "highest_relative_l2_error": max(r["metrics"]["relative_l2_error"] for r in members), "by_layer": {}, "by_position": {}, "by_q_head": {}, "by_kv_head": {}}
            for field, destination in (("layer", "by_layer"), ("position", "by_position"), ("q_head", "by_q_head"), ("kv_head", "by_kv_head")):
                buckets: dict[Any, list[dict[str, Any]]] = {}
                for row in members: buckets.setdefault(row[field], []).append(row)
                robustness[key][destination] = {str(group): {m: summary_for_metric(m, [r["metrics"][m] for r in group_rows]) for m in HIGHER_BETTER + LOWER_BETTER} for group, group_rows in buckets.items()}
    # The validated artifact is authoritative.  Status files are execution
    # bookkeeping and can legitimately predate a successful resumed shard.
    # Only a failed status for an expected shard that is *not currently valid*
    # is a live failure.
    valid_names = {p.name for p in valid_paths}
    missing = sorted(expected_names - valid_names)
    failed = []
    status_suffix = ".evaluation.status.json"
    for status_path in args.shard_dir.glob(f"*{status_suffix}"):
        shard_name = status_path.name.removesuffix(status_suffix) + ".json"
        if shard_name not in expected_names or shard_name in valid_names:
            continue
        try:
            status = json.loads(status_path.read_text())
        except json.JSONDecodeError:
            # A corrupt status cannot override artifact validation, but is a
            # failure record when its expected shard is absent/invalid.
            failed.append(status_path.name)
            continue
        if status.get("status") != "completed":
            failed.append(status_path.name)
    complete = not missing and not failed and not invalid_shards
    payload = {"experiment": "fresh_sequence_benchmark", "architecture": config["architecture_version"], "frozen_config": str(args.config), "development_reference": development_reference(), "expected_shards": len(expected), "completed_shards": len(shards), "missing_shards": missing, "invalid_shards": invalid_shards, "failed_shards": failed, "benchmark_complete": complete, "paper_level_metrics_status": "COMPLETE" if complete else "PARTIAL", "row_count": len(rows), "per_sequence_context_budget_method": per_sequence, "pooled_context_budget_method": pooled, "frozen_v1_robustness": robustness, "raw_shard_directory": str(args.shard_dir), "traffic_denominators": {"dense": "n * 256 FP16-K bytes", "flat": "n * 80 Q8K4 bytes + budget * 256 authoritative FP16-K bytes"}, "note": "Development values are loaded exactly from their source result and are comparison-only; no value in this file is used for tuning."}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__); actions = p.add_mutually_exclusive_group(required=True); actions.add_argument("--capture", action="store_true"); actions.add_argument("--evaluate", action="store_true"); actions.add_argument("--merge", action="store_true"); actions.add_argument("--preflight", action="store_true"); actions.add_argument("--validate-cache", action="store_true"); actions.add_argument("--validate-shard", action="store_true")
    p.add_argument("--sequence", help="built-in source name, or a local JSONL row id with --local-jsonl"); p.add_argument("--local-jsonl", type=Path); p.add_argument("--context-length", type=int, choices=(4096, 8192)); p.add_argument("--layer", type=int, choices=LAYERS); p.add_argument("--config", type=Path, default=Path("configs/cascadekv_v1.json")); p.add_argument("--cache-dir", type=Path, default=Path("results/fresh_sequence_qkv_cache")); p.add_argument("--shard-dir", type=Path, default=Path("results/fresh_sequence_benchmark_shards")); p.add_argument("--output", type=Path, default=Path("results/fresh_sequence_benchmark.json")); p.add_argument("--manifest", type=Path, default=Path("results/fresh_sequence_benchmark_manifest.json")); p.add_argument("--model-revision", default="main"); p.add_argument("--include-no-reserve", action="store_true"); p.add_argument("--run-8k", action="store_true")
    a = p.parse_args(); config = load_frozen(a.config)
    if not (a.merge or a.preflight) and (a.sequence is None or a.context_length is None or a.layer is None): p.error("capture/evaluate/validation require --sequence, --context-length, and --layer")
    if a.capture: capture(a, config)
    elif a.evaluate: evaluate(a, config)
    elif a.merge: merge(a, config)
    elif a.validate_cache:
        ok, reason = validate_cache_artifact(cache_path(a.cache_dir, a.sequence, a.context_length, a.layer), a.manifest, a.sequence, a.context_length, a.layer)
        if not ok: print(reason, file=sys.stderr); raise SystemExit(1)
    elif a.validate_shard:
        path = a.shard_dir / f"{a.sequence}_L{a.context_length}_layer{a.layer}.json"
        ok, reason = validate_shard_artifact(path, a.sequence, a.context_length, a.layer, include_no_reserve=a.include_no_reserve, manifest=a.manifest)
        if not ok: print(reason, file=sys.stderr); raise SystemExit(1)
    else: preflight(a, config)


if __name__ == "__main__": main()
