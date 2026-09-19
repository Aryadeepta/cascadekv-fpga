#!/usr/bin/env python3
"""Development-only Q/K CascadeKV-v2 confidence-routing experiment."""
from __future__ import annotations
import argparse, copy, hashlib, heapq, json, math, statistics, sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cascadekv.adaptive_lifting import LiftingNode, StreamingLiftingForest, reconstruct_pair
from cascadekv.evaluation_core import (
    RouteState as CoreRouteState,
    advance_until_budget as core_advance_until_budget,
    hierarchy_route as core_hierarchy_route,
    initialize_route as core_initialize_route,
    route_snapshot as core_route_snapshot,
)
from cascadekv.quantize import quantize_symmetric
from experiments.fresh_sequence_benchmark import frozen_routing_config, load_frozen, mean_route, quantized_flat, traffic
from experiments.grouped_variance_gate import candidate_budget, group_energy, reserve_ids
from experiments.streaming_grouped_variance_gate import forest_route

FROZEN_PATH_PARTS = ("fresh_sequence_benchmark_v1_4k_frozen", "fresh_sequence_benchmark_v1_4k_diagnostics", "fresh_sequence_qkv_cache")
LAYERS, POSITIONS = (0, 7, 14, 21, 27), (2047, 3071, 4095)
AVAILABLE_FAILURES = {"attention_mass_lt_090": ("relative_exact_attention_mass", lambda x: x < .90), "top8_recall_lt_075": ("top8_recall", lambda x: x < .75)}
UNAVAILABLE_FAILURES = {"relative_l2_gt_020": "Q/K-only development cache has no V/output error"}

def assert_development_path(path: Path) -> None:
    if any(part in str(path).replace("\\", "/").lower() for part in FROZEN_PATH_PARTS): raise ValueError("frozen fresh-test artifacts are prohibited in the v2 development experiment")

def query_identity(row: dict[str, Any]) -> tuple[str, int, int, int, int]:
    """Physical query only: operating budget and quality are intentionally absent."""
    return (str(row["source_id"]), int(row["layer"]), int(row["position"]), int(row["q_head"]), int(row["kv_head"]))

def deterministic_split(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    calibration, validation = [], []
    for row in rows:
        identity = "|".join(map(str, query_identity(row)))
        (calibration if int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16) % 2 == 0 else validation).append(row)
    return calibration, validation

@dataclass
class RouteState:
    forest: StreamingLiftingForest; query: torch.Tensor; groups: int; weight: float; chosen: set[int]
    # (-priority, node_index, reconstructed_mean, priority). Heap order is not signal order.
    heap: list[tuple[float, int, torch.Tensor, float]]; admitted_leaf_priorities: list[float]; active_root_reads: int
    detail_reads: int = 0; expanded_internal_nodes: int = 0; variance_scalar_reads: int = 0; variance_scalar_products: int = 0; variance_scalar_adds: int = 0

def _priority(state: RouteState, index: int, mean: torch.Tensor) -> float:
    node = state.forest.nodes[index]; assert node.grouped_m2 is not None
    state.variance_scalar_reads += state.groups; state.variance_scalar_products += state.groups; state.variance_scalar_adds += state.groups - 1
    variance = node.grouped_m2[state.groups] / (node.count * (mean.numel() // state.groups))
    return float(state.query @ mean) + state.weight * math.sqrt(max(0., float(group_energy(state.query, state.groups) @ variance))) * math.sqrt(2 * math.log(max(node.count, 2)))

def initialize_route(forest: StreamingLiftingForest, query: torch.Tensor, groups: int, weight: float, reserve: set[int]) -> RouteState:
    assert forest.nodes is not None and forest.roots is not None
    state = RouteState(forest, query.float(), groups, weight, set(reserve), [], [], len(forest.roots))
    for root in forest.roots:
        mean = quantize_symmetric(forest.nodes[root].mean, 4, 16).dequantize(); score = _priority(state, root, mean)
        heapq.heappush(state.heap, (-score, root, mean, score))
    return state

def _signals(state: RouteState) -> dict[str, float | int]:
    # Paired records stay together through a real global priority sort.
    records: list[tuple[float, int, LiftingNode]] = sorted(((score, index, state.forest.nodes[index]) for _, index, _, score in state.heap), key=lambda r: r[0], reverse=True)
    top8, all_nodes = records[:8], [r[2] for r in records]
    variances = []
    for _, _, node in top8:
        assert node.grouped_m2 is not None; variances.append(float((node.grouped_m2[state.groups] / max(node.count, 1)).max()))
    # Leaf priorities are recorded at admission; no token-id -> node-index inference.
    weakest = min(state.admitted_leaf_priorities) if state.admitted_leaf_priorities else 0.; best = records[0][0] if records else 0.; margin = best - weakest if records else 0.
    return {"frontier_uncertainty_margin": margin, "frontier_uncertainty_margin_normalized": margin / (abs(weakest) + 1e-6) if records else 0., "weakest_selected_leaf_priority": weakest, "best_unresolved_frontier_priority": best, "frontier_max_grouped_variance_top8": max(variances, default=0.), "frontier_mean_grouped_variance_top8": statistics.mean(variances) if variances else 0., "unresolved_high_level_nodes_entire_frontier": sum(not n.is_leaf for n in all_nodes), "max_unresolved_subtree_size_entire_frontier": max((n.count for n in all_nodes), default=0), "mean_unresolved_subtree_size_entire_frontier": statistics.mean([n.count for n in all_nodes]) if all_nodes else 0., "detail_reads": state.detail_reads}

def route_snapshot(state: RouteState) -> dict[str, Any]:
    return {"ids": set(state.chosen), "signals": _signals(state), "active_root_reads": state.active_root_reads, "detail_reads": state.detail_reads, "expanded_internal_nodes": state.expanded_internal_nodes, "variance_scalar_reads": state.variance_scalar_reads, "variance_scalar_products": state.variance_scalar_products, "variance_scalar_adds": state.variance_scalar_adds}

def advance_until_budget(state: RouteState, budget: int) -> dict[str, Any]:
    if budget < len(state.chosen): raise ValueError("cannot resume a route to a smaller budget")
    while state.heap and len(state.chosen) < budget:
        _, index, mean, score = heapq.heappop(state.heap); node = state.forest.nodes[index]
        if node.is_leaf:
            for token in range(node.start, node.end):
                if token not in state.chosen:
                    state.chosen.add(token); state.admitted_leaf_priorities.append(score)
                    if len(state.chosen) == budget: break
            continue
        assert node.left is not None and node.right is not None and node.detail is not None and node.alpha is not None
        left, right = reconstruct_pair(mean, quantize_symmetric(node.detail, 4, 16).dequantize(), node.alpha); state.detail_reads += 1; state.expanded_internal_nodes += 1
        for child, child_mean in ((node.left, left), (node.right, right)):
            child_score = _priority(state, child, child_mean); heapq.heappush(state.heap, (-child_score, child, child_mean, child_score))
    if len(state.chosen) != budget: raise AssertionError("routing frontier exhausted before requested candidate budget")
    return route_snapshot(state)

def confidence_route(forest: StreamingLiftingForest, query: torch.Tensor, budget: int, groups: int, weight: float, reserve: set[int]) -> dict[str, Any]:
    """One-shot wrapper, preserving frozen-v1 traversal at a fixed budget."""
    return advance_until_budget(initialize_route(forest, query, groups, weight, reserve), budget)

def adaptive_traffic(routes: list[dict[str, Any]], n: int, final_budget: int, *, fallback: bool = False) -> dict[str, float]:
    base = {key: sum(route[key] for route in routes) for key in ("active_root_reads", "detail_reads", "expanded_internal_nodes", "variance_scalar_reads")}
    result = traffic(base, n, final_budget, variance=True, flat=False)
    if fallback:
        # This pays complete flat Q8K4 in addition to completed hierarchy work.
        result["candidate_sketch_bytes"] += n * 80; result["total_k_bytes"] += n * 80
        result["total_k_vs_dense_fp16_k"] = result["total_k_bytes"] / (n * 256); result["total_k_vs_flat_q8k4_plus_rerank"] = result["total_k_bytes"] / (n * 80 + final_budget * 256)
    result["total_kv_bytes"] = result["total_k_bytes"] + result["selected_v_fp16_bytes"]
    result["total_kv_vs_dense"] = result["total_kv_bytes"] / (n * 512); result["total_kv_vs_flat"] = result["total_kv_bytes"] / (n * 80 + final_budget * 512)
    return result

def flat_traffic(n: int, budget: int) -> dict[str, float]:
    """Standalone flat Q8K4: one complete sketch scan plus authoritative rerank."""
    result = traffic({"active_root_reads": 0, "detail_reads": 0, "expanded_internal_nodes": 0, "variance_scalar_reads": 0}, n, budget, variance=False, flat=True)
    result["total_kv_bytes"] = result["total_k_bytes"] + result["selected_v_fp16_bytes"]
    result["total_kv_vs_dense"] = result["total_kv_bytes"] / (n * 512); result["total_kv_vs_flat"] = result["total_kv_bytes"] / (n * 80 + budget * 512)
    return result

# Preserve this historical module's public entry points, but route all new and
# legacy callers through the shared production mechanics.  The local bodies
# above remain the frozen readability oracle for old development reports.
RouteState = CoreRouteState
initialize_route = core_initialize_route
advance_until_budget = core_advance_until_budget
route_snapshot = core_route_snapshot
confidence_route = core_hierarchy_route

def _auc(x: list[float], labels: list[bool]) -> float | None:
    pos, neg = [v for v, y in zip(x, labels) if y], [v for v, y in zip(x, labels) if not y]
    return None if not pos or not neg else sum((p > n) + .5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))
def _corr(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2: return None
    mx, my = statistics.mean(x), statistics.mean(y); den = math.sqrt(sum((a-mx)**2 for a in x) * sum((b-my)**2 for b in y))
    return sum((a-mx)*(b-my) for a, b in zip(x, y)) / den if den else None
def _ranks(values: list[float]) -> list[float]:
    order, out, start = sorted(range(len(values)), key=values.__getitem__), [0.] * len(values), 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]: end += 1
        for i in order[start:end]: out[i] = (start + end - 1) / 2 + 1
        start = end
    return out

def predictive_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in sorted(rows[0]["signals"]) if rows else []:
        events: dict[str, Any] = {}
        for event, (metric, predicate) in AVAILABLE_FAILURES.items():
            usable = [r for r in rows if r["metrics"].get(metric) is not None]; values = [float(r["signals"][name]) for r in usable]; labels = [predicate(r["metrics"][metric]) for r in usable]; raw = _auc(values, labels)
            events[event] = {"pearson": _corr(values, [float(x) for x in labels]), "spearman": _corr(_ranks(values), _ranks([float(x) for x in labels])), "raw_auroc": raw, "orientation_adjusted_auroc": max(raw, 1-raw) if raw is not None else None, "positive_count": sum(labels), "count": len(labels)}
        events.update({event: {"available": False, "reason": reason} for event, reason in UNAVAILABLE_FAILURES.items()}); result[name] = {"failure_events": events}
    return result

def select_signal(stats: dict[str, Any]) -> tuple[str, str, float]:
    ranked = []
    for name, entry in stats.items():
        events = entry["failure_events"]; adjusted = [events[e]["orientation_adjusted_auroc"] for e in AVAILABLE_FAILURES if events[e]["orientation_adjusted_auroc"] is not None]; raw = [events[e]["raw_auroc"] for e in AVAILABLE_FAILURES if events[e]["raw_auroc"] is not None]
        if adjusted: ranked.append((statistics.mean(adjusted), name, "high_is_risky" if statistics.mean(raw) >= .5 else "low_is_risky"))
    if not ranked: raise ValueError("calibration has no usable positive/negative failure labels")
    return sorted(ranked, key=lambda item: (-item[0], item[1]))[0][1], sorted(ranked, key=lambda item: (-item[0], item[1]))[0][2], sorted(ranked, key=lambda item: (-item[0], item[1]))[0][0]

def fit_threshold(calibration: list[dict[str, Any]], signal: str, polarity: str = "high_is_risky") -> float:
    bad = [r["signals"][signal] for r in calibration if r["metrics"]["relative_exact_attention_mass"] < .90 or r["metrics"]["top8_recall"] < .75]
    return float(statistics.median(bad)) if bad else (float("inf") if polarity == "high_is_risky" else float("-inf"))
def _trigger(value: float, threshold: float, polarity: str) -> bool: return value >= threshold if polarity == "high_is_risky" else value <= threshold
def _route_metrics(q: torch.Tensor, k: torch.Tensor, ids: set[int]) -> dict[str, float | None]:
    logits = k.float() @ q.float(); p = torch.softmax(logits / math.sqrt(q.numel()), 0)
    return {"relative_exact_attention_mass": float(p[torch.tensor(sorted(ids))].sum()), "top8_recall": len(ids & set(logits.topk(8).indices.tolist())) / 8, "relative_l2_error": None, "cosine_similarity": None}

def progressive_policy(case: dict[str, Any], threshold: float, signal: str, polarity: str) -> dict[str, Any]:
    fractions = (.05, .075, .10, .15) if case["fraction"] == .05 else (.10, .15, .20); state = initialize_route(case["forest"], case["q"], 4, case["lambda"], case["reserve"]); stages = []
    for fraction in fractions:
        snapshot = advance_until_budget(state, candidate_budget(case["n"], fraction)); stages.append(snapshot)
        if not _trigger(float(snapshot["signals"][signal]), threshold, polarity): break
    final_fraction = fractions[len(stages)-1]
    return {"ids": stages[-1]["ids"], "final_fraction": final_fraction, "refined": len(stages) > 1, "traffic": adaptive_traffic([stages[-1]], case["n"], candidate_budget(case["n"], final_fraction)), "stages": stages}
def fallback_policy(case: dict[str, Any], threshold: float, signal: str, polarity: str) -> dict[str, Any]:
    budget = candidate_budget(case["n"], case["fraction"]); route = confidence_route(case["forest"], case["q"], budget, 4, case["lambda"], case["reserve"]); fallback = _trigger(float(route["signals"][signal]), threshold, polarity)
    return {"ids": quantized_flat(case["q"], case["k"], budget) if fallback else route["ids"], "fallback": fallback, "traffic": adaptive_traffic([route], case["n"], budget, fallback=fallback)}
def _mean_metrics(points: list[dict[str, Any]]) -> dict[str, float | None]:
    return {m: (statistics.mean(x) if (x := [p["metrics"][m] for p in points if p["metrics"][m] is not None]) else None) for m in ("relative_exact_attention_mass", "top8_recall", "relative_l2_error", "cosine_similarity")}
def _rates(points: list[dict[str, Any]]) -> dict[str, float]:
    groups: dict[str, list[bool]] = defaultdict(list)
    for point in points:
        c = point["case"]; groups[f"layer{c['layer']}|q{c['q_head']}|kv{c['kv_head']}"] .append(point["triggered"])
    return {key: statistics.mean(values) for key, values in groups.items()}
def _mean_traffic(points: list[dict[str, Any]]) -> dict[str, float]:
    keys = ("routing_index_bytes", "candidate_sketch_bytes", "authoritative_k_bytes", "selected_v_fp16_bytes", "total_k_bytes", "total_kv_bytes", "total_k_vs_dense_fp16_k", "total_k_vs_flat_q8k4_plus_rerank", "total_kv_vs_dense", "total_kv_vs_flat")
    return {key: statistics.mean(p["traffic"][key] for p in points) for key in keys}

def policy_curve(cases: list[dict[str, Any]], calibration: list[dict[str, Any]], signal: str, polarity: str) -> dict[str, list[dict[str, Any]]]:
    values = sorted(float(r["signals"][signal]) for r in calibration); inf = float("inf") if polarity == "high_is_risky" else float("-inf")
    thresholds = sorted({values[min(len(values)-1, int(q*(len(values)-1)))] for q in (.25, .5, .75)} | {inf}, reverse=polarity == "high_is_risky") if values else [inf]
    curves: dict[str, list[dict[str, Any]]] = {"frozen_v1": [], "mean_only": [], "flat_q8k4": [], "progressive_incremental_refinement": [], "selective_flat_fallback": []}
    zero = {k: 0 for k in ("active_root_reads", "detail_reads", "expanded_internal_nodes", "variance_scalar_reads")}
    for base in (.05, .10):
        subset = [c for c in cases if c["fraction"] == base]; fixed, mean_only, flat = [], [], []
        for case in subset:
            budget = candidate_budget(case["n"], base); route = confidence_route(case["forest"], case["q"], budget, 4, case["lambda"], case["reserve"])
            fixed.append({"metrics": _route_metrics(case["q"], case["k"], route["ids"]), "traffic": adaptive_traffic([route], case["n"], budget), "triggered": False, "case": case})
            mean = mean_route(case["forest"], case["q"], budget, case["reserve"])
            mean_only.append({"metrics": _route_metrics(case["q"], case["k"], mean["ids"]), "traffic": adaptive_traffic([mean], case["n"], budget), "triggered": False, "case": case})
            flat.append({"metrics": _route_metrics(case["q"], case["k"], quantized_flat(case["q"], case["k"], budget)), "traffic": flat_traffic(case["n"], budget), "triggered": False, "case": case})
        for name, points in (("frozen_v1", fixed), ("mean_only", mean_only), ("flat_q8k4", flat)): curves[name].append({"base_fraction": base, "metrics": _mean_metrics(points), "traffic": _mean_traffic(points), "trigger_rate": 0., "trigger_rate_by_layer_qhead_kvhead": _rates(points)})
        for threshold in thresholds:
            progressive, fallback = [], []
            for case in subset:
                p, f = progressive_policy(case, threshold, signal, polarity), fallback_policy(case, threshold, signal, polarity)
                progressive.append({"metrics": _route_metrics(case["q"], case["k"], p["ids"]), "traffic": p["traffic"], "triggered": p["refined"], "case": case}); fallback.append({"metrics": _route_metrics(case["q"], case["k"], f["ids"]), "traffic": f["traffic"], "triggered": f["fallback"], "case": case})
            for name, points in (("progressive_incremental_refinement", progressive), ("selective_flat_fallback", fallback)): curves[name].append({"base_fraction": base, "calibration_only_threshold": threshold, "metrics": _mean_metrics(points), "traffic": _mean_traffic(points), "trigger_rate": statistics.mean(p["triggered"] for p in points), "trigger_rate_by_layer_qhead_kvhead": _rates(points)})
    return curves

def _event_stats(rows: list[dict[str, Any]], signal: str, polarity: str = "high_is_risky") -> dict[str, Any]:
    result = {}
    for event, (metric, predicate) in AVAILABLE_FAILURES.items():
        usable = [r for r in rows if r["metrics"].get(metric) is not None]
        values = [float(r["signals"][signal]) for r in usable]
        labels = [predicate(r["metrics"][metric]) for r in usable]
        raw = _auc(values, labels)
        oriented = raw if polarity == "high_is_risky" else (1 - raw if raw is not None else None)
        result[event] = {"raw_auroc": raw, "orientation_adjusted_auroc": max(raw, 1-raw) if raw is not None else None,
                         "polarity_oriented_auroc": oriented, "positive_count": sum(labels), "count": len(labels)}
    return result

def structural_confound_audit(calibration: list[dict[str, Any]], signals: list[str]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for signal in signals:
        by_layer, by_position, correlations = {}, {}, {}
        for layer in LAYERS:
            subset = [r for r in calibration if r["layer"] == layer]
            by_layer[str(layer)] = _event_stats(subset, signal)
            correlations[str(layer)] = {metric: {"pearson": _corr([float(r["signals"][signal]) for r in subset], [float(r["metrics"][metric]) for r in subset]),
                                                   "spearman": _corr(_ranks([float(r["signals"][signal]) for r in subset]), _ranks([float(r["metrics"][metric]) for r in subset]))}
                                        for metric in ("relative_exact_attention_mass", "top8_recall")}
        for position in POSITIONS:
            by_position[str(position)] = _event_stats([r for r in calibration if r["position"] == position], signal)
        macro = {}
        for event in AVAILABLE_FAILURES:
            values = [x[event]["orientation_adjusted_auroc"] for x in by_layer.values() if x[event]["orientation_adjusted_auroc"] is not None]
            macro[event] = statistics.mean(values) if values else None
        report[signal] = {"overall": _event_stats(calibration, signal), "per_layer": by_layer, "macro_mean_valid_per_layer_adjusted_auroc": macro,
                          "per_position": by_position, "within_layer_signal_quality_correlations": correlations}
    lolo = {}
    for held in LAYERS:
        train, test = [r for r in calibration if r["layer"] != held], [r for r in calibration if r["layer"] == held]
        selected, polarity, score = select_signal(predictive_statistics(train))
        lolo[str(held)] = {"selected_signal": selected, "polarity": polarity, "training_adjusted_auroc_mean": score,
                           "held_out_predictor_statistics": _event_stats(test, selected, polarity)}
    return {"signals": report, "leave_one_layer_out": lolo}

def structural_baselines(calibration: list[dict[str, Any]], validation: list[dict[str, Any]]) -> dict[str, Any]:
    def score(rows: list[dict[str, Any]], keys: tuple[str, ...], metric: str, predicate: Any) -> list[float]:
        totals: dict[tuple[Any, ...], list[bool]] = defaultdict(list)
        for row in calibration: totals[tuple(row[k] for k in keys)].append(predicate(row["metrics"][metric]))
        global_rate = statistics.mean(predicate(r["metrics"][metric]) for r in calibration)
        return [statistics.mean(totals.get(tuple(row[k] for k in keys), [global_rate])) for row in rows]
    out = {}
    for name, keys in (("L", ("layer",)), ("LQ", ("layer", "q_head")), ("LP", ("layer", "position"))):
        out[name] = {}
        for event, (metric, predicate) in AVAILABLE_FAILURES.items():
            labels = [predicate(r["metrics"][metric]) for r in validation]; raw = _auc(score(validation, keys, metric, predicate), labels)
            out[name][event] = {"validation_auroc": raw, "validation_adjusted_auroc": max(raw, 1-raw) if raw is not None else None}
    return out

def oracle_curve(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for base in (.05, .10):
        points = []
        for case in [c for c in cases if c["fraction"] == base]:
            budget = candidate_budget(case["n"], base); route = confidence_route(case["forest"], case["q"], budget, 4, case["lambda"], case["reserve"])
            points.append({"case": case, "route": route, "base_metrics": _route_metrics(case["q"], case["k"], route["ids"])})
        ranked = sorted(points, key=lambda p: p["base_metrics"]["relative_exact_attention_mass"])
        for target in (0., .25, .50, .75, 1.):
            count = round(target * len(ranked)); selected = {id(p) for p in ranked[:count]}; evaluated = []
            for point in points:
                case, budget, fallback = point["case"], candidate_budget(point["case"]["n"], base), id(point) in selected
                ids = quantized_flat(case["q"], case["k"], budget) if fallback else point["route"]["ids"]
                evaluated.append({"metrics": _route_metrics(case["q"], case["k"], ids), "traffic": adaptive_traffic([point["route"]], case["n"], budget, fallback=fallback), "triggered": fallback, "case": case})
            out.append({"policy": "ORACLE flat fallback (diagnostic; ranked by true frozen-v1 mass)", "base_fraction": base, "target_trigger_rate": target,
                        "trigger_rate": statistics.mean(x["triggered"] for x in evaluated), "metrics": _mean_metrics(evaluated), "traffic": _mean_traffic(evaluated)})
    return out

def historical_consistency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    path = Path("results/streaming_budget_schedule_gate.json")
    old = json.loads(path.read_text())
    findings = {"historical_path": str(path), "identity": "same cache files/layers, binary_counter G4, frozen lambdas, reserve schedule, candidate-budget semantics; historical subset is positions 3071/4095 and 8 evenly-spaced q-heads", "comparable": "route identity and top-8 recall are comparable; attention mass is not directly comparable", "attention_mass_semantics": "historical mass() applies softmax(raw q@k); this audit applies transformer-scaled softmax((q@k)/sqrt(head_dim)). The candidate routes are the same, but these retained-mass values intentionally differ.", "comparisons": {}}
    for layer in LAYERS:
        # The historical harness selected these eight physical Q heads.
        qcount = max(r["q_head"] for r in rows if r["layer"] == layer) + 1
        qheads = set(torch.linspace(0, qcount - 1, 8).round().long().tolist())
        for fraction, key in ((.05, "five_percent"), (.10, "ten_percent_reused")):
            subset = [r for r in rows if r["layer"] == layer and r["position"] in (3071, 4095) and r["q_head"] in qheads and r["fraction"] == fraction]
            current = {"mass": statistics.mean(r["metrics"]["relative_exact_attention_mass"] for r in subset), "top8_recall": statistics.mean(r["metrics"]["top8_recall"] for r in subset), "count": len(subset)}
            prior = old["builders"]["binary_counter"]["layers"][str(layer)][key]
            if key == "ten_percent_reused": prior = prior["measurement"]
            expected = {"mass": prior["relative_exact_attention_mass"]["mean"], "top8_recall": prior["top8_candidate_recall"]["mean"]}
            findings["comparisons"][f"layer{layer}_{fraction}"] = {"current": current, "historical": expected,
                "mass_delta": current["mass"] - expected["mass"], "top8_recall_delta": current["top8_recall"] - expected["top8_recall"]}
    return findings

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--cache-dir", type=Path, default=Path("results/qk_cache")); p.add_argument("--output", type=Path, default=Path("results/confidence_routing_v2_dev.json")); p.add_argument("--layers", default=",".join(map(str, LAYERS))); p.add_argument("--max-queries", type=int, default=0); a = p.parse_args(); assert_development_path(a.cache_dir); torch.set_num_threads(min(4, torch.get_num_threads()))
    config, rows, cases, mismatches = load_frozen(Path("configs/cascadekv_v1.json")), [], [], []
    for layer in (int(x) for x in a.layers.split(",")):
        path = a.cache_dir / f"qwen3_0.6b_layer{layer}_4096.pt"; assert_development_path(path); data = torch.load(path, map_location="cpu", weights_only=False); queries, keys = data["query"][0].float(), data["key"][0].float(); qpk = queries.shape[0] // keys.shape[0]; forests = {}
        for kv in range(keys.shape[0]):
            forest = StreamingLiftingForest(mode="binary_counter", atom_size=8, window=4, group_counts=(4,))
            for start in range(0, 4096, 8):
                forest.append_atom(keys[kv, start:start+8])
                if start + 7 in POSITIONS: forests[start+7, kv] = copy.deepcopy(forest)
        for pos in POSITIONS:
            for qh in range(queries.shape[0]):
                kv, q, n, k = qh // qpk, queries[qh, pos], pos+1, keys[qh//qpk, :pos+1]; logits = k @ q; probs = torch.softmax(logits / math.sqrt(q.numel()), 0)
                for fraction in (.05, .10):
                    budget = candidate_budget(n, fraction); schedule, weight = frozen_routing_config(config, layer, fraction); reserve = reserve_ids(n, schedule["sink"], schedule["local"], budget); route = confidence_route(forests[pos,kv], q, budget, 4, weight, reserve); metrics = {"relative_exact_attention_mass": float(probs[torch.tensor(sorted(route["ids"]))].sum()), "top8_recall": len(route["ids"] & set(logits.topk(8).indices.tolist())) / 8, "relative_l2_error": None, "cosine_similarity": None}; source = f"legacy_development_qk_cache:{path.name}:sha256={hashlib.sha256(path.read_bytes()).hexdigest()}"
                    frozen = forest_route(forests[pos,kv], q, budget, 4, weight, reserve)
                    fields = ("ids", "active_root_reads", "detail_reads", "expanded_internal_nodes", "variance_scalar_reads", "variance_scalar_products", "variance_scalar_adds")
                    if any(route[field] != frozen[field] for field in fields): mismatches.append({"layer": layer, "position": pos, "q_head": qh, "fraction": fraction})
                    rows.append({"source_id": source, "layer": layer, "position": pos, "q_head": qh, "kv_head": kv, "fraction": fraction, "candidate_budget": budget, "signals": route["signals"], "metrics": metrics, "v1_route": {**route, "ids": sorted(route["ids"])}}); cases.append({"forest": forests[pos,kv], "q": q, "k": k, "n": n, "lambda": weight, "reserve": reserve, "layer": layer, "position": pos, "q_head": qh, "kv_head": kv, "fraction": fraction})
                    if a.max_queries and len(rows) >= a.max_queries: break
                if a.max_queries and len(rows) >= a.max_queries: break
            if a.max_queries and len(rows) >= a.max_queries: break
        if a.max_queries and len(rows) >= a.max_queries: break
    if mismatches: raise RuntimeError(f"route-equivalence mismatch_count={len(mismatches)}; refusing statistical interpretation")
    for row, case in zip(rows, cases):
        weak, best, qnorm = float(row["signals"]["weakest_selected_leaf_priority"]), float(row["signals"]["best_unresolved_frontier_priority"]), float(case["q"].norm())
        row["signals"].update({"weakest_over_abs_best_frontier": weak / (abs(best) + 1e-6),
                               "frontier_leaf_signed_gap_normalized": (best - weak) / (abs(best) + abs(weak) + 1e-6),
                               "query_norm_normalized_weakest_leaf_priority": weak / (qnorm + 1e-6),
                               "query_norm_normalized_best_frontier_priority": best / (qnorm + 1e-6)})
    calibration, validation = deterministic_split(rows); stats = predictive_statistics(calibration); signal, polarity, score = select_signal(stats); calibration_ids = {id(r) for r in calibration}; validation_cases = [c for r, c in zip(rows, cases) if id(r) not in calibration_ids]
    required = [signal, "best_unresolved_frontier_priority", "frontier_uncertainty_margin", "frontier_uncertainty_margin_normalized"]
    output = {"schema_version": 3, "experiment": "confidence_routing_v2_qk_confidence_validity_audit", "frozen_test_files_loaded": False, "development_sources": sorted({r["source_id"] for r in rows}), "route_equivalence": {"mismatch_count": 0, "fields": ["candidate IDs", "active_root_reads", "detail_reads", "expanded_internal_nodes", "variance_scalar_reads", "variance_scalar_products", "variance_scalar_adds"]}, "split": {"identity_definition": ["source_id", "layer", "position", "q_head", "kv_head"], "method": "sha256 physical-query identity parity", "calibration_unique_query_count": len({query_identity(r) for r in calibration}), "validation_unique_query_count": len({query_identity(r) for r in validation}), "calibration_row_count": len(calibration), "validation_row_count": len(validation)}, "limitations": ["Q/K-only development cache: relative_l2_error and cosine_similarity are unavailable (null), never inferred."], "historical_development_result_consistency": historical_consistency(rows), "signal_predictive_statistics_calibration": stats, "policy_signal_selection_calibration_only": {"rule": "highest mean orientation-adjusted AUROC across mass<0.90 and top8<0.75; lexical signal-name tie-break", "selected_signal": signal, "polarity": polarity, "calibration_score": score}, "selected_signal_validation_diagnostic": _event_stats(validation, signal, polarity), "structural_confound_audit_calibration_only": structural_confound_audit(calibration, required), "structural_only_baselines_fit_calibration_apply_validation": structural_baselines(calibration, validation), "normalized_signal_diagnostics_calibration_only": {name: _event_stats(calibration, name) for name in ("weakest_over_abs_best_frontier", "frontier_leaf_signed_gap_normalized", "query_norm_normalized_weakest_leaf_priority", "query_norm_normalized_best_frontier_priority")}, "threshold_candidates": {"source": "calibration signal quantiles only", "signal": signal, "polarity": polarity}, "pareto_operating_points_development_validation": policy_curve(validation_cases, calibration, signal, polarity), "oracle_flat_fallback_development_validation_diagnostic_only": oracle_curve(validation_cases), "raw_rows": rows}
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(output, indent=2) + "\n"); print(json.dumps({"output": str(a.output), "rows": len(rows), "calibration": len(calibration), "validation": len(validation), "signal": signal, "polarity": polarity}, indent=2))
if __name__ == "__main__": main()
