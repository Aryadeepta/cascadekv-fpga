import copy
import math

import pytest

from cascadekv import phi35_8k_c5 as c5
from cascadekv import phi35_8k_c6 as c6


def stat(loss, traffic=1, cosine=.99):
    return {"relative_l2": loss, "total_kv_bytes": traffic, "cosine": cosine}


def groups(extra=False):
    actions = c6.ACTIONS if extra else c5.ACTIONS
    return [{a: stat(1) for a in actions} for _ in range(160)]


def test_a0_a9_are_exact_c5_inheritance():
    assert {a: c6.action_definitions()[a] for a in c5.ACTIONS} == c5.action_definitions()


def test_a10_a11_exact():
    assert c6.action_definitions()["A10"] == {"kind": "hierarchy", "candidate_fraction": .25, "routing_profile": "phi35-8k-depth-transfer-qwen5-v1"}
    assert c6.action_definitions()["A11"] == {"kind": "hierarchy", "candidate_fraction": .25, "routing_profile": "phi35-8k-depth-transfer-qwen10-v1"}


def test_domains_and_schedule_validation_every_forbidden_pair():
    assert c6.allowed_actions(0) == c6.ACTIONS
    for layer in (8, 16, 24, 31):
        assert c6.allowed_actions(layer) == c6.BASE_ACTIONS
        for forbidden in ("A10", "A11"):
            table = ["A0"] * 160
            table[c6.c4.LAYERS.index(layer) * 32] = forbidden
            with pytest.raises(c6.C6Error): c6.validate_schedule(table)


@pytest.mark.parametrize("case", ["generous", "tight", "boundary", "strict", "infeasible", "rel", "cos", "traffic", "lexical"])
def test_c5_c6_masked_dp_parity(case):
    base = groups()
    if case == "tight":
        for row in base: row.update(A0=stat(5, 2), A1=stat(1, 1))
        cap = 160000
    elif case == "boundary":
        for row in base: row["A0"] = stat(1, 1.001)
        cap = 160160
    elif case == "strict":
        for row in base: row["A0"] = stat(1, 1)
        cap = 159999
    elif case == "infeasible":
        for row in base:
            for action in c5.ACTIONS: row[action] = stat(1, 2)
        cap = 1
    elif case == "rel":
        for row in base: row["A1"] = stat(.5, 1)
        cap = 160000
    elif case == "cos":
        for row in base: row.update(A0=stat(1, 1, .8), A1=stat(1, 1, .9))
        cap = 160000
    elif case == "traffic":
        for row in base: row.update(A0=stat(1, 2), A1=stat(1, 1))
        cap = 320000
    else:
        cap = 160000
    extended = [{**row, "A10": stat(99, 99), "A11": stat(99, 99)} for row in base]
    assert c6.optimize_action_cells(extended, cap, actions=c6.ACTIONS, domains=(c6.BASE_ACTIONS,) * 160) == c5.optimize_action_cells(base, cap)


@pytest.mark.parametrize("layer", [8, 16, 24, 31])
@pytest.mark.parametrize("forbidden", ["A10", "A11"])
@pytest.mark.parametrize("axis", ["rel", "cos", "traffic", "all"])
def test_forbidden_actions_cannot_affect_masked_dp(layer, forbidden, axis):
    data = groups(True); index = c6.c4.LAYERS.index(layer) * 32
    if axis in ("rel", "all"): data[index][forbidden]["relative_l2"] = -100
    if axis in ("cos", "all"): data[index][forbidden]["cosine"] = 100
    if axis in ("traffic", "all"): data[index][forbidden]["total_kv_bytes"] = 0
    masked = c6.optimize_action_cells(data, 160000, domains=c6.action_domain_by_cell())
    control = copy.deepcopy(data); control[index][forbidden] = stat(10_000, 10_000, -10_000)
    assert masked == c6.optimize_action_cells(control, 160000, domains=c6.action_domain_by_cell())
    assert masked[-1][index] != forbidden


@pytest.mark.parametrize("action", ["A10", "A11"])
def test_layer_zero_new_actions_work_normally(action):
    data = groups(True); data[0][action] = stat(0, .5, 1)
    result = c6.optimize_action_cells(data, 160000, domains=c6.action_domain_by_cell())
    assert result and result[-1][0] == action


def observation_rows(sources=None):
    sources = sources or [f"{family}:{index}:development" for family in c6.c4.FAMILIES for index in c6.DEVELOPMENT[family]]
    return [{"source": source, "layer": layer, "position": position, "head": head,
             "metrics": {"cosine_similarity": .99, "relative_l2_error": .01}, "traffic": {"total_kv_bytes": 1.0}}
            for source in sources for layer in c6.c4.LAYERS for position in c6.c4.POSITIONS for head in c6.c4.HEADS]


@pytest.mark.parametrize("action", c6.ACTIONS)
def test_exact_7200_inventory_accepts_all_actions(action):
    assert len(c6.validate_observation_inventory(observation_rows(), action=action, role="development")) == 7200


@pytest.mark.parametrize("mutation", ["delete", "duplicate", "family", "index", "report20", "layer", "position", "headneg", "headhigh", "nan", "inf"])
def test_7200_inventory_rejects_adversarial_mutations(mutation):
    changed = copy.deepcopy(observation_rows())
    if mutation == "delete": changed.pop()
    elif mutation == "duplicate": changed[-1] = copy.deepcopy(changed[0])
    elif mutation == "family": changed[0]["source"] = "bad:1:development"
    elif mutation == "index": changed[0]["source"] = "narrative:99:development"
    elif mutation == "report20": changed[0]["source"] = "report:20:development"
    elif mutation == "layer": changed[0]["layer"] = 1
    elif mutation == "position": changed[0]["position"] = 1
    elif mutation == "headneg": changed[0]["head"] = -1
    elif mutation == "headhigh": changed[0]["head"] = 32
    elif mutation == "nan": changed[0]["metrics"]["cosine_similarity"] = math.nan
    elif mutation == "inf": changed[0]["metrics"]["relative_l2_error"] = math.inf
    with pytest.raises(c6.C6Error): c6.validate_observation_inventory(changed, action="A0", role="development")


def test_holdout_observations_are_bound_to_manifest_identity_set():
    selected = ["narrative:18:holdout", "report:22:holdout", "qa:19:holdout"]
    valid = observation_rows(selected)
    assert len(c6.validate_observation_inventory(valid, action="A0", role="holdout", expected_holdout_sources=selected)) == 1440
    changed = copy.deepcopy(valid); changed[0]["source"] = "narrative:19:holdout"
    with pytest.raises(c6.C6Error): c6.validate_observation_inventory(changed, action="A0", role="holdout", expected_holdout_sources=selected)
    with pytest.raises(c6.C6Error): c6.validate_observation_inventory(valid, action="A0", role="holdout")


def test_execution_preflight_requires_c6_tag(monkeypatch):
    monkeypatch.setattr(c6, "verify_runtime_closure", lambda: ())
    monkeypatch.setattr(c6, "validate_protocol", lambda: {})
    def fake_sha(path):
        return c6.C5_PARENT["recovered_result_sha256"] if "recovered_result" in str(path) else (c6.C5_PARENT["recovered_postmortem_sha256"] if "recovered_postmortem" in str(path) else "0" * 64)
    monkeypatch.setattr(c6, "sha256_path", fake_sha)
    monkeypatch.setattr(c6, "_git", lambda ref: c6.C5_PARENT["closure_commit"] if ref == c6.C5_PARENT["recovered_result_tag"] else "same")
    assert c6.preflight(execution=False)["local_only"]
    assert not c6.preflight(execution=True)["local_only"]
    monkeypatch.setattr(c6, "_git", lambda ref: "parent" if ref == c6.C5_PARENT["recovered_result_tag"] else ("head" if ref == "HEAD" else "other"))
    with pytest.raises(c6.C6Error): c6.preflight(execution=True)


def test_protocol_and_runtime_closure():
    assert c6.validate_protocol()["geometry"]["development_observations_per_method"] == 7200
    assert len(c6.verify_runtime_closure()) == 27
