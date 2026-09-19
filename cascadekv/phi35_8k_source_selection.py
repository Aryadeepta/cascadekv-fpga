"""Mechanical, text-free Phase-C1 source selection for Phi-3.5 8K.

This module is deliberately limited to frozen dataset identities, required-field
existence/type, character count, and a threshold-capped tokenizer predicate.
It never imports or loads a target model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping

from cascadekv.phi35_phaseb import PROTOCOL, RUNTIME_MANIFEST, verify_runtime_closure

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/cascadekv_phi35_8k_dev_sources.json"
PHASE_B_TAG = "cascadekv-phi35-8k-phaseb-protocol-freeze"
PHASE_B_COMMIT = "d94cff8e1e11048ff3d8edc9fb1c605dfe422d6e"
PROTOCOL_SHA256 = "0e3a2e7b51e01c64dca292c7dd667ed9899c5bacc08e470e2b19e12433363c8b"
RUNTIME_MANIFEST_SHA256 = "e4e10f7659180471eb791eeafa97e8f525b6c14a3ff311798349c07438ec7c92"
TARGET_MODEL = "microsoft/Phi-3.5-mini-instruct"
TARGET_REVISION = "2fe192450127e6a83f7441aef6e3ca586c338b77"
MINIMUM = 8192
IDENTITY_KIND = "dataset_index"
FAMILIES = ("narrative", "report", "qa")
NEXT_UNTOUCHED_INDICES = {"narrative": 16, "report": 19, "qa": 16}
PROOF_KEYS = (
    "field_exists", "is_python_string", "character_count",
    "bounded_frozen_tokenizer_length", "at_least_8192",
)


class SourceSelectionError(RuntimeError):
    """The fail-closed C1 source-selection contract was not met."""


def _complete_prior_unavailable_indices(
    first_permitted_index: int, historically_ineligible: list[int],
) -> list[int]:
    """Return the complete pre-C1 inventory fixed by the Phase-B frontier."""
    return sorted(set(range(first_permitted_index)).union(historically_ineligible))


def next_uninspected_index(last_inspected_index: int) -> int:
    """Mechanical identity frontier; this helper never resolves that identity."""
    if not isinstance(last_inspected_index, int) or isinstance(last_inspected_index, bool) or last_inspected_index < 0:
        raise SourceSelectionError("last inspected index is not a non-negative integer")
    return last_inspected_index + 1


class _PinnedParquetRows:
    """Index-only access to one pinned Parquet split and one permitted field."""
    def __init__(self, path: str, field: str) -> None:
        import pyarrow.parquet as pq
        self._parquet = pq.ParquetFile(path)
        self._field = field
        self._row_group_starts: list[int] = []
        offset = 0
        for group in range(self._parquet.num_row_groups):
            self._row_group_starts.append(offset)
            offset += self._parquet.metadata.row_group(group).num_rows
        self._length = offset
        self._field_exists = field in self._parquet.schema.names

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= index < self._length:
            raise IndexError(index)
        if not self._field_exists:
            return {}
        group = max(i for i, start in enumerate(self._row_group_starts) if start <= index)
        row_offset = index - self._row_group_starts[group]
        # Parquet decompression is row-group based, but this exposes exactly
        # one random-access row and never requests any non-permitted column.
        column = self._parquet.read_row_group(group, columns=[self._field]).column(self._field)
        return {self._field: column[row_offset].as_py()}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _protocol() -> dict[str, Any]:
    value = json.loads(PROTOCOL.read_text())
    if not isinstance(value, dict):
        raise SourceSelectionError("frozen protocol root is not an object")
    return value


def verify_phase_b_freeze() -> str:
    """Verify all C1 provenance gates before any tokenizer/source operation."""
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", f"{PHASE_B_TAG}^{{commit}}"], text=True
    ).strip()
    if commit != PHASE_B_COMMIT:
        raise SourceSelectionError("Phase-B freeze tag does not resolve to its frozen commit")
    if subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", commit, "HEAD"], check=False
    ).returncode:
        raise SourceSelectionError("current branch is not descended from the Phase-B freeze")
    if _sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise SourceSelectionError("Phase-B protocol SHA256 differs")
    if _sha256(RUNTIME_MANIFEST) != RUNTIME_MANIFEST_SHA256:
        raise SourceSelectionError("Phase-B runtime-manifest SHA256 differs")
    # This checks every Phase-B bound file and its Phase-A closure before source access.
    verify_runtime_closure()
    return commit


def frozen_specs() -> dict[str, dict[str, Any]]:
    protocol = _protocol()
    target = protocol.get("target_model")
    selection = protocol.get("source_selection")
    if target != {"name": TARGET_MODEL, "revision": TARGET_REVISION, "target_family": "phi3"}:
        raise SourceSelectionError("target model/revision is not the frozen Phi-3.5 target")
    if not isinstance(selection, dict) or selection.get("tokenizer_revision") != TARGET_REVISION or selection.get("minimum_bounded_tokenizer_length") != MINIMUM:
        raise SourceSelectionError("frozen tokenizer boundary differs")
    raw_specs = protocol.get("source_specs")
    if not isinstance(raw_specs, dict) or set(raw_specs) != set(FAMILIES):
        raise SourceSelectionError("frozen source families differ")
    specs: dict[str, dict[str, Any]] = {}
    for family in FAMILIES:
        raw = raw_specs[family]
        required = {"dataset", "config", "split", "revision", "text_field", "first_permitted_index"}
        if not isinstance(raw, dict) or not required <= set(raw):
            raise SourceSelectionError(f"frozen {family} source spec is malformed")
        first_permitted_index = raw["first_permitted_index"]
        historically_ineligible = raw.get("historically_ineligible", [])
        if (not isinstance(first_permitted_index, int) or isinstance(first_permitted_index, bool)
                or first_permitted_index < 0
                or not isinstance(historically_ineligible, list)
                or any(not isinstance(index, int) or isinstance(index, bool) or index < 0
                       for index in historically_ineligible)):
            raise SourceSelectionError(f"frozen {family} unavailable inventory is malformed")
        specs[family] = {
            "dataset": raw["dataset"], "config": raw["config"], "split": raw["split"],
            "revision": raw["revision"], "field": raw["text_field"],
            "first_permitted_index": first_permitted_index,
            "historically_ineligible_indices": list(historically_ineligible),
            "prior_unavailable_indices": _complete_prior_unavailable_indices(
                first_permitted_index, historically_ineligible,
            ),
        }
    return specs


def load_frozen_tokenizer() -> Any:
    """Load tokenizer assets only, at the exact target revision and without auth."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TARGET_MODEL, revision=TARGET_REVISION, token=False)


def load_dataset_for_spec(spec: Mapping[str, Any]) -> Any:
    """Load the pinned frontier shard only; never materialize unrelated shards."""
    from huggingface_hub import hf_hub_download
    frontier_files = {
        # The frozen frontiers lie in shard zero.  Selection fails closed if
        # that pinned random-access shard cannot supply three eligible rows;
        # it does not prefetch a later identity/shard.
        "emozilla/pg19": "data/train-00000-of-00023-5263dc4323e881d2.parquet",
        "ccdv/govreport-summarization": "document/train-00000-of-00002.parquet",
        "zai-org/LongBench": "narrativeqa/test-00000-of-00001.parquet",
    }
    filename = frontier_files.get(spec["dataset"])
    if filename is None:
        raise SourceSelectionError("source dataset is not frozen for C1")
    path = hf_hub_download(
        repo_id=spec["dataset"], repo_type="dataset", revision=spec["revision"],
        filename=filename, token=False,
    )
    return _PinnedParquetRows(path, spec["field"])


def bounded_token_length(tokenizer: Any, value: str) -> tuple[bool, int]:
    """Return only the threshold-capped length, never a complete document length."""
    encoded = tokenizer(value, add_special_tokens=True, truncation=True, max_length=MINIMUM)
    bounded = len(encoded.input_ids)
    return bounded >= MINIMUM, bounded


def mechanical_proof(dataset: Any, tokenizer: Any, spec: Mapping[str, Any], index: int) -> dict[str, Any]:
    """Inspect precisely one random-access row through the permitted predicate."""
    row = dataset[index]
    exists = spec["field"] in row
    value = row.get(spec["field"])
    is_string = isinstance(value, str)
    proof: dict[str, Any] = {
        "field_exists": exists,
        "is_python_string": is_string,
        "character_count": len(value) if is_string else None,
        "bounded_frozen_tokenizer_length": None,
        "at_least_8192": False,
    }
    if is_string:
        proof["at_least_8192"], proof["bounded_frozen_tokenizer_length"] = bounded_token_length(tokenizer, value)
    return proof


def _eligible(proof: Mapping[str, Any]) -> bool:
    return proof["field_exists"] is True and proof["is_python_string"] is True and proof["at_least_8192"] is True


def select_family(
    family: str, spec: Mapping[str, Any], tokenizer: Any, *,
    dataset_loader: Callable[[Mapping[str, Any]], Any] = load_dataset_for_spec,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Ascending selection; do not access any identity after acceptance number three."""
    dataset = dataset_loader(spec)
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    index = spec["first_permitted_index"]
    while index < len(dataset):
        proof = mechanical_proof(dataset, tokenizer, spec, index)
        if _eligible(proof):
            role = ("calibration_1", "calibration_2", "validation")[len(selected)]
            selected.append({"family": family, "dataset_index": index, "role": role, "proof": proof})
            if len(selected) == 3:
                return selected, rejected, index
        else:
            rejected.append({"dataset_index": index, "proof": proof})
        index += 1
    raise SourceSelectionError(f"insufficient eligible {family} identities from the frozen frontier")


def _source_record(selection: Mapping[str, Any], spec: Mapping[str, Any], reproof: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "family": selection["family"], "dataset": spec["dataset"], "config": spec["config"],
        "split": spec["split"], "revision": spec["revision"], "field": spec["field"],
        "identity_kind": IDENTITY_KIND, "dataset_index": selection["dataset_index"],
        "role": selection["role"], "proof": selection["proof"], "reproof": dict(reproof),
    }


def reproof_selected(
    selections: list[dict[str, Any]], specs: Mapping[str, Mapping[str, Any]], tokenizer: Any, *,
    dataset_loader: Callable[[Mapping[str, Any]], Any] = load_dataset_for_spec,
) -> list[dict[str, Any]]:
    """Freshly resolve only the selected identities, again, at frozen boundaries."""
    output = []
    for selection in selections:
        dataset = dataset_loader(specs[selection["family"]])
        proof = mechanical_proof(dataset, tokenizer, specs[selection["family"]], selection["dataset_index"])
        if proof != selection["proof"]:
            raise SourceSelectionError("selected source reproof differs from its stored mechanical proof")
        output.append(proof)
    return output


def build_manifest(
    specs: Mapping[str, Mapping[str, Any]], selections: Mapping[str, list[dict[str, Any]]],
    reproofs: Mapping[str, list[dict[str, Any]]], rejections: Mapping[str, list[dict[str, Any]]],
    last_inspected: Mapping[str, int], phase_b_commit: str,
) -> dict[str, Any]:
    sources = []
    for family in FAMILIES:
        sources.extend(_source_record(choice, specs[family], reproof)
                       for choice, reproof in zip(selections[family], reproofs[family], strict=True))
    unavailable = {}
    for family in FAMILIES:
        prior = list(specs[family]["prior_unavailable_indices"])
        newly_rejected = [item["dataset_index"] for item in rejections[family]]
        selected_indices = [item["dataset_index"] for item in selections[family]]
        unavailable[family] = {
            "prior_unavailable_indices": prior,
            "newly_rejected_indices": newly_rejected,
            "selected_development_indices": selected_indices,
            "indices": sorted(set(prior + newly_rejected + selected_indices)),
        }
    return {
        "schema_version": "cascadekv-phi35-8k-dev-sources-v1",
        "purpose": "Phase-C1 mechanical development-source selection only; never confirmatory data",
        "phase_b_freeze": {"tag": PHASE_B_TAG, "commit": phase_b_commit, "protocol_sha256": PROTOCOL_SHA256, "runtime_manifest_sha256": RUNTIME_MANIFEST_SHA256},
        "target": {"model": TARGET_MODEL, "model_revision": TARGET_REVISION, "tokenizer_revision": TARGET_REVISION},
        "identity_kind": IDENTITY_KIND,
        "minimum_bounded_frozen_tokenizer_length": MINIMUM,
        "source_specs": {family: {key: specs[family][key] for key in ("dataset", "config", "split", "revision", "field", "first_permitted_index")} for family in FAMILIES},
        "selected_sources": sources,
        "rejected_sources": {family: rejections[family] for family in FAMILIES},
        "last_inspected_index": dict(last_inspected),
        "unavailable_inventory": unavailable,
        "reproof_status": {"all_nine_selected_sources_equal": True, "selected_source_count": len(sources)},
    }


def _proof_is_valid(proof: Any, expected: bool) -> bool:
    if not isinstance(proof, dict) or set(proof) != set(PROOF_KEYS):
        return False
    if not isinstance(proof["field_exists"], bool) or not isinstance(proof["is_python_string"], bool) or not isinstance(proof["at_least_8192"], bool):
        return False
    if proof["character_count"] is not None and (not isinstance(proof["character_count"], int) or proof["character_count"] < 0):
        return False
    bounded = proof["bounded_frozen_tokenizer_length"]
    if proof["is_python_string"]:
        if not isinstance(bounded, int) or not 0 <= bounded <= MINIMUM:
            return False
    elif bounded is not None or proof["character_count"] is not None:
        return False
    return proof["at_least_8192"] is expected and (bounded >= MINIMUM if expected else True)


def validate_manifest(payload: Mapping[str, Any]) -> None:
    """Strictly validate the C1 selection boundary and text-free manifest shape."""
    specs = frozen_specs()
    expected_top = {"schema_version", "purpose", "phase_b_freeze", "target", "identity_kind", "minimum_bounded_frozen_tokenizer_length", "source_specs", "selected_sources", "rejected_sources", "last_inspected_index", "unavailable_inventory", "reproof_status"}
    if not isinstance(payload, Mapping) or set(payload) != expected_top:
        raise SourceSelectionError("manifest schema differs or permits an unbound payload")
    if payload["schema_version"] != "cascadekv-phi35-8k-dev-sources-v1" or payload["identity_kind"] != IDENTITY_KIND or payload["minimum_bounded_frozen_tokenizer_length"] != MINIMUM:
        raise SourceSelectionError("manifest version/identity/token threshold differs")
    if payload["phase_b_freeze"] != {"tag": PHASE_B_TAG, "commit": PHASE_B_COMMIT, "protocol_sha256": PROTOCOL_SHA256, "runtime_manifest_sha256": RUNTIME_MANIFEST_SHA256}:
        raise SourceSelectionError("manifest Phase-B binding differs")
    if payload["target"] != {"model": TARGET_MODEL, "model_revision": TARGET_REVISION, "tokenizer_revision": TARGET_REVISION}:
        raise SourceSelectionError("manifest target/tokenizer binding differs")
    expected_specs = {family: {key: specs[family][key] for key in ("dataset", "config", "split", "revision", "field", "first_permitted_index")} for family in FAMILIES}
    if payload["source_specs"] != expected_specs:
        raise SourceSelectionError("manifest source revisions differ from Phase B")
    sources = payload["selected_sources"]
    if not isinstance(sources, list) or len(sources) != 9:
        raise SourceSelectionError("manifest must contain exactly nine selected sources")
    selected_by_family = {family: [] for family in FAMILIES}
    source_keys = {"family", "dataset", "config", "split", "revision", "field", "identity_kind", "dataset_index", "role", "proof", "reproof"}
    all_indices: dict[str, set[int]] = {family: set() for family in FAMILIES}
    for source in sources:
        if not isinstance(source, dict) or set(source) != source_keys or source.get("family") not in FAMILIES:
            raise SourceSelectionError("selected source schema permits a text payload or is malformed")
        family = source["family"]; spec = specs[family]
        for key in ("dataset", "config", "split", "revision", "field"):
            if source[key] != spec[key]:
                raise SourceSelectionError("selected source does not bind a frozen source spec")
        if source["identity_kind"] != IDENTITY_KIND or not isinstance(source["dataset_index"], int) or source["dataset_index"] < spec["first_permitted_index"]:
            raise SourceSelectionError("selected source identity violates the frozen frontier")
        if not _proof_is_valid(source["proof"], True) or source["reproof"] != source["proof"]:
            raise SourceSelectionError("selected source proof/reproof differs or is ineligible")
        if source["dataset_index"] in all_indices[family]:
            raise SourceSelectionError("selected identities are not unique")
        all_indices[family].add(source["dataset_index"]); selected_by_family[family].append(source)
    rejected = payload["rejected_sources"]
    if not isinstance(rejected, Mapping) or set(rejected) != set(FAMILIES):
        raise SourceSelectionError("rejected-source families differ")
    last = payload["last_inspected_index"]
    if not isinstance(last, Mapping) or set(last) != set(FAMILIES):
        raise SourceSelectionError("last-inspected identities differ")
    for family in FAMILIES:
        ordered = sorted(selected_by_family[family], key=lambda item: item["dataset_index"])
        if len(ordered) != 3 or [item["role"] for item in ordered] != ["calibration_1", "calibration_2", "validation"]:
            raise SourceSelectionError("family must have two calibration and one validation sources in eligibility order")
        if [item["dataset_index"] for item in selected_by_family[family]] != [item["dataset_index"] for item in ordered]:
            raise SourceSelectionError("selections are not ascending")
        validation_index = ordered[-1]["dataset_index"]
        if last[family] != validation_index:
            raise SourceSelectionError("last inspected index must equal the validation identity")
        if next_uninspected_index(last[family]) != NEXT_UNTOUCHED_INDICES[family]:
            raise SourceSelectionError("next untouched identity differs from the frozen C1 boundary")
        family_rejected = rejected[family]
        if not isinstance(family_rejected, list):
            raise SourceSelectionError("rejected sources must be lists")
        for item in family_rejected:
            if not isinstance(item, dict) or set(item) != {"dataset_index", "proof"} or not isinstance(item["dataset_index"], int):
                raise SourceSelectionError("rejection schema permits a text payload or is malformed")
            if item["dataset_index"] < specs[family]["first_permitted_index"] or item["dataset_index"] >= validation_index or item["dataset_index"] in all_indices[family] or not _proof_is_valid(item["proof"], False):
                raise SourceSelectionError("rejection proof/position differs")
            all_indices[family].add(item["dataset_index"])
        if any(index > last[family] for index in all_indices[family]):
            raise SourceSelectionError("a proof exists past the scientifically required stop boundary")
        inventory = payload["unavailable_inventory"].get(family) if isinstance(payload["unavailable_inventory"], Mapping) else None
        expected_prior = _complete_prior_unavailable_indices(
            specs[family]["first_permitted_index"],
            specs[family]["historically_ineligible_indices"],
        )
        if not isinstance(inventory, Mapping) or not isinstance(inventory.get("prior_unavailable_indices"), list):
            raise SourceSelectionError("prior unavailable inventory differs from the frozen frontier")
        actual_prior = inventory["prior_unavailable_indices"]
        frontier = specs[family]["first_permitted_index"]
        if any(index not in actual_prior for index in range(frontier)):
            raise SourceSelectionError("prior unavailable inventory omits a pre-frontier identity")
        historical = set(specs[family]["historically_ineligible_indices"])
        if any(index >= frontier and index not in historical for index in actual_prior):
            raise SourceSelectionError("prior unavailable inventory invents a post-frontier identity")
        if actual_prior != expected_prior:
            raise SourceSelectionError("prior unavailable inventory differs from the frozen frontier")
        expected_rejected = [item["dataset_index"] for item in family_rejected]
        expected_selected = [item["dataset_index"] for item in ordered]
        expected_indices = sorted(set(expected_prior + expected_rejected + expected_selected))
        if inventory != {"prior_unavailable_indices": expected_prior, "newly_rejected_indices": expected_rejected, "selected_development_indices": expected_selected, "indices": expected_indices}:
            raise SourceSelectionError("unavailable inventory does not account for exactly consumed identities")
    if payload["reproof_status"] != {"all_nine_selected_sources_equal": True, "selected_source_count": 9}:
        raise SourceSelectionError("nine-source reproof status differs")


def execute(
    *, output: Path = OUTPUT, tokenizer: Any | None = None,
    dataset_loader: Callable[[Mapping[str, Any]], Any] = load_dataset_for_spec,
) -> dict[str, Any]:
    """Run complete C1 selection after provenance closure, then write its manifest."""
    phase_b_commit = verify_phase_b_freeze()
    specs = frozen_specs()
    frozen_tokenizer = tokenizer or load_frozen_tokenizer()
    selections: dict[str, list[dict[str, Any]]] = {}
    rejections: dict[str, list[dict[str, Any]]] = {}
    last: dict[str, int] = {}
    for family in FAMILIES:
        selections[family], rejections[family], last[family] = select_family(family, specs[family], frozen_tokenizer, dataset_loader=dataset_loader)
    reproofs = {family: reproof_selected(selections[family], specs, frozen_tokenizer, dataset_loader=dataset_loader) for family in FAMILIES}
    manifest = build_manifest(specs, selections, reproofs, rejections, last, phase_b_commit)
    validate_manifest(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Mechanical Phi-3.5 8K Phase-C1 source selection")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    try:
        execute(output=args.output)
        print(f"wrote text-free source manifest: {args.output}")
    except (SourceSelectionError, OSError, ValueError, IndexError) as exc:
        print(f"PHI35-8K-SOURCE-SELECTION-INVALID: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
