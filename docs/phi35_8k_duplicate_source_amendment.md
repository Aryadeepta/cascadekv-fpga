# Phi-3.5 8K development-source duplicate-input amendment

The frozen C1 selection used unique dataset indices.  A post-capture mechanical
identity check established that LongBench NarrativeQA indices 13 and 14 produce
the same exact 8192-token model input (`input_ids_sha256`
`6efe85ed32e8f8e28ba946fa301f65bb7559f7ed596da3f62a519833ce3a80a4`), while
index 15 has a different input hash.  Consequently, the five sampled Q/K/V
artifacts for indices 13 and 14 were byte-identical.

This is a source-identity selection defect, not model or capture corruption.
No CascadeKV retrieval/output quality metric was observed and no schedule was
optimized.  C1 (`cascadekv-phi35-8k-sources-freeze`, commit
`667900a5307fe231565033a774d7788942fb869d`, manifest SHA
`af8e805bad5c85a1f9d6ae5571b378e4df46fdb1de778a5ec7b5b68b13dc4a8e`) and all
historical capture artifacts remain immutable.

The prospective v2 manifest schema is `cascadekv-phi35-8k-dev-sources-v2`.
It binds that v1 manifest SHA and changes only uniqueness to the SHA-256 of
the exact target-tokenizer input: `add_special_tokens=true`,
`truncation=true`, and `max_length=8192`, canonicalized as CPU-contiguous
`int64` bytes.  QA 13 is calibration_1; QA 14 is permanently unavailable as a
`duplicate_input` of 13; QA 15 is calibration_2; selection starts at 16 and
stops as soon as the first new eligible input becomes validation.  The winning
validation index is never hardcoded.

Narrative 13/14/15 and Report 16/17/18 are re-proved only.  Any duplicate in
either set fails closed; no replacement is selected.  The final nine selected
input hashes must also be globally unique or selection fails closed.

`preflight` is repository-only.  `select` is the only tokenizer/dataset mode,
does not load model weights, and writes only beneath
`/kaggle/working/cascadekv_phi35_8k_source_amendment/`; it never writes into
the immutable `/kaggle/working/cascadekv_phi35_8k_capture/` directory.  The
CLI requires an explicit `preflight` or `select` subcommand; it has no default
selection operation.

The real Kaggle CLI requires the future amendment-freeze tag
`cascadekv-phi35-8k-source-dedup-amendment-prep-v1` to exist and resolve
exactly to `HEAD`.  Before that tag is created, maintainers may run only the
data-free library preflight with `require_amendment_tag=False`; the CLI is
intentionally not relaxed.  A future v2 source manifest records this tag and
commit plus the exact amendment protocol and runtime-manifest SHA-256 values.

QA15's historical tensor bytes were captured while its old role was
`validation`; the amended source role is `calibration_2`.  Those bytes are not
invalidated.  A later capture-amendment stage must either provide a new
role-correct provenance wrapper/reference for the unchanged artifact SHA and
input hash, or recapture QA15 under amended provenance.  It must not silently
reinterpret the old per-artifact provenance as calibration_2.

```bash
uv run python3 -m cascadekv.phi35_8k_source_amendment preflight
uv run python3 -m cascadekv.phi35_8k_source_amendment select \
  --output /kaggle/working/cascadekv_phi35_8k_source_amendment/cascadekv_phi35_8k_dev_sources_v2.json
```

The manifest stores identities, mechanical proofs, canonical input hashes,
rejections, inventories, and frontiers; it stores no source text.
