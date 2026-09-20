# Phi-3.5 8K C4: wider fixed action frontier

C4 is additive. It does not amend, reinterpret, or overwrite Phase A/B/C1/C2,
source-amendment, capture-amendment, or C3 artifacts. C3 is explicitly bound as
a no-pass: all schedules were feasible and beat both comparison baselines, but
none met both absolute output gates.

The sole C4 hypothesis is that the existing hierarchy/action frontier stopped
too early. The architecture, profiles, reserves, lambdas, representation, and
quality gates are unchanged. C4 adds only hierarchy A5 (0.175) and A6 (0.20)
to A0--A4, preserving the exact order A0 through A6. Its local optimizer is a
seven-action, 160-cell integer-microbyte DP; it does not mutate the frozen
five-action Phase-B optimizer.

`preflight` is completely local and imports no model, tokenizer, dataset, or
tensor stack. It verifies the C3 binding, the C4 runtime closure, and the
final C4 tag/HEAD identity. The final tag is
`cascadekv-phi35-8k-c4-prep-v1`.

Before the final commit and tag exist, use the explicitly preparation-only
Python validation path (for local checks and unit tests):

```bash
uv run python -c 'from cascadekv.phi35_8k_c4 import preparation_preflight; preparation_preflight()'
```

It deliberately disables only the final C4 tag and tracked-file requirements.
After committing and creating the frozen tag, normal preflight is mandatory:

```bash
uv run python -m cascadekv.phi35_8k_c4 preflight
```

The intended prospective sequence is:

```text
preflight
capture-development  # 9 corrected development sources, 45 artifacts
develop               # 45 development layers, 4,320 observations/method
select-holdout        # Kaggle only: one new source per family
capture-holdout       # 3 selected holdouts, 15 artifacts
test                  # 15 holdout layers, 1,440 observations/method
```

`select-holdout` begins at narrative 16, report 19, and QA 17. It uses the
frozen source fields, revisions, tokenizer revision, and exact 8192-token
construction. It rejects exact input-id hashes already in the nine-source
development pool or already selected by another family, selects the first
eligible globally-new candidate in each ascending frontier, and stores no raw
text. QA14 is rejected historical material and is never a C4 source.

Selection is an irreversible prospective boundary: it requires the complete
`--development-result`, validates it before loading a tokenizer or dataset,
hashes that exact result, and records both `development_result_sha256` and its
`development_schedule_sha256` in the holdout manifest. It derives the nine
development input hashes only from the exact canonical corrected source
manifest; there is no user-supplied development-hash argument. Holdout capture
and test rehash the supplied development result and fail if either binding has
changed. Test also rehashes the actual holdout manifest and requires the
capture manifest and every source artifact to bind that precise manifest and
its frozen input hash.

Both capture modes directly recapture sources under C4 provenance, one model
forward per source for all five layers. They require the exact C2 qualification
and backend IDs, preserve FP16/SDPA/no-cache/no-quantization/T4x2/post-RoPE
semantics, serialize CPU-contiguous FP16 Q/K/V safetensors, and refuse every
overwrite. Progress is concise, flushed STDERR; stdout remains canonical JSON.

`develop` can only resolve a development capture manifest and records no
holdout access. `test` first validates the complete development result, can
only resolve the holdout capture inventory, does not call the optimizer, and
checks its schedule digest before and after. A target passes only if it meets
both absolute gates, improves relative-L2 over uniform10, and uses less modeled
K+V traffic than flat5. C4 reports the first passing target or `null`; it does
not invent a classification mapping.

## One Kaggle runtime, in order

Use one uninterrupted Kaggle notebook/runtime: ephemeral artifacts are not a
portable handoff. Do not run C4 against a historical duplicate QA14 capture.
After cloning the frozen C4 tag and installing the frozen `uv` environment,
run the following stages in one session (paths are illustrative). Kaggle always
uses normal strict preflight from that frozen tag; it never uses the
preparation-only path:

```bash
uv run python -m cascadekv.phi35_8k_c4 preflight
# Reproduce/obtain the exact frozen C2 qualification and corrected v2 source manifest.
uv run python -m cascadekv.phi35_8k_c4 capture-development --development-source-manifest /kaggle/working/dev_sources_v2.json --qualification /kaggle/working/c2_qualification.json --output /kaggle/working/c4-dev-capture
uv run python -m cascadekv.phi35_8k_c4 develop --capture-root /kaggle/working/c4-dev-capture --capture-manifest /kaggle/working/c4-dev-capture/final_capture_manifest.json --output /kaggle/working/c4-development-result.json
uv run python -m cascadekv.phi35_8k_c4 select-holdout --development-source-manifest /kaggle/working/dev_sources_v2.json --development-result /kaggle/working/c4-development-result.json --output /kaggle/working/c4-holdout-selection.json
# Once source manifest, development result, and hashes are safely persisted,
# large development tensors may optionally be deleted to recover space.
uv run python -m cascadekv.phi35_8k_c4 capture-holdout --holdout-manifest /kaggle/working/c4-holdout-selection.json --development-result /kaggle/working/c4-development-result.json --qualification /kaggle/working/c2_qualification.json --output /kaggle/working/c4-holdout-capture
uv run python -m cascadekv.phi35_8k_c4 test --capture-root /kaggle/working/c4-holdout-capture --capture-manifest /kaggle/working/c4-holdout-capture/final_capture_manifest.json --holdout-manifest /kaggle/working/c4-holdout-selection.json --development-result /kaggle/working/c4-development-result.json --output /kaggle/working/c4-test-result.json
```

Each command emits canonical JSON on stdout and progress only on stderr. End
by printing preflight protocol/runtime hashes, development and holdout-manifest
hashes, capture hashes, final metrics, and the test result's two tensor-firewall
lists.
