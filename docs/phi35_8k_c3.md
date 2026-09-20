# Phi-3.5 8K C3 calibration and one-shot validation

`preflight` is repository-only and does not import a tensor, data, tokenizer,
or model stack. It is the only permitted local command before the C3 tag.

```bash
uv run python3 -m cascadekv.phi35_8k_c3 preflight
```

After `cascadekv-phi35-8k-c3-prep-v1` exists at `HEAD`, the Kaggle-only
calibration command takes the two manifest files and the immutable base and
supplement roots explicitly. It opens only the six calibration identities and
refuses an existing output.

```bash
uv run python3 -m cascadekv.phi35_8k_c3 calibrate --source-manifest SOURCE --amended-capture-manifest CAPTURE --base-capture-root BASE --supplement-root SUPPLEMENT --output CALIBRATION.json
```

Validation requires the resulting calibration file. It hashes and validates it,
uses its action tables verbatim, opens only Narrative 15, Report 18, and QA 16,
and evaluates targets once in `T0` through `T5` order. Modeled K+V traffic is
reported solely as modeled traffic/fraction, never as speed, energy, or cost.

Calibration records every target even where the frozen integer-microbyte DP has
no feasible table: such a target has `feasible: false`, a null table, and null
microbytes used. Validation records that target as non-passing and continues to
later targets. Before and after validation it emits a canonical schedule digest;
the matching digests are the schedule-immutability proof. Validation does not
rerun the optimizer.

The Phase-B protocol names optional percentile diagnostics (`cosine_p5`,
`relative_l2_p95`, and `relative_l2_worst`), but no frozen repository
implementation establishes their percentile convention. C3 consequently marks
them explicitly unimplemented and retains the frozen mean-gate facts as the
primary result. Likewise, its frozen classification vocabulary is bound but C3
does not invent a mapping absent from frozen prior code.

```bash
uv run python3 -m cascadekv.phi35_8k_c3 validate --source-manifest SOURCE --amended-capture-manifest CAPTURE --base-capture-root BASE --supplement-root SUPPLEMENT --calibration-result CALIBRATION.json --output VALIDATION.json
```
