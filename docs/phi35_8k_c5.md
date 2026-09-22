# Phi-3.5 8K C5 preparation

C5 is a single, narrow finite-action expansion over recovered C4. A0--A6 are
identical to C4. A7/A8/A9 are hierarchy actions at 0.10/0.15/0.20 using only
the frozen `phi35-8k-depth-transfer-qwen10-v1` profile. No routing parameter
is tunable.

Development is exactly narrative 13--16, report 16--19, and QA 13,15,16,17.
Narrative 16, report 19, and QA 17 were C4 prospective sources and are now
already-consumed C5 development evidence. QA14 remains excluded. The untouched
prospective frontier is narrative 17, QA 18, report 20; preparation neither
selects nor accesses it.

The development executor evaluates and freezes all targets before the gate. A
target passes only if feasible, cosine >= .985, relative-L2 <= .120, lower
relative-L2 than development uniform10, and lower modeled KV than development
flat5. No passing target emits `C5-DEVELOPMENT-NO-GO` and terminates before any
prospective selection. On GO, the first passing target in T0..T5 order is
frozen with the result and schedule SHA before a separate selection command.

Prospective test opens only holdout tensors, never reruns the optimizer, and
requires identical schedule hashes before/after. Terminal paths write a small
tarball and print framed base64 bytes for every existing small artifact; raw
text and tensors are never printed.
