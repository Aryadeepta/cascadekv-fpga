# Phi-3.5 8K C6 executor

C6 tests whether .25 hierarchy candidate capacity helps only at the earliest
sampled layer. It adds A10 (`qwen5-v1`, .25) and A11 (`qwen10-v1`, .25) to
the unchanged C5 A0–A9 menu. Every action is diagnostically evaluated over
the full 5 × 3 × 32 geometry. The optimizer may select A10/A11 only at layer
0; layers 8, 16, 24, and 31 remain strictly A0–A9.

Development is exactly 15 already-consumed identities: narrative 13–17,
report 16–19/21, and qa 13/15–18. Report 20 is structurally excluded. Thus
each method has exactly 7,200 development observations; the prospective
holdout remains three sources and 1,440 observations per method. The frozen
frontier is narrative 18, qa 19, report 22. Targets T0–T5 and all gates are
unchanged.

Development computes flat5/uniform10 baselines, all target schedules, and
the first passing target in T0–T5 order. A GO writes and SHA-binds that
result before prospective selection. The selected target alone determines
the prospective PASS/NO-PASS verdict; descriptive results for other targets
cannot change it. No optimizer or development tensor can be reached during
the prospective test.

Capture uses the unchanged isolated, detached C2 qualification boundary.
Source, artifact, provenance, capture, development, and holdout records bind
their hashes and never store raw text. The runtime closure additionally binds
the recovered C5 result and postmortem bytes; it expressly records that the
stdout reconstruction is not the original Kaggle result container. Terminal
small evidence is bundled create-only and emitted through canonical stdout
recovery before tensor cleanup.
