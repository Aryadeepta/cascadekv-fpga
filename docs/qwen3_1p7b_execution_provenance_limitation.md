# Qwen3-1.7B confirmatory execution provenance limitation

The confirmatory result and its shard-derived numerical outputs remain intact:
`results/cascadekv_v3_qwen3_1p7b_t1_4k_test_v2.json` has SHA256
`076f8fac9f4d6b16c5acfbd4235cc26215f44af3068a0f20e16d4adfa61a2e12`,
and its final audit has SHA256
`c513e44299f07ae32ac17be92600791d3a89d1823d4e6bcff9fbbd65be51ad64`.
The stored numerical result was independently recomputed from validated
shards.  The protocol and executor commits were frozen at
`999eb9a2c8767188c2b3d922f415194bd2a5bf54` and
`fe578c8ed56a2b635e389de841b08a5cd4c35169`, respectively.

The historical forensic audit found that some material runtime local
dependencies were untracked and unhashed.  Their exact execution-time bytes
are now unrecoverable.  Consequently, exact byte-for-byte reproduction of
the execution code is incomplete.  This is an execution-code-provenance
limitation, not evidence that the stored numerical result was altered; no
such evidence was found.

For later paper disclosure, this result should be described with that
limitation.  Future CascadeKV scientific runs require tracked, hash-bound
runtime closure before execution.
