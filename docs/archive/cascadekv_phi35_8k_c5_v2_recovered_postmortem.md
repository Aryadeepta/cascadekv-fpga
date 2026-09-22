# C5-v2 stdout-recovered result closure

The frozen scientific execution is `cascadekv-phi35-8k-c5-freeze-v2` at `36783eb8c75e9d4006d8ef4bae7d527942e95331`. Its protocol and runtime-manifest SHA-256 values are `2ab29d9ef370ba2f9308763e916b0c15795fbe379e317bf95bba20ae27b412d1` and `c6644332a676a839450814bcd2c8c22d5bfee4bb445c5e34099527b07511a818`.

The preserved stdout evidence is `artifacts/c5-v2/evidence/c5_v2_kaggle_stdout.md` (SHA-256 `907bd5cafd940035f7114ab001c65aa4eb08ad198ca7728e1ada86e096bab6bf`). All ten small artifacts were recovered byte-exact from its canonical framing, including a checked `SHA256SUMS`.

## Container provenance

The original Kaggle result tarball is unavailable locally. Its SHA-256, `9c87f801e1b037572e2c631b00f959bc01ae578ed0fe0d98028c585a387de2b0`, is transcript-attested metadata only and is not locally byte-verified.

`artifacts/c5-v2/recovered/cascadekv_phi35_8k_c5_stdout_recovery.tar.gz` is a **stdout-reconstructed convenience archive**, SHA-256 `b7ba80ae227c9b557d9ef1e39cc493fb6294c08686cdc5e371ec27c532e4347b`. Its members equal the recovered bytes, but it is a new container and is not the original Kaggle result tarball.

## Result and firewall

Development is `C5-DEVELOPMENT-GO`, with passing/selected target `T5`: cosine `0.9850333360217822`, relative L2 `0.11976855315762129`, and mean modeled KV `485339.77222222224`. Margins against frozen gates are +`0.0000333360217822376` cosine, +`0.00023144684237870472` relative-L2 headroom, +`0.10820278247813579` versus uniform10 relative L2, and `36.22777777776355` bytes below flat5.

Prospective T5 is `C5-PROSPECTIVE-NO-PASS`: cosine `0.9843795528635383`, relative L2 `0.1221502432927124`, mean modeled KV `484990.2166666667`; the cosine and relative-L2 margins are `-0.0006204471364616992` and `-0.002150243292712406`. The optimizer was not rerun, the schedule stayed `1dd4a48bb3596c2605cbbde7589c593500711b6cc85f8944ec97025595c0c646`, and prospective testing opened no development tensors. Later descriptive targets do not rescue the frozen selected-target result.

## Holdout and diagnosis

Selected holdouts are narrative 17, QA 18, and report 21. Report 20 was mechanically rejected: bounded frozen-tokenizer length 7218 fails `>=8192`. The untouched frontier is narrative 18, QA 19, report 22.

T5 actions are A0–A9 = 3, 6, 8, 2, 81, 4, 36, 1, 3, 16; qwen10 actions A7–A9 occupy 20/160 cells. Layer 0 is the dominant observed error bottleneck (relative L2 `0.32451231299207656`). That is not a causal comparison against C4, because C5's development population changed from 9 to 12 already-consumed sources.
