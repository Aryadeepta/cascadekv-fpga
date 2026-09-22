# C6-v1 result closure

The original Kaggle result tarball is locally present and independently byte-verified: `artifacts/c6-v1/cascadekv-c6-result.tar.gz`, 74,912 bytes, SHA-256 `4e95dfde98e1e35a61dd08dffed9e7ef93ca466791485f9068392d1b2bded685`. Its ten members and its internal `SHA256SUMS` were verified.

The repository evidence file is separately preserved without normalization: `artifacts/c6-v1/evidence/c6_v1_kaggle_stdout.md`, SHA-256 `c8efdf27958af3de3c31f68456bedc7debb6ed9b067e273e155519097c391af4`. The earlier `c8b43eb0d8d0a809a207f49ce29ddc584948228d2db53c0ba30ffa0536ee44ad` identifies a separate uploaded/rendered Markdown representation; it is not attributed to this repository file.

Canonical stdout recovery reconstructed all ten original tar members and every recovered byte equals the corresponding tar member byte. Its schema label, `cascadekv-c5-stdout-recovery-v1`, is inherited historical transport framing only, not an assertion that this experiment is C5.

## Result

Frozen execution is `cascadekv-phi35-8k-c6-freeze-v1` (`f9dfdfffb5f2922a471f2bb955645ed52dda096c`). Development is `C6-DEVELOPMENT-GO`, passing and selecting T5. T5 development metrics are cosine `0.9863149033600671`, relative L2 `0.11671852580316984`, and modeled KV `485372.5788888889`; the cosine margin is `+0.0013149033600671048` and relative-L2 headroom `+0.003281474196830154`.

Prospective T5 is `C6-PROSPECTIVE-NO-PASS`: cosine `0.9847454029756287`, relative L2 `0.12684388312686273`, and modeled KV `485284.0777777778`. The optimizer was not rerun, the schedule remained `0f90a03895a02ce1efe2619cba25995ec7e6118e28a9fb5592cd3a15f03dbbdd`, and no development tensors were opened.

The holdouts are narrative 18, QA 19, and report 22; no candidates were rejected. The untouched frontier is narrative 19, QA 20, report 23, and raw text was not persisted. The 160-cell schedule counts A0--A11 are 5, 11, 6, 9, 80, 0, 11, 1, 6, 9, 17, 5. A10/A11 occur only in layer 0 (22/32 cells). Selected T5 layer relative-L2 values are L0 `0.29791271631291294`, L8 `0.069572871993719`, L16 `0.06630654037917003`, L24 `0.06250550337153506`, and L31 `0.08729499695851221`.
