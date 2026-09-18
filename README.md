## Current status

CascadeKV-v3 T0 has passed a frozen untouched 4K confirmatory test on
Qwen3-0.6B.

Frozen-v3 results:

- Mean relative-L2 output error: 0.1072
- Mean cosine similarity: 0.9900
- Mean total K+V traffic: 175,465 bytes/query
- Dense total K+V traffic: 1,572,864 bytes/query
- Total K+V traffic vs dense: 11.13%
- Reduction vs dense: 88.87%

Compared with frozen CascadeKV-v2 at essentially identical traffic:

- Relative-L2: 0.1275 -> 0.1072 (-15.9%)
- Absolute-L2: 0.7834 -> 0.5707 (-27.2%)
- Cosine: 0.9836 -> 0.9900

The v3 architecture and T0 layer×KV-head schedule were frozen before the
untouched test. The test contained 720 queries per method and 5,040 total
method/query measurements.

Result SHA256:
01c0bbda902a723599b709c8bc6be159c6191706a9c3531fd2da5fbb3fe5f09d

Status:
CASCADEKV-V3-4K-CONFIRMATORY-TEST-PASSED
