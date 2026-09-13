# CascadeKV-FPGA

## Frozen routing-scale contract

Each 16-coordinate Q8×K4 group uses a nonnegative scale. Query scale codes
are group-specific U0.16 (`real = code / 2^16`); a query has eight Q8 groups
and therefore eight corresponding `q_scale` codes. Each code is shared by all
candidate keys for that query group. K-group-cache scale metadata is unsigned
U7.9 (`real = code / 2^9`). `rtl/scale_product.sv` multiplies those codes and
rounds `(q_code * k_code + 2^12) >> 13` into saturated positive signed Q4.12
for `progressive_dot`.

## Online radix-16 hierarchy schedule

An append writes its ordinary authoritative leaf K every token.  A completed
16-token P1/P2 summary is finalized every 16 tokens; a parent is finalized
from its 16 child summaries every 256 tokens, then at 4096 and 65536 tokens.
Thus summary finalizations cost `1/16 + 1/256 + ... = 1/15` writes per
appended token amortized.  Parent radii can be maintained without rescanning
old leaf K using `max_child(||child_prototype-parent_prototype|| + child_radius)`.

## Experiment 1: Progressive Partial-Dot Attention

This experiment tests whether 16-, 32-, and 64-dimensional nested partial dot
products predict rankings of full 128-dimensional Qwen3 attention scores. A
shared randomized signed Hadamard rotation spreads each vector's information
across coordinates while preserving full dot products. The nested
16/32/64/128 prefixes model a progressively refined hardware score.

Run it on CPU with:

```bash
uv run python3 experiments/qwen_partial_dot.py --layer 12 --seed 0 --max-length 256 --num-query-positions 8
```

Inspect Pearson and Spearman score correlation, top-8/top-32 recall, and the
dense full-attention probability mass captured by approximate top-8/top-32
selection. Results are saved under `results/`.
