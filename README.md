# CascadeKV-FPGA

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
