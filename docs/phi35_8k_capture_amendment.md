# Phi-3.5 8K minimal capture amendment supplement

This is a prospective, Kaggle-only execution package.  `preflight` is local
provenance validation only.  `capture` requires explicit inputs and is the
only mode that may load QA16, tokenizer assets, or the pinned Phi model.

Local pre-tag validation:

```bash
uv run python3 -m cascadekv.phi35_8k_capture_amendment preflight
```

The real Kaggle invocation, after the supplement freeze tag exists at `HEAD`, is:

```bash
uv run python3 -m cascadekv.phi35_8k_capture_amendment capture \
  --source-manifest /kaggle/working/cascadekv_phi35_8k_source_amendment/cascadekv_phi35_8k_dev_sources_v2.json \
  --base-capture-root /kaggle/working/cascadekv_phi35_8k_capture \
  --qualification /kaggle/working/cascadekv_phi35_8k_capture/qualification.json \
  --output /kaggle/working/cascadekv_phi35_8k_capture_amendment
```

Historical tensor files are read only as raw bytes for bytewise SHA-256
integrity verification. They are never deserialized/interpreted as tensors,
linked, copied, or rewritten by the supplement. The supplement directory
contains only five QA16 tensors/provenances, five QA15 role-reference JSON
records, and its own progress/final manifests.
