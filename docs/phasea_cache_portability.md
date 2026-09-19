# Phase A cache portability boundary

Real-model capture is deliberately outside this repository's local evaluation
path.  A memory-capable host resolves the predeclared source, loads the target
model, captures one layer using a family adapter, and validates a
`CacheProvenance` record before exporting the tensors.  Another host may only
run routing evaluation after `CacheProvenance.validate()` succeeds.

The record binds the model/revision/config hash/tokenizer revision, geometry and
context, source identity/proof, layer, Q/K/V shapes and dtype, attention scale,
and adapter identity.  This preserves the same scientific semantics across
hosts without requiring local target weights or a local source corpus.
