#!/usr/bin/env python3
"""Pre-data-only V2 confirmatory protocol harness; execution is disabled."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from experiments import cascadekv_v3_qwen3_1p7b_t1_source as source
CONFIG=ROOT/'configs/cascadekv_v3_qwen3_1p7b_t1.json'; MANIFEST=ROOT/'results/cascadekv_v3_qwen3_1p7b_t1_4k_test_v2_manifest.json'; INCIDENT=ROOT/'results/cascadekv_v3_qwen3_1p7b_t1_4k_test_incident.json'; RESULT=ROOT/'results/cascadekv_v3_qwen3_1p7b_t1_4k_test_v2.json'
CONFIG_SHA='5995a0f7fd6c3ebe762f27959d10de0398980e1aa013b669379f6b794cf5e88e'
METHODS=('dense_exact','flat_q8k4_5','flat_q8k4_10','uniform_v1_5','uniform_v1_10','frozen_cascadekv_v2','original_frozen_cascadekv_v3_zero_shot','frozen_qwen3_1p7b_T1')
LABELS={'invalid':'T1-CONFIRMATORY-TEST-INVALID','absolute_fail':'T1-CONFIRMATORY-OUTPUT-GATE-FAILED','comparative_fail':'T1-CONFIRMATORY-COMPARATIVE-GATE-FAILED','pass':'T1-CONFIRMATORY-TEST-PASSED'}
PROOF_KEYS=('dataset','config','split','revision','family','identity_kind','dataset_index','field','field_exists','is_string','character_count','tokenizer_model','tokenizer_revision','minimum','at_least_4096','bounded_frozen_tokenizer_length')
# A future capture phase must call this alias, never a duplicate loader or
# tokenizer predicate.  This pre-data harness intentionally never invokes it.
PRODUCTION_SOURCE_RESOLVER=source.resolve_frozen_source
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_manifest():
 m=json.loads(MANIFEST.read_text())
 if sha(CONFIG)!=CONFIG_SHA or m['target_t1_config']['sha256']!=CONFIG_SHA or tuple(m['methods'])!=METHODS: raise RuntimeError('T1-CONFIRMATORY-TEST-INVALID: frozen bindings')
 if sha(INCIDENT)!=m['predecessor_invalid_protocol']['incident_sha256']: raise RuntimeError('T1-CONFIRMATORY-TEST-INVALID: incident binding')
 for x in m['sources'].values():
  if x['identity_kind']!='dataset_index' or x['dataset_index'] in m['unavailable_inventory'][x['family']]['indices']: raise RuntimeError('T1-CONFIRMATORY-TEST-INVALID: source inventory')
  if (set(x['selection_proof']) != set(PROOF_KEYS) or
      set(x['production_resolution_reproof']) != set(PROOF_KEYS) or
      x['selection_proof'] != x['production_resolution_reproof'] or
      not x['selection_proof']['at_least_4096'] or
      x['selection_proof']['bounded_frozen_tokenizer_length'] != 4096):
   raise RuntimeError('T1-CONFIRMATORY-TEST-INVALID: source proof binding')
 return m
def preflight(): return load_manifest()
def status():
 load_manifest(); return {'manifest_sha256':sha(MANIFEST),'valid_caches':0,'total_caches':15,'valid_shards':0,'total_shards':15,'result_exists':RESULT.exists(),'execution_enabled':False}
def main():
 p=argparse.ArgumentParser(); g=p.add_mutually_exclusive_group(required=True); g.add_argument('--preflight',action='store_true'); g.add_argument('--status',action='store_true'); a=p.parse_args(); print(json.dumps(preflight() if a.preflight else status(),indent=2))
if __name__=='__main__': main()
