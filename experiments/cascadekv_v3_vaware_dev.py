#!/usr/bin/env python3
"""Development-only V-aware static LK schedule experiment.

The sole source-selection inputs are identity, text existence, character count,
and a bounded tokenizer length probe.  Calibration is the only input to table
construction.  This file never writes CascadeKV-v2.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, os, statistics, sys, tempfile
from itertools import product
from pathlib import Path
from typing import Any
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.fresh_sequence_benchmark import (MODEL, LAYERS, attention_reference,
 candidate_budget, file_sha256, frozen_routing_config, load_frozen, output_metrics,
 quantized_flat, reserve_ids, streaming_dataset, token_length_at_least, traffic)
from experiments.qwen_partial_dot import capture_target_layer_post_rope_qkv_low_memory
from experiments.confidence_routing_v2 import initialize_route, advance_until_budget
from cascadekv.adaptive_lifting import StreamingLiftingForest

MODEL_SHA='c1899de289a04d12100db370d81485cdf75e47ca'
V1_SHA='2964e4295719e696587f9853177c38399b64edc753e0630fc0d44208c719559b'
V2_SHA='ef36c5bd1b4211083ac5680e554877281f0a09e37f3dea9e4eb877afba18259c'
V2_TEST_SHA='31690daff3d60f4be0bed73139818c0adb3b71bed4e04ede77b20accbe3b140c'
V3_RESULT_SHA='1a5b0f78816b3748a832bc0cf9b6dc791795cfc3ee8c6050bc7130fad186268e'
POSITIONS=(2047,3071,4095); QH=16; KVH=8
MANIFEST=Path('results/cascadekv_v3_vaware_dev_manifest.json')
# Updated after the source-only audit.  Every operational mode validates this
# before parsing the manifest, so reconstruction can never silently drift.
V3_MANIFEST_SHA='0bf48150465b3c6cc0e31c35181bda3e337397b2a9d327d0fc1c5de25b16de8a'
V1=Path('configs/cascadekv_v1.json'); V2=Path('configs/cascadekv_v2.json'); V2_TEST=Path('results/cascadekv_v2_4k_test.json')
SPECS={'narrative':{'dataset':'emozilla/pg19','config':None,'split':'train','text_field':'text','revision':'c021754c8e01c5b1cc83a1f549c1f97fbbb756b8'},'report':{'dataset':'ccdv/govreport-summarization','config':None,'split':'train','text_field':'report','revision':'4e21184e01ae8017e2c036e180fe5e541fef60a0'},'qa':{'dataset':'zai-org/LongBench','config':'narrativeqa','split':'test','text_field':'context','revision':'75b6d5bffbcaa2cf4da85a9fa99939b13ee5b00b','loader':'direct_parquet','parquet_prefix':'narrativeqa/'}}
ACTIONS={'A0':(.05,'hierarchy'),'A1':(.075,'hierarchy'),'A2':(.10,'hierarchy'),'A3':(.15,'hierarchy'),'A4':(.05,'flat')}
METHODS=('dense_exact','flat_q8k4_5','uniform_v1_10','frozen_cascadekv_v2',*ACTIONS)
CAL=('narrative_calibration','report_calibration','qa_calibration'); VAL=('narrative_validation','report_validation','qa_validation')
T0_METRICS={'mean_relative_l2':0.11098487845085982,'mean_cosine':0.9898174457252026,
            'mean_kv_bytes':175642.35,'K+V/dense':0.11149707900153266,
            'K+V/uniform10':0.9145324838053707,'K+V/flat5':0.5451029632051112}

def immutable():
 if file_sha256(V1)!=V1_SHA: raise RuntimeError('STOP frozen CascadeKV-v1 hash differs')
 if file_sha256(V2)!=V2_SHA: raise RuntimeError('STOP frozen CascadeKV-v2 hash differs')
 if file_sha256(V2_TEST)!=V2_TEST_SHA: raise RuntimeError('STOP consumed test provenance hash differs')

def _identity_from_mapping(x):
 """Return an explicit source identity, never inventing one from a hash/cache."""
 if not isinstance(x,dict) or not {'dataset','config','split'} <= set(x): return None
 index=x.get('index',x.get('example_index'))
 if not isinstance(index,int): return None
 stable=x.get('stable_example_id',x.get('example_identifier'))
 return {'dataset':x['dataset'],'config':x['config'],'split':x['split'],
         'index':index,'stable_example_id':stable}

def prior_manifest_identity_inventory(paths=None):
 """Inventory explicit identities from the declared historical manifests only.

 ``paths`` is intentionally supplied by the frozen manifest during validation.
 A manifest created after that freeze is not historical evidence and must not
 change the frozen inventory protected by ``V3_MANIFEST_SHA``.
 """
 found=[]
 paths = (sorted(Path('results').glob('*manifest*.json')) if paths is None
          else [Path(path) for path in paths])
 for path in paths:
  if path.resolve()==MANIFEST.resolve(): continue
  try: document=json.loads(path.read_text())
  except (OSError,json.JSONDecodeError): continue
  def visit(node):
   identity=_identity_from_mapping(node)
   if identity is not None:
    found.append({'manifest':str(path),**identity})
   if isinstance(node,dict):
    for value in node.values(): visit(value)
   elif isinstance(node,list):
    for value in node: visit(value)
  visit(document)
 # Source and nested descriptive copies can name the same identity.  Keep an
 # auditable, deterministic one-entry-per-manifest/identity inventory.
 return sorted({json.dumps(x,sort_keys=True):x for x in found}.values(),
               key=lambda x:(x['manifest'],x['dataset'],str(x['config']),x['split'],x['index'],str(x['stable_example_id'])))

def identity_overlaps(source, prior):
 if any(source[k]!=prior[k] for k in ('dataset','config','split','index')): return False
 # A stable ID strengthens comparison when both manifests explicitly provide it.
 return source.get('stable_example_id') is None or prior.get('stable_example_id') is None or source['stable_example_id']==prior['stable_example_id']

def build_manifest():
 from transformers import AutoTokenizer
 immutable(); inventory=prior_manifest_identity_inventory(); tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA); sources={}
 for family,spec in SPECS.items():
  ds,_=streaming_dataset(spec,spec['revision']); chosen=[]
  for i,row in enumerate(ds):
   if i<3: continue
   text=row.get(spec['text_field'])
   if not isinstance(text,str): continue
   ok,n=token_length_at_least(tok,text,4096)
   if ok:
    candidate={'dataset':spec['dataset'],'config':spec['config'],'split':spec['split'],'exact_revision':spec['revision'],'index':i,'stable_example_id':row.get('id',i),'text_field':spec['text_field'],'character_count':len(text),'token_length_proof':{'minimum':4096,'observed_capped':n,'at_least':True}}
    if not any(identity_overlaps(candidate,old) for old in inventory): chosen.append(candidate)
   if len(chosen)==2: break
  if len(chosen)!=2: raise RuntimeError(f'not two eligible examples: {family}')
  for split,x in zip(('calibration','validation'),chosen): sources[f'{family}_{split}']={**x,'split_assignment':split}
 return {'schema_version':2,'purpose':'CascadeKV-v3 development-only V-aware LK calibration; identity/length-only selection','model':{'name':MODEL,'resolved_commit_sha':MODEL_SHA},'context_length':4096,'selection_constraints':'first two examples index >= 3 whose explicit identities are absent from prior manifests; only identity/index, text existence, character count, and token length inspected; indices 0/1/2 forbidden','split_policy':'first eligible per family is calibration; second is validation; validation must not influence schedule construction, traffic targets, or tie-breaks','prior_explicit_identity_inventory':inventory,'sources':sources,'frozen_inputs':{'cascadekv_v1':{'path':str(V1),'sha256':V1_SHA},'cascadekv_v2':{'path':str(V2),'sha256':V2_SHA},'consumed_test_provenance_only':{'path':str(V2_TEST),'sha256':V2_TEST_SHA,'row_metrics_used_for_fitting':False}}}

def preflight():
 if MANIFEST.exists(): return load_manifest()
 d=build_manifest(); MANIFEST.write_text(json.dumps(d,indent=2)+'\n'); return load_manifest()
def load_manifest():
 immutable()
 if not MANIFEST.is_file(): raise FileNotFoundError(f'STOP missing frozen manifest: {MANIFEST}')
 if file_sha256(MANIFEST)!=V3_MANIFEST_SHA: raise RuntimeError('STOP frozen v3 manifest hash differs')
 d=json.loads(MANIFEST.read_text())
 if d.get('schema_version')!=2 or d.get('model',{}).get('resolved_commit_sha')!=MODEL_SHA or d.get('context_length')!=4096: raise ValueError('manifest model/context mismatch')
 if set(d.get('sources',{}))!=set(CAL+VAL): raise ValueError('manifest split schema mismatch')
 snapshot=d.get('prior_explicit_identity_inventory')
 if not isinstance(snapshot,list): raise ValueError('prior explicit identity inventory missing')
 historical_paths=sorted({x.get('manifest') for x in snapshot if isinstance(x,dict) and isinstance(x.get('manifest'),str)})
 if any(not isinstance(x,dict) or x.get('manifest') not in historical_paths for x in snapshot): raise ValueError('prior explicit identity inventory malformed')
 if snapshot!=prior_manifest_identity_inventory(historical_paths): raise ValueError('prior explicit identity inventory mismatch')
 if d.get('frozen_inputs',{}).get('cascadekv_v1',{}).get('sha256')!=V1_SHA: raise ValueError('manifest v1 binding mismatch')
 for k,x in d['sources'].items():
  family=k.rsplit('_',1)[0]
  if x['index']<3 or x['split_assignment']!=k.rsplit('_',1)[1] or x['dataset']!=SPECS[family]['dataset']: raise ValueError('consumed index or split mismatch')
 return d
def cache_path(root,s,l): return root/f'{s}_L4096_layer{l}_qkv.pt'
def shard_path(root,s,l): return root/f'{s}_L4096_layer{l}.json'
def validate_cache(p,s,l):
 try:
  x=torch.load(p,map_location='cpu',weights_only=False); m=x['metadata']; mf=load_manifest()
  q,k,v=(x[n] for n in ('query','key','value'))
  if m.get('manifest_sha256')!=V3_MANIFEST_SHA or m.get('model_sha')!=MODEL_SHA or m.get('model_revision')!=MODEL_SHA or m.get('context_length')!=4096 or m.get('sequence')!=s or m.get('layer')!=l or m.get('source_identity')!=mf['sources'][s]: return False,'metadata identity binding'
  if tuple(q.shape)!=(1,16,4096,128) or tuple(k.shape)!=(1,8,4096,128) or tuple(v.shape)!=(1,8,4096,128) or q.dtype!=torch.float16 or k.dtype!=torch.float16 or v.dtype!=torch.float16:return False,'exact FP16 Q/K/V production shapes'
  return True,None
 except Exception as e:return False,str(e)
def validate_shard(p,s,l):
 try:
  d=json.loads(p.read_text()); expected={(z,h,m) for z in POSITIONS for h in range(QH) for m in METHODS}; rows=d['rows']; got=[(r.get('position'),r.get('q_head'),r.get('method')) for r in rows]
  if d.get('manifest_sha256')!=V3_MANIFEST_SHA or d.get('model_sha')!=MODEL_SHA or d.get('sequence')!=s or d.get('layer')!=l or d.get('context_length')!=4096:return False,'top-level metadata binding'
  if len(rows)!=len(expected) or set(got)!=expected or len(set(got))!=len(got):return False,'exact method/action rows or duplicates/missing/extras'
  if any(r.get('sequence')!=s or r.get('layer')!=l or r.get('context_length')!=4096 or r.get('position') not in POSITIONS or not isinstance(r.get('q_head'),int) or not 0<=r['q_head']<QH or r.get('kv_head')!=r['q_head']//2 or r.get('method') not in METHODS for r in rows):return False,'row metadata binding'
  return True,None
 except Exception as e:return False,str(e)
def status(cache,shards):
 load_manifest(); pairs=[]; c=v=0
 for s in CAL+VAL:
  for l in LAYERS:
   a,am=validate_cache(cache_path(cache,s,l),s,l) if cache_path(cache,s,l).exists() else (False,'absent'); b,bm=validate_shard(shard_path(shards,s,l),s,l) if shard_path(shards,s,l).exists() else (False,'absent'); c+=a;v+=b;pairs.append({'sequence':s,'layer':l,'cache':{'status':'valid' if a else am},'shard':{'status':'valid' if b else bm}})
 return {'manifest_sha256':file_sha256(MANIFEST),'pairs':pairs,'valid_caches':c,'total_caches':30,'valid_shards':v,'total_shards':30}
def resolve(s,tok,m):
 x=m['sources'][s]; spec=SPECS[s.rsplit('_',1)[0]]; ds,_=streaming_dataset(spec,spec['revision'])
 for i,row in enumerate(ds):
  if i==x['index']:
   if row.get('id',i)!=x['stable_example_id'] or not token_length_at_least(tok,row[spec['text_field']],4096)[0]:raise ValueError('frozen source fails resolve')
   return row[spec['text_field']]
 raise ValueError('frozen source absent')
def capture(s,l,root):
 if cache_path(root,s,l).exists() and validate_cache(cache_path(root,s,l),s,l)[0]:return
 from transformers import AutoModel,AutoTokenizer
 m=load_manifest();tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA);ids=tok(resolve(s,tok,m),return_tensors='pt',truncation=True,max_length=4096).input_ids
 model=AutoModel.from_pretrained(MODEL,revision=MODEL_SHA,dtype=torch.float16,low_cpu_mem_usage=True).eval();q,k,v=capture_target_layer_post_rope_qkv_low_memory(model,ids,l); root.mkdir(parents=True,exist_ok=True)
 out=cache_path(root,s,l)
 torch.save({'metadata':{'manifest_sha256':V3_MANIFEST_SHA,'model_sha':MODEL_SHA,'model_revision':MODEL_SHA,'sequence':s,'layer':l,'source_identity':m['sources'][s],'context_length':4096,'attention_scaling':float(model.layers[l].self_attn.scaling),'storage':'fp16 authoritative Q/K/V'},'query':q.half(),'key':k.half(),'value':v.half()},out)
 ok,msg=validate_cache(out,s,l)
 if not ok: raise RuntimeError(f'STOP captured cache failed validation: {msg}')
def route(f,q,b,lam,res):return advance_until_budget(initialize_route(f,q,4,lam,res),b)
def action_ids(action,f,q,k,n,lam,sch):
 frac,kind=ACTIONS[action];b=candidate_budget(n,frac)
 if kind=='flat':return set(quantized_flat(q,k,b)),traffic({'active_root_reads':0,'detail_reads':0,'expanded_internal_nodes':0,'variance_scalar_reads':0},n,b,variance=False,flat=True)
 r=route(f,q,b,lam,reserve_ids(n,sch['sink'],sch['local'],b));return set(r['ids']),traffic(r,n,b,variance=True)
def v2_action(layer,kv):
 cfg=json.loads(V2.read_text()); return cfg['exact_layer_head_action_table'][str((layer,kv))]
def evaluate(s,l,cache,shards):
 p=shard_path(shards,s,l)
 if p.exists() and validate_shard(p,s,l)[0]:return
 ok,msg=validate_cache(cache_path(cache,s,l),s,l)
 if not ok:raise ValueError(msg)
 d=torch.load(cache_path(cache,s,l),map_location='cpu',weights_only=False);qall,kall,vall=d['query'][0].float(),d['key'][0].float(),d['value'][0].float();scale=d['metadata']['attention_scaling'];cfg=load_frozen(V1); forests={};rows=[]
 for kv in range(8):
  f=StreamingLiftingForest(mode='binary_counter',atom_size=8,window=4,group_counts=(4,))
  for st in range(0,4096,8):
   f.append_atom(kall[kv,st:st+8])
   if st+7 in POSITIONS:forests[st+7,kv]=copy.deepcopy(f)
 for pos in POSITIONS:
  n=pos+1
  for h in range(16):
   kv=h//2;q,k,v=qall[h,pos],kall[kv,:n],vall[kv,:n];ref=attention_reference(q,k,v,scale); sch10,lam10=frozen_routing_config(cfg,l,.10); methods={'dense_exact':(set(range(n)),{'total_k_bytes':float(n*256),'selected_v_fp16_bytes':float(n*256)}),'flat_q8k4_5':action_ids('A4',forests[pos,kv],q,k,n,lam10,sch10),'uniform_v1_10':action_ids('A2',forests[pos,kv],q,k,n,lam10,sch10)}
   sch5,lam5=frozen_routing_config(cfg,l,.05)
   methods['frozen_cascadekv_v2']=action_ids(v2_action(l,kv),forests[pos,kv],q,k,n,lam5,sch5)
   for a in ACTIONS:methods[a]=action_ids(a,forests[pos,kv],q,k,n,lam5,sch5)
   for name,(ids,tr) in methods.items():
    tr={**tr};tr['total_kv_bytes']=tr['total_k_bytes']+tr['selected_v_fp16_bytes'];tr['total_k_vs_dense']=tr['total_k_bytes']/(n*256);tr['total_kv_vs_dense']=tr['total_kv_bytes']/(n*512)
    rows.append({'sequence':s,'layer':l,'context_length':4096,'position':pos,'q_head':h,'kv_head':kv,'method':name,'metrics':output_metrics(q,k,v,ids,ref,scale),'traffic':tr})
 shards.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps({'manifest_sha256':V3_MANIFEST_SHA,'model_sha':MODEL_SHA,'sequence':s,'layer':l,'context_length':4096,'rows':rows},indent=2)+'\n')
 ok,msg=validate_shard(p,s,l)
 if not ok: raise RuntimeError(f'STOP written shard failed validation: {msg}')

def aggregate(rows):
 out={}
 for l in LAYERS:
  for kv in range(8):
   for a in ACTIONS:
    r=[x for x in rows if x['layer']==l and x['kv_head']==kv and x['method']==a];out[f'{l}:{kv}:{a}']={'query_count':len(r),'relative_l2':statistics.mean(x['metrics']['relative_l2_error'] for x in r),'cosine':statistics.mean(x['metrics']['cosine_similarity'] for x in r),'mass':statistics.mean(x['metrics']['relative_exact_attention_mass'] for x in r),'total_k_bytes':statistics.mean(x['traffic']['total_k_bytes'] for x in r),'total_kv_bytes':statistics.mean(x['traffic']['total_kv_bytes'] for x in r)}
 return out
def optimize(groups,target,diagnostics=None):
 """Exact lexicographic multiple-choice knapsack over integer microbytes.

 The final objective is (relative-L2 sum, -cosine sum, traffic, actions),
 not a three-objective Pareto objective.  At a fixed prefix length, a state
 with no more traffic and lexicographically better (error, -cosine) remains
 better after any identical suffix, so a low-to-high traffic sweep is exact.
 """
 cap=round(target*len(groups)*1000)
 # A base-five prefix code is the action tuple without copying an O(groups)
 # tuple into every frontier state.  At one prefix depth its integer ordering
 # is exactly the ordering of (A0, ..., A4) tuples.
 action_index={a:i for i,a in enumerate(ACTIONS)}
 states=[(0,0.,0.,0)]
 max_frontier_size=1; expanded_state_count=0
 for g in groups:
  # Future traffic is nonnegative, so over-cap prefixes cannot become feasible.
  nxt=[]
  for b,e,c,code in states:
   for a in ACTIONS:
    x=g[a]; candidate=(b+round(x['total_kv_bytes']*1000),e+x['relative_l2'],c+x['cosine'],code*5+action_index[a])
    if candidate[0]<=cap:nxt.append(candidate)
  expanded_state_count+=len(nxt)
  # First select the frozen lexicographic winner for each exact traffic value.
  nxt.sort(key=lambda z:(z[0],z[1],-z[2],z[3]))
  traffic_winners=[]; last_traffic=None
  for x in nxt:
   if x[0]!=last_traffic:
    traffic_winners.append(x);last_traffic=x[0]
  # A later traffic value is useful only if it improves the primary quality
  # pair.  Equal quality is dominated by the already-seen lower traffic.
  states=[]; best_quality=None
  for x in traffic_winners:
   quality=(x[1],-x[2])
   if best_quality is None or quality<best_quality:
    states.append(x);best_quality=quality
  max_frontier_size=max(max_frontier_size,len(states))
  if not states:break
 if diagnostics is not None:
  diagnostics.update(max_frontier_size=max_frontier_size,expanded_state_count=expanded_state_count)
 if not states:return None
 b,e,c,code=min(states,key=lambda z:(z[1],-z[2],z[0],z[3]))
 actions=[None]*len(groups)
 for i in range(len(groups)-1,-1,-1):
  code,digit=divmod(code,5);actions[i]=tuple(ACTIONS)[digit]
 return b,e,c,tuple(actions)
def classify(candidates,validation,uniform):
 good=[x for x in candidates if x and x['relative_l2']<=.12 and x['cosine']>=.985 and x['kv']<=uniform and x['relative_l2']<x['v2_relative_l2']]
 if good:return 'V-AWARE-STATIC-PROMISING'
 if validation.get('oracle_headroom'):return 'V-AWARE-STATIC-NOT-GENERALIZING'
 return 'ACTION-MENU-LIMITED'
def table_summary(rows, table):
 """Apply an already frozen LK table by selecting precomputed action rows."""
 picked=[r for r in rows if r['method']==table[f"{r['layer']}:{r['kv_head']}"]]
 if not picked: return None
 def q(xs,p):
  xs=sorted(xs); return xs[min(len(xs)-1,math.ceil(p*len(xs))-1)]
 rel=[r['metrics']['relative_l2_error'] for r in picked]; cos=[r['metrics']['cosine_similarity'] for r in picked]
 return {'mean_mass':statistics.mean(r['metrics']['relative_exact_attention_mass'] for r in picked),'mean_top8':statistics.mean(r['metrics']['top8_recall'] for r in picked),'mean_cosine':statistics.mean(cos),'mean_relative_l2':statistics.mean(rel),'p5_cosine':q(cos,.05),'p95_relative_l2':q(rel,.95),'worst_relative_l2':max(rel),'mean_kv_bytes':statistics.mean(r['traffic']['total_kv_bytes'] for r in picked),'mean_k_bytes':statistics.mean(r['traffic']['total_k_bytes'] for r in picked),'K/dense':statistics.mean(r['traffic']['total_k_vs_dense'] for r in picked),'K+V/dense':statistics.mean(r['traffic']['total_kv_vs_dense'] for r in picked)}
def method_summary(rows, method):
 return table_summary(rows,{f'{l}:{kv}':method for l in LAYERS for kv in range(8)})
def by_lk(rows):
 return {f'{l}:{kv}':statistics.mean(r['metrics']['relative_l2_error'] for r in rows if r['layer']==l and r['kv_head']==kv) for l in LAYERS for kv in range(8)}
def construct_calibration_schedule(cal, validation_rows=None):
 """The only schedule-construction path; deliberately accepts calibration rows only."""
 del validation_rows  # Explicitly prove this input cannot affect fitting.
 ag=aggregate(cal); groups=[{a:ag[f'{l}:{kv}:{a}'] for a in ACTIONS} for l in LAYERS for kv in range(8)]
 mean=lambda method:statistics.mean(x['traffic']['total_kv_bytes'] for x in cal if x['method']==method)
 v2,u10=mean('frozen_cascadekv_v2'),mean('uniform_v1_10')
 targets={'T0':v2,'T1':v2+.5*(u10-v2),'T2':u10}; tables={}
 for name,t in targets.items():
  z=optimize(groups,t)
  tables[name]=None if z is None else {'nondeployable':False,'target_kv':t,'calibration_mean_relative_l2':z[1]/40,'table':{f'{l}:{kv}':a for (l,kv),a in zip(product(LAYERS,range(8)),z[3])}}
 return ag,targets,tables

def _validate_t0_table(table):
 expected={f'{layer}:{kv}' for layer in LAYERS for kv in range(8)}
 if not isinstance(table,dict) or set(table)!=expected or any(action not in ACTIONS for action in table.values()):
  raise RuntimeError('STOP T0 table is not exactly 40 layer/KV-head A0..A4 actions')

def reconstruct_t0_table(shards):
 """Read calibration shards only; validation rows can never reach the optimizer."""
 calibration=[]
 for sequence in CAL:
  for layer in LAYERS:
   path=shard_path(shards,sequence,layer)
   ok,message=validate_shard(path,sequence,layer) if path.exists() else (False,'missing')
   if not ok: raise RuntimeError(f'STOP calibration shard invalid: {sequence} layer {layer}: {message}')
   calibration.extend(json.loads(path.read_text())['rows'])
 _,_,tables=construct_calibration_schedule(calibration)
 table=tables.get('T0',{}).get('table')
 _validate_t0_table(table)
 return table

def _atomic_write(path, data):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
 fd,name=tempfile.mkstemp(prefix=f'.{path.name}.',dir=path.parent)
 try:
  with os.fdopen(fd,'wb') as handle: handle.write(data)
  os.replace(name,path)
 except Exception:
  try: os.unlink(name)
  except FileNotFoundError: pass
  raise

def _v3_config(source_table, manifest_sha):
 """Derive v3 from v2 without changing its router or representation settings."""
 cfg=json.loads(V2.read_text())
 cfg.update({'architecture_version':'cascadekv-v3','schedule_family':'LK_VAWARE',
             'base_operating_concept':'T0 / frozen CascadeKV-v2 total-KV traffic envelope',
             'target_name':'T0',
             'exact_layer_head_action_table':{str((layer,kv)):source_table[f'{layer}:{kv}'] for layer in LAYERS for kv in range(8)},
             'v_aware_calibration_objective':'mean relative-L2',
             'v3_manifest_sha256':manifest_sha,'v3_development_result_sha256':V3_RESULT_SHA,
             'v2_parent_config_sha256':V2_SHA,'v1_provenance_sha256':V1_SHA,
             'development_classification':'V-AWARE-STATIC-PROMISING',
             'predeclared_development_candidates':'T0/T1/T2 were predeclared development candidates.',
             'development_selection_rationale':'After development validation, T0 is chosen as the minimum-bandwidth predefined target satisfying the v3 development criteria. This is development selection and requires a new untouched holdout; it was not a predeclared selection rule before v3 results.'})
 # This v2-specific assertion must not be inherited as a false statement about v3.
 cfg.pop('predeclared_selection_rule',None)
 return cfg

def validate_freeze_v3(result_path=Path('results/cascadekv_v3_vaware_dev.json'), manifest_path=MANIFEST,
                       shards=Path('results/cascadekv_v3_vaware_dev_shards'),
                       cache=Path('results/cascadekv_v3_vaware_dev_cache')):
 """Fail closed before any production freeze output is written."""
 result_path,manifest_path=Path(result_path),Path(manifest_path)
 immutable()
 if file_sha256(result_path)!=V3_RESULT_SHA: raise RuntimeError('STOP completed v3 result hash differs')
 if file_sha256(manifest_path)!=V3_MANIFEST_SHA: raise RuntimeError('STOP frozen v3 manifest hash differs')
 if manifest_path.resolve()!=MANIFEST.resolve():
  # validate_shard/load_manifest intentionally bind to the immutable canonical manifest.
  raise RuntimeError('STOP freeze manifest must be the immutable canonical manifest')
 result=json.loads(result_path.read_text())
 if result.get('status')!='complete' or result.get('classification')!='V-AWARE-STATIC-PROMISING': raise RuntimeError('STOP v3 status/classification differs')
 if result.get('manifest_sha256')!=V3_MANIFEST_SHA: raise RuntimeError('STOP result manifest provenance differs')
 t0=result.get('validation_frozen_table_metrics',{}).get('T0')
 if not isinstance(t0,dict) or any(t0.get(key)!=value for key,value in T0_METRICS.items()): raise RuntimeError('STOP exact T0 development metrics differ')
 v2=result.get('validation_baselines',{}).get('frozen_cascadekv_v2',{})
 uniform=result.get('validation_baselines',{}).get('uniform_v1_10',{})
 if not (t0['mean_relative_l2']<=.12 and t0['mean_cosine']>=.985 and t0['mean_kv_bytes']<=uniform.get('mean_kv_bytes',-1) and t0['mean_relative_l2']<v2.get('mean_relative_l2',-1)):
  raise RuntimeError('STOP T0 development gate differs')
 artifact_status=status(cache,shards)
 if artifact_status['valid_caches']!=30 or artifact_status['valid_shards']!=30: raise RuntimeError('STOP v3 artifact status is not 30/30')
 source_table=result.get('calibration_frozen_tables',{}).get('T0',{}).get('table')
 _validate_t0_table(source_table)
 reconstructed=reconstruct_t0_table(shards)
 if reconstructed!=source_table: raise RuntimeError('STOP calibration-only T0 reconstruction differs')
 return result,source_table,artifact_status

def freeze_v3(result_path=Path('results/cascadekv_v3_vaware_dev.json'),
              config_path=Path('configs/cascadekv_v3.json'),
              frozen_result_path=Path('results/cascadekv_v3_vaware_dev_frozen.json'),
              manifest_path=MANIFEST, shards=Path('results/cascadekv_v3_vaware_dev_shards'),
              cache=Path('results/cascadekv_v3_vaware_dev_cache')):
 """Explicitly freeze the existing completed development result; never evaluate or merge."""
 result,table,artifact_status=validate_freeze_v3(result_path,manifest_path,shards,cache)
 # Both byte payloads are prepared only after every validation above has passed.
 config_bytes=(json.dumps(_v3_config(table,V3_MANIFEST_SHA),indent=2)+'\n').encode()
 result_bytes=Path(result_path).read_bytes()
 _atomic_write(config_path,config_bytes)
 _atomic_write(frozen_result_path,result_bytes)
 return {'config_sha256':file_sha256(config_path),'frozen_result_sha256':file_sha256(frozen_result_path),
         'artifact_status':artifact_status,'t0_table':table,'classification':result['classification']}
def merge(shards,out):
 load_manifest(); allrows=[]; bad=[]
 for s in CAL+VAL:
  for l in LAYERS:
   p=shard_path(shards,s,l);ok,msg=validate_shard(p,s,l) if p.exists() else (False,'missing')
   if ok:allrows+=json.loads(p.read_text())['rows']
   else:bad.append({'sequence':s,'layer':l,'reason':msg})
 if bad:return out.write_text(json.dumps({'status':'partial','invalid_shards':bad},indent=2)+'\n')
 cal=[r for r in allrows if r['sequence'] in CAL];val=[r for r in allrows if r['sequence'] in VAL];ag,targets,tables=construct_calibration_schedule(cal); vag=aggregate(val); groups=[{a:ag[f'{l}:{kv}:{a}'] for a in ACTIONS} for l in LAYERS for kv in range(8)]
 frozen_validation={n:table_summary(val,x['table']) if x else None for n,x in tables.items()}
 # Fitted only after frozen validation; these objects are marked nondeployable.
 vgroups=[{a:vag[f'{l}:{kv}:{a}'] for a in ACTIONS} for l in LAYERS for kv in range(8)]
 posthoc={}; per_query={}
 for n,t in targets.items():
  z=optimize(vgroups,t); table=None if z is None else {f'{l}:{kv}':a for (l,kv),a in zip(product(LAYERS,range(8)),z[3])}
  posthoc[n]={'nondeployable':True,'deployment_status':'NONDEPLOYABLE diagnostic only','fit_partition':'validation','table':table,'summary':None if table is None else table_summary(val,table)}
  qgroups=[]
  for key in sorted({(r['sequence'],r['layer'],r['position'],r['q_head']) for r in val}):
   rs={r['method']:r for r in val if (r['sequence'],r['layer'],r['position'],r['q_head'])==key and r['method'] in ACTIONS}
   qgroups.append({a:{'total_kv_bytes':rs[a]['traffic']['total_kv_bytes'],'relative_l2':rs[a]['metrics']['relative_l2_error'],'cosine':rs[a]['metrics']['cosine_similarity']} for a in ACTIONS})
  zq=optimize(qgroups,t); per_query[n]={'nondeployable':True,'deployment_status':'NONDEPLOYABLE diagnostic only','fit_partition':'validation','construction':'deterministic multiple-choice knapsack over query action rows','mean_relative_l2':None if zq is None else zq[1]/len(qgroups),'mean_cosine':None if zq is None else zq[2]/len(qgroups),'mean_kv_bytes':None if zq is None else zq[0]/1000/len(qgroups)}
 uval=method_summary(val,'uniform_v1_10'); v2val=method_summary(val,'frozen_cascadekv_v2'); baseline={m:method_summary(val,m) for m in ('frozen_cascadekv_v2','uniform_v1_10','flat_q8k4_5')}
 candidates=[]
 for x in frozen_validation.values():
  if x:candidates.append({'relative_l2':x['mean_relative_l2'],'cosine':x['mean_cosine'],'kv':x['mean_kv_bytes'],'v2_relative_l2':v2val['mean_relative_l2']})
 headroom=any(x['summary'] and x['summary']['mean_relative_l2']<=.12 and x['summary']['mean_cosine']>=.985 and x['summary']['mean_kv_bytes']<=uval['mean_kv_bytes'] for x in posthoc.values()) or any(x['mean_relative_l2'] is not None and x['mean_relative_l2']<=.12 and x['mean_cosine']>=.985 and x['mean_kv_bytes']<=uval['mean_kv_bytes'] for x in per_query.values())
 for x in list(frozen_validation.values())+list(baseline.values()):
  if x: x['K+V/uniform10']=x['mean_kv_bytes']/uval['mean_kv_bytes']; x['K+V/flat5']=x['mean_kv_bytes']/baseline['flat_q8k4_5']['mean_kv_bytes']
 payload={'experiment':'cascadekv_v3_vaware_development_only','status':'complete','manifest_sha256':file_sha256(MANIFEST),'calibration_lk_action_statistics':ag,'traffic_targets':targets,'calibration_frozen_tables':tables,'validation_frozen_table_metrics':frozen_validation,'validation_baselines':baseline,'layer_kv_mean_relative_l2':{'calibration':by_lk(cal),'validation':by_lk(val)},'oracles':{'posthoc_lk_static':posthoc,'per_query_action':per_query},'classification':classify(candidates,{'oracle_headroom':headroom},uval['mean_kv_bytes']),'classification_rule':'PROMISING iff validation relL2<=.12 cosine>=.985 KV<=uniform10 and improves v2; NOT-GENERALIZING iff oracle headroom but frozen tables miss; ACTION-MENU-LIMITED iff neither oracle meets both under uniform10'}
 out.write_text(json.dumps(payload,indent=2)+'\n')
def main():
 p=argparse.ArgumentParser();[p.add_argument(x,action='store_true') for x in ('--preflight','--status','--capture','--evaluate','--merge','--freeze','--validate-cache','--validate-shard')];p.add_argument('--sequence',choices=CAL+VAL);p.add_argument('--layer',type=int,choices=LAYERS);p.add_argument('--cache-dir',type=Path,default=Path('results/cascadekv_v3_vaware_dev_cache'));p.add_argument('--shard-dir',type=Path,default=Path('results/cascadekv_v3_vaware_dev_shards'));p.add_argument('--output',type=Path,default=Path('results/cascadekv_v3_vaware_dev.json'));p.add_argument('--status-output',type=Path);p.add_argument('--freeze-result',type=Path,default=Path('results/cascadekv_v3_vaware_dev.json'));p.add_argument('--freeze-config',type=Path,default=Path('configs/cascadekv_v3.json'));p.add_argument('--frozen-result',type=Path,default=Path('results/cascadekv_v3_vaware_dev_frozen.json'));a=p.parse_args()
 if a.preflight:preflight()
 elif a.status:
  x=json.dumps(status(a.cache_dir,a.shard_dir),indent=2);print(x)
  if a.status_output: a.status_output.parent.mkdir(parents=True,exist_ok=True); a.status_output.write_text(x+'\n')
 elif a.capture:capture(a.sequence,a.layer,a.cache_dir)
 elif a.evaluate:evaluate(a.sequence,a.layer,a.cache_dir,a.shard_dir)
 elif a.merge:merge(a.shard_dir,a.output)
 elif a.freeze: print(json.dumps(freeze_v3(a.freeze_result,a.freeze_config,a.frozen_result,MANIFEST,a.shard_dir,a.cache_dir),indent=2))
 elif a.validate_cache:
  ok,msg=validate_cache(cache_path(a.cache_dir,a.sequence,a.layer),a.sequence,a.layer); print('valid' if ok else f'invalid: {msg}')
  if not ok: raise SystemExit(1)
 elif a.validate_shard:
  ok,msg=validate_shard(shard_path(a.shard_dir,a.sequence,a.layer),a.sequence,a.layer); print('valid' if ok else f'invalid: {msg}')
  if not ok: raise SystemExit(1)
 else:p.error('select phase')
if __name__=='__main__':main()
