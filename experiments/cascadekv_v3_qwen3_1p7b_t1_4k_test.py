#!/usr/bin/env python3
"""Fail-closed future executor for the immutable Qwen3-1.7B T1 test.

It intentionally contains no calibration, optimizer, candidate construction,
or schedule-selection path. Production operations are only reached by explicit
CLI phases; focused tests replace their expensive boundaries.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, os, statistics, subprocess, tempfile, sys
from itertools import product
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
CONFIG=ROOT/'configs/cascadekv_v3_qwen3_1p7b_t1.json'; MANIFEST=ROOT/'results/cascadekv_v3_qwen3_1p7b_t1_4k_test_manifest.json'
DEVELOPMENT=ROOT/'results/cascadekv_v3_qwen3_1p7b_wide_vaware_dev.json'; DEVELOPMENT_MANIFEST=ROOT/'results/cascadekv_v3_qwen3_1p7b_wide_vaware_dev_manifest.json'
CONFIG_SHA='5995a0f7fd6c3ebe762f27959d10de0398980e1aa013b669379f6b794cf5e88e'; MANIFEST_SHA='53af3b7dfe1dc7ccdd994851e4e22d329cf888f02c11d7f76e858735e106890b'; DEVELOPMENT_SHA='16dad49612d5e02742e8b6cf75f7cac3f3a692668228fe36056beb32a02b4b2e'; DEVELOPMENT_MANIFEST_SHA='654596830b06bfc6e9740c5363273bcbc408b769fabd92a69ef80fb182a84fb9'
MODEL='Qwen/Qwen3-1.7B'; REVISION='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'; MODEL_CONFIG_SHA='1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197'
TAG='cascadekv-v3-qwen3-1p7b-t1-4k-test-protocol-freeze'; TAG_COMMIT='a45918680e45026c180ce3696d16009603398183'
V1=ROOT/'configs/cascadekv_v1.json';V2=ROOT/'configs/cascadekv_v2.json';V3=ROOT/'configs/cascadekv_v3.json';V1_SHA='2964e4295719e696587f9853177c38399b64edc753e0630fc0d44208c719559b';V2_SHA='ef36c5bd1b4211083ac5680e554877281f0a09e37f3dea9e4eb877afba18259c';V3_SHA='5d364bc4e351061c243a9166c0a9ccdd68c7cefd74265e4135d937b323e6e505'
SOURCES=('narrative','report','qa');LAYERS=(0,7,14,21,27);POSITIONS=(2047,3071,4095)
METHODS=('dense_exact','flat_q8k4_5','flat_q8k4_10','uniform_v1_5','uniform_v1_10','frozen_cascadekv_v2','original_frozen_cascadekv_v3_zero_shot','frozen_qwen3_1p7b_T1');METRICS=('relative_exact_attention_mass','top8_recall','cosine_similarity','relative_l2_error','absolute_l2_error');TRAFFIC_FIELDS=('active_root_k4_bytes','lifting_detail_k4_bytes','grouped_variance_metadata_bytes','topology_bytes','candidate_k4_bytes','authoritative_fp16_k_bytes','routing_index_bytes','candidate_sketch_bytes','authoritative_k_bytes','total_k_bytes','total_k_vs_dense_fp16_k','total_k_vs_flat_q8k4_plus_rerank','selected_v_fp16_bytes','total_kv_bytes','total_kv_vs_dense')
CACHE_DIR=ROOT/'results/cascadekv_v3_qwen3_1p7b_t1_4k_test_cache';SHARD_DIR=ROOT/'results/cascadekv_v3_qwen3_1p7b_t1_4k_test_shards';RESULT=ROOT/'results/cascadekv_v3_qwen3_1p7b_t1_4k_test.json';LABELS={'invalid':'T1-CONFIRMATORY-TEST-INVALID','absolute_fail':'T1-CONFIRMATORY-OUTPUT-GATE-FAILED','comparative_fail':'T1-CONFIRMATORY-COMPARATIVE-GATE-FAILED','pass':'T1-CONFIRMATORY-TEST-PASSED'}
EXPECTED_TABLE=json.loads(CONFIG.read_text())['exact_layer_head_action_table']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cache_path(r,s,l):return Path(r)/f'{s}_L4096_layer{l}_qkv.pt'
def shard_path(r,s,l):return Path(r)/f'{s}_L4096_layer{l}.json'
def atomic_json(d,p):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);fd,t=tempfile.mkstemp(dir=p.parent,prefix='.'+p.name)
 try:
  with os.fdopen(fd,'w')as f:json.dump(d,f,indent=2);f.write('\n')
  os.replace(t,p)
 finally:
  if os.path.exists(t):os.unlink(t)
def atomic_torch(d,p):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);fd,t=tempfile.mkstemp(dir=p.parent,prefix='.'+p.name);os.close(fd)
 try:torch.save(d,t);os.replace(t,p)
 finally:
  if os.path.exists(t):os.unlink(t)
def require_inputs():
 for p,h in ((CONFIG,CONFIG_SHA),(MANIFEST,MANIFEST_SHA),(DEVELOPMENT,DEVELOPMENT_SHA),(DEVELOPMENT_MANIFEST,DEVELOPMENT_MANIFEST_SHA),(V1,V1_SHA),(V2,V2_SHA),(V3,V3_SHA)):
  if not p.is_file()or sha(p)!=h:raise RuntimeError(f'{LABELS["invalid"]}: immutable hash mismatch: {p.name}')
 if subprocess.check_output(['git','rev-parse',f'{TAG}^{{commit}}'],text=True).strip()!=TAG_COMMIT:raise RuntimeError(f'{LABELS["invalid"]}: protocol tag mismatch')
 c,m,d=(json.loads(p.read_text())for p in(CONFIG,MANIFEST,DEVELOPMENT))
 if(c.get('target_model')!={'name':MODEL,'revision':REVISION,'config_sha256':MODEL_CONFIG_SHA}or(d.get('status'),d.get('classification'),d.get('winner_target_name'))!=('complete','1P7B-WIDE-V-AWARE-STATIC-PROMISING','target_wide_T1')or c.get('schedule_cell_count')!=40 or len(c.get('exact_layer_head_action_table',{}))!=40):raise RuntimeError(f'{LABELS["invalid"]}: frozen provenance')
 return c,m,d
def load_manifest():
 c,m,_=require_inputs();ss=m.get('sources',{})
 if m.get('target_t1_config',{}).get('sha256')!=CONFIG_SHA or tuple(m.get('methods',()))!=METHODS or set(ss)!=set(SOURCES)or any(ss[s].get('index')!={'narrative':11,'report':12,'qa':11}[s]for s in SOURCES)or c['action_semantics']['A2']!={'candidate_budget':.10,'routing_profile':'frozen 5%-routing profile','mode':'hierarchy'}:raise RuntimeError(f'{LABELS["invalid"]}: manifest/action binding')
 return m
def preflight():return load_manifest()
def _meta(s,l):return {'manifest_sha256':MANIFEST_SHA,'t1_config_sha256':CONFIG_SHA,'protocol_tag':TAG,'protocol_commit':TAG_COMMIT,'model_name':MODEL,'model_sha':REVISION,'model_revision':REVISION,'target_config_sha256':MODEL_CONFIG_SHA,'source_identity':load_manifest()['sources'][s],'layer':l,'context_length':4096}
def _meta_ok(d,s,l):return all(d.get(k)==v for k,v in _meta(s,l).items())
def finite(x):return isinstance(x,dict)and all(isinstance(v,(int,float))and not isinstance(v,bool)and math.isfinite(v)for v in x.values())
def validate_cache(p,s,l):
 try:
  d=torch.load(p,map_location='cpu',weights_only=False);q,k,v=d['query'],d['key'],d['value'];m=d['metadata']
  if not _meta_ok(m,s,l):return False,'provenance'
  if m.get('tensor_shapes')!={'q':[1,16,4096,128],'k':[1,8,4096,128],'v':[1,8,4096,128]}or tuple(q.shape)!=(1,16,4096,128)or tuple(k.shape)!=(1,8,4096,128)or tuple(v.shape)!=(1,8,4096,128)or any(x.dtype!=torch.float16 for x in(q,k,v)):return False,'FP16 shape'
  return(isinstance(m.get('attention_scaling'),(int,float))and math.isfinite(m['attention_scaling']),'attention scaling')
 except Exception as e:return False,str(e)
def validate_shard(p,s,l):
 try:
  d=json.loads(Path(p).read_text());rows=d['rows'];got=[(r.get('position'),r.get('q_head'),r.get('method'))for r in rows];want={(p,h,m)for p in POSITIONS for h in range(16)for m in METHODS}
  if not _meta_ok(d,s,l):return False,'provenance'
  if len(rows)!=384 or len(set(got))!=384 or set(got)!=want:return False,'row cardinality/duplicates'
  if any((r.get('source'),r.get('layer'),r.get('context_length'))!=(s,l,4096)or r.get('kv_head')!=r.get('q_head',-1)//2 or set(r.get('metrics',{}))!=set(METRICS)or set(r.get('traffic',{}))!=set(TRAFFIC_FIELDS)or not finite(r['metrics'])or not finite(r['traffic'])for r in rows):return False,'row binding/schema/nonfinite'
  return True,'valid'
 except Exception as e:return False,str(e)
def status(cache=CACHE_DIR,shards=SHARD_DIR):
 load_manifest();pairs=[];c=z=0
 for s,l in product(SOURCES,LAYERS):
  cp,sp=cache_path(cache,s,l),shard_path(shards,s,l);a=validate_cache(cp,s,l)if cp.exists()else(False,'absent');b=validate_shard(sp,s,l)if sp.exists()else(False,'absent');c+=a[0];z+=b[0];pairs.append({'source':s,'layer':l,'cache':a[1],'shard':b[1]})
 return {'manifest_sha256':MANIFEST_SHA,'protocol_tag':{'name':TAG,'commit':TAG_COMMIT},'valid_caches':c,'total_caches':15,'valid_shards':z,'total_shards':15,'result_exists':RESULT.exists(),'pairs':pairs}
def require_complete(cache=CACHE_DIR,shards=SHARD_DIR):
 x=status(cache,shards)
 if(x['valid_caches'],x['valid_shards'])!=(15,15):raise RuntimeError(f'{LABELS["invalid"]}: execution incomplete')
 return x

def capture(s,l,root=CACHE_DIR):
 p=cache_path(root,s,l)
 if p.exists()and validate_cache(p,s,l)[0]:return
 from transformers import AutoModel,AutoTokenizer
 from experiments.cascadekv_v3_qwen3_1p7b_vaware_dev import resolve
 from experiments.qwen_partial_dot import capture_target_layer_post_rope_qkv_low_memory
 m=load_manifest();tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION);ids=tok(resolve(s,tok,m),return_tensors='pt',truncation=True,max_length=4096).input_ids;model=AutoModel.from_pretrained(MODEL,revision=REVISION,dtype=torch.float16,low_cpu_mem_usage=True).eval();q,k,v=capture_target_layer_post_rope_qkv_low_memory(model,ids,l)
 md={**_meta(s,l),'tensor_shapes':{'q':[1,16,4096,128],'k':[1,8,4096,128],'v':[1,8,4096,128]},'attention_scaling':float(model.layers[l].self_attn.scaling),'storage':'authoritative post-norm/post-RoPE FP16 Q/K/V'};atomic_torch({'metadata':md,'query':q.half(),'key':k.half(),'value':v.half()},p)
 if not validate_cache(p,s,l)[0]:raise RuntimeError(f'{LABELS["invalid"]}: invalid promoted cache')

def _tr(x,n):
 d={k:0. for k in TRAFFIC_FIELDS};d.update(x);d['total_kv_bytes']=d['total_k_bytes']+d['selected_v_fp16_bytes'];d['total_k_vs_dense_fp16_k']=d['total_k_bytes']/(n*256);d['total_kv_vs_dense']=d['total_kv_bytes']/(n*512);return d
def frozen_table_action(table,layer,kv_head,table_name):
 """Read the frozen tables in their on-disk, deliberately different formats."""
 key=f'{layer}:{kv_head}' if table_name=='t1' else str((layer,kv_head))
 try:return table[key]
 except KeyError as e:raise RuntimeError(f'{LABELS["invalid"]}: missing {table_name} action {key}') from e
def action_profile(action_semantics,action):
 """Frozen actions never turn A2 into uniform-10 routing."""
 x=action_semantics[action]
 return x['candidate_budget'],(None if x['mode']=='flat' else .05),x['mode']
def evaluate(s,l,cache=CACHE_DIR,shards=SHARD_DIR):
 p=shard_path(shards,s,l)
 if p.exists()and validate_shard(p,s,l)[0]:return
 if not validate_cache(cache_path(cache,s,l),s,l)[0]:raise RuntimeError(f'{LABELS["invalid"]}: invalid cache')
 from experiments.fresh_sequence_benchmark import attention_reference,candidate_budget,frozen_routing_config,load_frozen,output_metrics,quantized_flat,reserve_ids,traffic
 from experiments.confidence_routing_v2 import initialize_route,advance_until_budget
 from cascadekv.adaptive_lifting import StreamingLiftingForest
 d=torch.load(cache_path(cache,s,l),map_location='cpu',weights_only=False);qa,ka,va=d['query'][0].float(),d['key'][0].float(),d['value'][0].float();scale=d['metadata']['attention_scaling'];cfg=load_frozen(V1);fs={}
 for kv in range(8):
  f=StreamingLiftingForest(mode='binary_counter',atom_size=8,window=4,group_counts=(4,))
  for st in range(0,4096,8):
   f.append_atom(ka[kv,st:st+8])
   if st+7 in POSITIONS:fs[st+7,kv]=copy.deepcopy(f)
 def flat(fr,q,k,n):b=candidate_budget(n,fr);return set(quantized_flat(q,k,b)),traffic({'active_root_reads':0,'detail_reads':0,'expanded_internal_nodes':0,'variance_scalar_reads':0},n,b,variance=False,flat=True)
 def hier(fr,prof,f,q,n):
  b=candidate_budget(n,fr);sch,lam=frozen_routing_config(cfg,l,prof);r=advance_until_budget(initialize_route(f,q,4,lam,reserve_ids(n,sch['sink'],sch['local'],b)),b);return set(r['ids']),traffic(r,n,b,variance=True)
 semantics=json.loads(CONFIG.read_text())['action_semantics']
 def action(a,f,q,k,n):
  budget,profile,mode=action_profile(semantics,a)
  return flat(budget,q,k,n)if mode=='flat'else hier(budget,profile,f,q,n)
 v2=json.loads(V2.read_text())['exact_layer_head_action_table'];v3=json.loads(V3.read_text())['exact_layer_head_action_table'];table=json.loads(CONFIG.read_text())['exact_layer_head_action_table'];rows=[]
 for pos,h in product(POSITIONS,range(16)):
  kv=h//2;n=pos+1;q,k,v=qa[h,pos],ka[kv,:n],va[kv,:n];ref=attention_reference(q,k,v,scale);f=fs[pos,kv];common={'source':s,'layer':l,'context_length':4096,'position':pos,'q_head':h,'kv_head':kv};choices={'dense_exact':(set(range(n)),{'total_k_bytes':float(n*256),'selected_v_fp16_bytes':float(n*256)}),'flat_q8k4_5':flat(.05,q,k,n),'flat_q8k4_10':flat(.10,q,k,n),'uniform_v1_5':hier(.05,.05,f,q,n),'uniform_v1_10':hier(.10,.10,f,q,n),'frozen_cascadekv_v2':action(frozen_table_action(v2,l,kv,'v2'),f,q,k,n),'original_frozen_cascadekv_v3_zero_shot':action(frozen_table_action(v3,l,kv,'v3'),f,q,k,n),'frozen_qwen3_1p7b_T1':action(frozen_table_action(table,l,kv,'t1'),f,q,k,n)}
  for name,(ids,x)in choices.items():rows.append({**common,'method':name,'metrics':output_metrics(q,k,v,ids,ref,scale),'traffic':_tr(x,n)})
 atomic_json({**_meta(s,l),'rows':rows},p)
 if not validate_shard(p,s,l)[0]:raise RuntimeError(f'{LABELS["invalid"]}: invalid promoted shard')
def summary(rows):
 def mean(w,k):return statistics.mean(r[w][k]for r in rows)
 cos=[r['metrics']['cosine_similarity']for r in rows];rel=[r['metrics']['relative_l2_error']for r in rows];q=lambda x,p:sorted(x)[min(len(x)-1,math.ceil(len(x)*p)-1)]
 return {'row_count':len(rows),'metrics':{'cosine_similarity':{'mean':statistics.mean(cos),'p5':q(cos,.05)},'relative_l2_error':{'mean':statistics.mean(rel),'p95':q(rel,.95),'worst':max(rel)},'absolute_l2_error':{'mean':mean('metrics','absolute_l2_error')},'relative_exact_attention_mass':{'mean':mean('metrics','relative_exact_attention_mass')},'top8_recall':{'mean':mean('metrics','top8_recall')}},'traffic':{'total_k_bytes':{'mean':mean('traffic','total_k_bytes')},'selected_v_fp16_bytes':{'mean':mean('traffic','selected_v_fp16_bytes')},'total_kv_bytes':{'mean':mean('traffic','total_kv_bytes')}}}
def classify(pooled):
 try:
  t,v2,v3,f=(pooled[x]for x in('frozen_qwen3_1p7b_T1','frozen_cascadekv_v2','original_frozen_cascadekv_v3_zero_shot','flat_q8k4_5'));c=t['metrics']['cosine_similarity']['mean'];r=t['metrics']['relative_l2_error']['mean'];b=t['traffic']['total_kv_bytes']['mean']
  if c<.985 or r>.120:return LABELS['absolute_fail']
  if r>=v2['metrics']['relative_l2_error']['mean']or r>=v3['metrics']['relative_l2_error']['mean']or b>=f['traffic']['total_kv_bytes']['mean']:return LABELS['comparative_fail']
  return LABELS['pass']
 except(KeyError,TypeError):return LABELS['invalid']
def merge(cache=CACHE_DIR,shards=SHARD_DIR,out=RESULT):
 bad=[];rows=[]
 for s,l in product(SOURCES,LAYERS):
  cp,sp=cache_path(cache,s,l),shard_path(shards,s,l);c=validate_cache(cp,s,l)if cp.exists()else(False,'missing');z=validate_shard(sp,s,l)if sp.exists()else(False,'missing')
  if not c[0]or not z[0]:bad.append({'source':s,'layer':l,'cache':c[1],'shard':z[1]})
  else:rows+=json.loads(sp.read_text())['rows']
 if bad:atomic_json({'experiment':'cascadekv_v3_qwen3_1p7b_t1_4k_confirmatory_test','confirmatory_test':True,'untouched_holdout':True,'status':'partial','classification':LABELS['invalid'],'invalid_artifacts':bad,'test_manifest_sha256':MANIFEST_SHA,'t1_config_sha256':CONFIG_SHA,'protocol_tag':{'name':TAG,'commit':TAG_COMMIT}},out);return
 pooled={m:summary([r for r in rows if r['method']==m])for m in METHODS};label=classify(pooled);t=pooled['frozen_qwen3_1p7b_T1'];delta=lambda m:{'mean_cosine':t['metrics']['cosine_similarity']['mean']-pooled[m]['metrics']['cosine_similarity']['mean'],'mean_relative_l2':t['metrics']['relative_l2_error']['mean']-pooled[m]['metrics']['relative_l2_error']['mean'],'mean_total_kv_bytes':t['traffic']['total_kv_bytes']['mean']-pooled[m]['traffic']['total_kv_bytes']['mean']}
 atomic_json({'experiment':'cascadekv_v3_qwen3_1p7b_t1_4k_confirmatory_test','confirmatory_test':True,'untouched_holdout':True,'status':'complete','classification':label,'test_manifest_path':str(MANIFEST.relative_to(ROOT)),'test_manifest_sha256':MANIFEST_SHA,'t1_config_path':str(CONFIG.relative_to(ROOT)),'t1_config_sha256':CONFIG_SHA,'protocol_tag':{'name':TAG,'commit':TAG_COMMIT},'development_result_sha256':DEVELOPMENT_SHA,'development_manifest_sha256':DEVELOPMENT_MANIFEST_SHA,'target_model':{'name':MODEL,'revision':REVISION,'config_sha256':MODEL_CONFIG_SHA},'frozen_v1_sha256':V1_SHA,'frozen_v2_sha256':V2_SHA,'frozen_v3_sha256':V3_SHA,'sources':load_manifest()['sources'],'geometry':{'context':4096,'layers':list(LAYERS),'positions':list(POSITIONS),'q_heads':list(range(16)),'kv_head':'q_head // 2'},'methods':list(METHODS),'pooled_summaries':pooled,'absolute_gates':{'cosine_gte':.985,'relative_l2_lte':.120},'comparative_gates':{'relative_l2_lt':['frozen_cascadekv_v2','original_frozen_cascadekv_v3_zero_shot'],'total_kv_lt':'flat_q8k4_5'},'failure_label_rules':LABELS,'t1_gate_margins':{'cosine':t['metrics']['cosine_similarity']['mean']-.985,'relative_l2':.120-t['metrics']['relative_l2_error']['mean']},'t1_deltas':{m:delta(m)for m in('frozen_cascadekv_v2','original_frozen_cascadekv_v3_zero_shot','uniform_v1_10','flat_q8k4_5','dense_exact')},'no_fitting_or_schedule_selection_occurred_on_test_data':True},out)
def main():
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
 for x in('preflight','status','capture','evaluate','require-complete','merge'):g.add_argument('--'+x,action='store_true')
 p.add_argument('--source',choices=SOURCES);p.add_argument('--layer',type=int,choices=LAYERS);p.add_argument('--cache-dir',type=Path,default=CACHE_DIR);p.add_argument('--shard-dir',type=Path,default=SHARD_DIR);p.add_argument('--output',type=Path,default=RESULT);a=p.parse_args()
 if a.preflight:print(json.dumps(preflight(),indent=2))
 elif a.status:print(json.dumps(status(a.cache_dir,a.shard_dir),indent=2))
 elif a.require_complete:print(json.dumps(require_complete(a.cache_dir,a.shard_dir),indent=2))
 elif a.capture:
  if a.source is None or a.layer is None:p.error('--capture requires --source --layer')
  capture(a.source,a.layer,a.cache_dir)
 elif a.evaluate:
  if a.source is None or a.layer is None:p.error('--evaluate requires --source --layer')
  evaluate(a.source,a.layer,a.cache_dir,a.shard_dir)
 else:merge(a.cache_dir,a.shard_dir,a.output)
if __name__=='__main__':main()
