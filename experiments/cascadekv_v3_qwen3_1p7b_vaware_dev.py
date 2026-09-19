#!/usr/bin/env python3
"""Resumable execution harness for the frozen Qwen3-1.7B V-aware protocol."""
from __future__ import annotations
import argparse,copy,hashlib,json,math,os,statistics,tempfile,subprocess,sys
from itertools import product
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.fresh_sequence_benchmark import attention_reference,candidate_budget,frozen_routing_config,load_frozen,output_metrics,quantized_flat,reserve_ids,streaming_dataset,token_length_at_least,traffic
from experiments.qwen_partial_dot import capture_target_layer_post_rope_qkv_low_memory
from experiments.confidence_routing_v2 import initialize_route,advance_until_budget
from cascadekv.adaptive_lifting import StreamingLiftingForest
MODEL='Qwen/Qwen3-1.7B'; MODEL_SHA='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'; CONFIG_SHA='1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197'
TAG='cascadekv-v3-qwen3-1p7b-vaware-dev-protocol-freeze'; TAG_COMMIT='73a517ed88a68cd788593105b034984b1057030e'
MANIFEST=Path('results/cascadekv_v3_qwen3_1p7b_vaware_dev_manifest.json'); MANIFEST_SHA='46341cc2c9de946879ef38ab3b7e28f50de19602b814e324f92a5aff27ccf637'
# V1 is a live scientific input: it defines every hierarchy routing profile,
# including the frozen V2/V3 action-table interpretation.
V1=Path('configs/cascadekv_v1.json');V1_SHA='2964e4295719e696587f9853177c38399b64edc753e0630fc0d44208c719559b'
V2=Path('configs/cascadekv_v2.json');V2_SHA='ef36c5bd1b4211083ac5680e554877281f0a09e37f3dea9e4eb877afba18259c';V3=Path('configs/cascadekv_v3.json');V3_SHA='5d364bc4e351061c243a9166c0a9ccdd68c7cefd74265e4135d937b323e6e505';ARCHIVED=Path('results/cascadekv_v3_qwen3_1p7b_4k_test.json');ARCHIVED_SHA='776ebaca657e8913215565b882bc55b491fe0ae390fc947a1320275e253604da';ARCHIVED_MANIFEST=Path('results/cascadekv_v3_qwen3_1p7b_4k_test_manifest.json');ARCHIVED_MANIFEST_SHA='20927a2a34bb5b03faf622b0c44195e50f7a482942a3f08da0cc80ecaa568574'
LAYERS=(0,7,14,21,27);POSITIONS=(2047,3071,4095);ACTIONS=('A0','A1','A2','A3','A4'); ACTION={'A0':(.05,'hierarchy'),'A1':(.075,'hierarchy'),'A2':(.10,'hierarchy'),'A3':(.15,'hierarchy'),'A4':(.05,'flat')}; BASELINES=('dense_exact','flat_q8k4_5','flat_q8k4_10','uniform_v1_5','uniform_v1_10','frozen_cascadekv_v2','original_frozen_cascadekv_v3_zero_shot');CAL=('narrative_calibration','report_calibration','qa_calibration');VAL=('narrative_validation','report_validation','qa_validation');METRICS=('relative_exact_attention_mass','top8_recall','cosine_similarity','relative_l2_error','absolute_l2_error')
# This is the complete accounting schema emitted by traffic(), plus the
# shard-level totals.  Validators deliberately require it for every method;
# synthetic fixtures may not accidentally bless a reduced traffic schema.
TRAFFIC_FIELDS=('active_root_k4_bytes','lifting_detail_k4_bytes','grouped_variance_metadata_bytes','topology_bytes','candidate_k4_bytes','authoritative_fp16_k_bytes','routing_index_bytes','candidate_sketch_bytes','authoritative_k_bytes','total_k_bytes','total_k_vs_dense_fp16_k','total_k_vs_flat_q8k4_plus_rerank','selected_v_fp16_bytes','total_kv_bytes','total_kv_vs_dense')
SPECS={'narrative':{'dataset':'emozilla/pg19','config':None,'split':'train','text_field':'text','revision':'c021754c8e01c5b1cc83a1f549c1f97fbbb756b8'},'report':{'dataset':'ccdv/govreport-summarization','config':None,'split':'train','text_field':'report','revision':'4e21184e01ae8017e2c036e180fe5e541fef60a0'},'qa':{'dataset':'zai-org/LongBench','config':'narrativeqa','split':'test','text_field':'context','revision':'75b6d5bffbcaa2cf4da85a9fa99939b13ee5b00b','loader':'direct_parquet','parquet_prefix':'narrativeqa/'}}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def identity(x):return (x['dataset'],x.get('config'),x['split'],x['index'],x.get('stable_example_id'))
def inherited_targets(v2_mean_kv,uniform10_mean_kv):return {'T0':v2_mean_kv,'T1':v2_mean_kv+.5*(uniform10_mean_kv-v2_mean_kv),'T2':uniform10_mean_kv}
def passes(candidate,uniform10,v2,zero):return candidate['cosine']>=.985 and candidate['relative_l2']<=.120 and candidate['kv']<=uniform10 and candidate['relative_l2']<v2 and candidate['relative_l2']<zero
def choose_winner(candidates,uniform10,v2,zero):
 order=('target_vaware_T0','target_vaware_T1','target_vaware_T2')
 if not isinstance(candidates,list) or len(candidates)!=3:return None
 try:
  by={}
  for x in candidates:
   if not isinstance(x,dict) or x.get('name') in by: return None
   by[x['name']]=x
 except (KeyError,TypeError):return None
 if set(by)!=set(order):return None
 try:return next((by[n] for n in order if passes(by[n],uniform10,v2,zero)),None)
 except (KeyError,TypeError):return None
def cache_path(r,s,l):return Path(r)/f'{s}_L4096_layer{l}_qkv.pt'
def shard_path(r,s,l):return Path(r)/f'{s}_L4096_layer{l}.json'
def atomic_json(d,p):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);fd,t=tempfile.mkstemp(dir=p.parent,prefix='.'+p.name)
 try:
  with os.fdopen(fd,'w') as f:json.dump(d,f,indent=2);f.write('\n')
  os.replace(t,p)
 finally:
  if os.path.exists(t):os.unlink(t)
def require_inputs():
 for p,h in ((MANIFEST,MANIFEST_SHA),(V1,V1_SHA),(V2,V2_SHA),(V3,V3_SHA),(ARCHIVED,ARCHIVED_SHA),(ARCHIVED_MANIFEST,ARCHIVED_MANIFEST_SHA)):
  if not p.is_file() or sha(p)!=h:raise RuntimeError(f'STOP immutable input hash differs: {p}')
 if subprocess.check_output(['git','rev-parse',f'{TAG}^{{commit}}'],text=True).strip()!=TAG_COMMIT:raise RuntimeError('STOP protocol tag differs')
def load_v1():
 """Load only the provenance-verified immutable V1 routing configuration."""
 require_inputs()
 return load_frozen(V1)
def load_manifest():
 require_inputs();d=json.loads(MANIFEST.read_text());ss=d.get('sources',{})
 if d.get('target_model',{}).get('resolved_commit_sha')!=MODEL_SHA or d.get('target_model',{}).get('config_sha256')!=CONFIG_SHA or set(ss)!=set(CAL+VAL):raise RuntimeError('STOP frozen manifest binding')
 if any(ss[x].get('role')!=('calibration' if x in CAL else 'validation') for x in CAL+VAL):raise RuntimeError('STOP role binding')
 return d
def meta_ok(m,s,l):
 x=load_manifest()['sources'][s];return tuple(m.get(k) for k in ('manifest_sha256','protocol_tag','protocol_commit','model_name','model_sha','model_revision','target_config_sha256','source_identity','role','layer','context_length'))==(MANIFEST_SHA,TAG,TAG_COMMIT,MODEL,MODEL_SHA,MODEL_SHA,CONFIG_SHA,x,x['role'],l,4096)
def validate_cache(p,s,l):
 try:
  d=torch.load(p,map_location='cpu',weights_only=False);q,k,v=(d[x] for x in ('query','key','value'));m=d['metadata']
  if not meta_ok(m,s,l) or m.get('tensor_shapes')!={'q':[1,16,4096,128],'k':[1,8,4096,128],'v':[1,8,4096,128]}:return False,'metadata binding'
  if any(x.dtype!=torch.float16 for x in(q,k,v)) or (tuple(q.shape),tuple(k.shape),tuple(v.shape))!=((1,16,4096,128),(1,8,4096,128),(1,8,4096,128)):return False,'FP16 shape'
  return True,'valid'
 except Exception as e:return False,str(e)
def finite(x):return isinstance(x,dict) and all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in x.values())
def validate_shard(p,s,l):
 try:
  d=json.loads(Path(p).read_text());rows=d['rows'];bs=d['baseline_rows'];src=load_manifest()['sources'][s]
  if not meta_ok(d,s,l):return False,'metadata binding'
  for rs,key,want,n in ((rows,'action',set(ACTIONS),240),(bs,'method',set(BASELINES),336)):
   got=[(r.get('position'),r.get('q_head'),r.get(key)) for r in rs];exp={(p,h,a) for p in POSITIONS for h in range(16) for a in want}
   if len(rs)!=n or len(set(got))!=n or set(got)!=exp:return False,'row cardinality/duplicates'
   if any((r.get('source'),r.get('role'),r.get('layer'),r.get('context_length'))!=(s,src['role'],l,4096) or r.get('kv_head')!=r.get('q_head',-1)//2 or set(r.get('metrics',{}))!=set(METRICS) or set(r.get('traffic',{}))!=set(TRAFFIC_FIELDS) or not finite(r.get('metrics')) or not finite(r.get('traffic')) for r in rs):return False,'row binding/schema/nonfinite'
  return True,'valid'
 except Exception as e:return False,str(e)
def status(cache=Path('results/cascadekv_v3_qwen3_1p7b_vaware_dev_cache'),shards=Path('results/cascadekv_v3_qwen3_1p7b_vaware_dev_shards')):
 load_manifest();pairs=[];c=z=0
 for s,l in product(CAL+VAL,LAYERS):
  cp,sp=cache_path(cache,s,l),shard_path(shards,s,l);a=validate_cache(cp,s,l) if cp.exists() else(False,'absent');b=validate_shard(sp,s,l) if sp.exists() else(False,'absent');c+=a[0];z+=b[0];pairs.append({'source':s,'layer':l,'cache':a[1],'action_shard':b[1]})
 return {'manifest_sha256':MANIFEST_SHA,'valid_caches':c,'total_caches':30,'valid_action_shards':z,'total_action_shards':30,'pairs':pairs}
def resolve(s,tok,m):
 x=m['sources'][s];sp=SPECS[s.rsplit('_',1)[0]];ds,_=streaming_dataset(sp,sp['revision'])
 for i,r in enumerate(ds):
  if i==x['index']:
   t=r[sp['text_field']]
   if r.get('id',i)!=x['stable_example_id'] or not token_length_at_least(tok,t,4096)[0]:raise RuntimeError('STOP source resolve')
   return t
 raise RuntimeError('STOP source absent')
def capture(s,l,root):
 p=cache_path(root,s,l)
 if p.exists() and validate_cache(p,s,l)[0]:return
 from transformers import AutoModel,AutoTokenizer
 m=load_manifest();tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA);ids=tok(resolve(s,tok,m),return_tensors='pt',truncation=True,max_length=4096).input_ids
 model=AutoModel.from_pretrained(MODEL,revision=MODEL_SHA,dtype=torch.float16,low_cpu_mem_usage=True).eval();q,k,v=capture_target_layer_post_rope_qkv_low_memory(model,ids,l);x=m['sources'][s];md={'manifest_sha256':MANIFEST_SHA,'protocol_tag':TAG,'protocol_commit':TAG_COMMIT,'model_name':MODEL,'model_sha':MODEL_SHA,'model_revision':MODEL_SHA,'target_config_sha256':CONFIG_SHA,'source_identity':x,'role':x['role'],'layer':l,'context_length':4096,'tensor_shapes':{'q':[1,16,4096,128],'k':[1,8,4096,128],'v':[1,8,4096,128]},'attention_scaling':float(model.layers[l].self_attn.scaling),'storage':'authoritative post-norm/post-RoPE FP16 Q/K/V'}
 root=Path(root);root.mkdir(parents=True,exist_ok=True);fd,t=tempfile.mkstemp(dir=root);os.close(fd)
 try:torch.save({'metadata':md,'query':q.half(),'key':k.half(),'value':v.half()},t);os.replace(t,p)
 finally:
  if os.path.exists(t):os.unlink(t)
 if not validate_cache(p,s,l)[0]:raise RuntimeError('STOP invalid promoted cache')
def route(f,q,b,lam,res):return advance_until_budget(initialize_route(f,q,4,lam,res),b)
def evaluate_flat(frac,q,k,n):
 b=candidate_budget(n,frac)
 return set(quantized_flat(q,k,b)),traffic({'active_root_reads':0,'detail_reads':0,'expanded_internal_nodes':0,'variance_scalar_reads':0},n,b,variance=False,flat=True)
def _evaluate_hierarchy(budget_fraction,routing_profile_fraction,f,q,k,n,l,cfg=None):
 """Evaluate distinct candidate-budget and routing-profile scientific inputs."""
 b=candidate_budget(n,budget_fraction)
 cfg=load_v1() if cfg is None else cfg
 sch,lam=frozen_routing_config(cfg,l,routing_profile_fraction)
 r=route(f,q,b,lam,reserve_ids(n,sch['sink'],sch['local'],b))
 return set(r['ids']),traffic(r,n,b,variance=True)
def evaluate_frozen_action(action,f,q,k,n,l,cfg=None):
 """Original v3 A0--A4 semantics: hierarchy actions always use sch5/lam5.

 The action budget and routing-profile fraction are distinct scientific
 concepts.  In particular A2 is a 10% candidate budget with the frozen 5%
 routing schedule/lambda profile; tables name actions, never routing profiles.
 """
 budget,kind=ACTION[action]
 if kind=='flat':return evaluate_flat(budget,q,k,n)
 return _evaluate_hierarchy(budget,.05,f,q,k,n,l,cfg)
def evaluate_uniform_v1(frac,f,q,k,n,l,cfg=None):
 """Uniform baselines intentionally couple their candidate and profile fractions."""
 return _evaluate_hierarchy(frac,frac,f,q,k,n,l,cfg)
def tr(x,n):
 d={key:0. for key in TRAFFIC_FIELDS};d.update(x);d['total_kv_bytes']=d['total_k_bytes']+d['selected_v_fp16_bytes'];d['total_k_vs_dense_fp16_k']=d['total_k_bytes']/(n*256);d['total_kv_vs_dense']=d['total_kv_bytes']/(n*512);return d
def evaluate(s,l,cache,shards):
 p=shard_path(shards,s,l)
 if p.exists() and validate_shard(p,s,l)[0]:return
 if not validate_cache(cache_path(cache,s,l),s,l)[0]:raise RuntimeError('STOP invalid cache')
 d=torch.load(cache_path(cache,s,l),map_location='cpu',weights_only=False);qa,ka,va=d['query'][0].float(),d['key'][0].float(),d['value'][0].float();scale=d['metadata']['attention_scaling'];fs={};src=load_manifest()['sources'][s]
 for kv in range(8):
  f=StreamingLiftingForest(mode='binary_counter',atom_size=8,window=4,group_counts=(4,))
  for st in range(0,4096,8):
   f.append_atom(ka[kv,st:st+8])
   if st+7 in POSITIONS:fs[st+7,kv]=copy.deepcopy(f)
 rows=[];bs=[];v2=json.loads(V2.read_text())['exact_layer_head_action_table'];v3=json.loads(V3.read_text())['exact_layer_head_action_table']
 for pos,h in product(POSITIONS,range(16)):
  kv=h//2;n=pos+1;q,k,v=qa[h,pos],ka[kv,:n],va[kv,:n];ref=attention_reference(q,k,v,scale);common={'source':s,'role':src['role'],'layer':l,'context_length':4096,'position':pos,'q_head':h,'kv_head':kv}
  # Verify all immutable inputs before deriving any routing behavior from V1.
  cfg=load_v1()
  for a in ACTIONS:
   ids,x=evaluate_frozen_action(a,fs[pos,kv],q,k,n,l,cfg);rows.append({**common,'action':a,'metrics':output_metrics(q,k,v,ids,ref,scale),'traffic':tr(x,n)})
  choices={'dense_exact':(set(range(n)),{'total_k_bytes':float(n*256),'selected_v_fp16_bytes':float(n*256)}),'flat_q8k4_5':None,'flat_q8k4_10':None,'uniform_v1_5':None,'uniform_v1_10':None,'frozen_cascadekv_v2':None,'original_frozen_cascadekv_v3_zero_shot':None}
  choices['flat_q8k4_5']=evaluate_flat(.05,q,k,n)
  choices['uniform_v1_5']=evaluate_uniform_v1(.05,fs[pos,kv],q,k,n,l,cfg)
  choices['uniform_v1_10']=evaluate_uniform_v1(.10,fs[pos,kv],q,k,n,l,cfg)
  # Both frozen tables are action tables; apply original frozen-action semantics.
  choices['frozen_cascadekv_v2']=evaluate_frozen_action(v2[str((l,kv))],fs[pos,kv],q,k,n,l,cfg)
  choices['original_frozen_cascadekv_v3_zero_shot']=evaluate_frozen_action(v3[str((l,kv))],fs[pos,kv],q,k,n,l,cfg)
  choices['flat_q8k4_10']=evaluate_flat(.10,q,k,n)
  for name,(ids,x) in choices.items():bs.append({**common,'method':name,'metrics':output_metrics(q,k,v,ids,ref,scale),'traffic':tr(x,n)})
 md={'manifest_sha256':MANIFEST_SHA,'protocol_tag':TAG,'protocol_commit':TAG_COMMIT,'model_name':MODEL,'model_sha':MODEL_SHA,'model_revision':MODEL_SHA,'target_config_sha256':CONFIG_SHA,'source_identity':src,'role':src['role'],'layer':l,'context_length':4096};atomic_json({**md,'rows':rows,'baseline_rows':bs},p)
 if not validate_shard(p,s,l)[0]:raise RuntimeError('STOP invalid promoted shard')
def optimize_calibration(groups,target):
 states=[(0,0.,0.,0)];cap=round(target*len(groups)*1000)
 for g in groups:
  n=[(b+round(g[a]['total_kv_bytes']*1000),e+g[a]['relative_l2'],c+g[a]['cosine'],code*5+i) for b,e,c,code in states for i,a in enumerate(ACTIONS) if b+round(g[a]['total_kv_bytes']*1000)<=cap];by={}
  for z in sorted(n,key=lambda x:(x[0],x[1],-x[2],x[3])):by.setdefault(z[0],z)
  states=[];best=None
  for z in by.values():
   if best is None or(z[1],-z[2])<best:states.append(z);best=(z[1],-z[2])
 if not states:return None
 b,e,c,code=min(states,key=lambda x:(x[1],-x[2],x[0],x[3]));out=[]
 for _ in groups:code,i=divmod(code,5);out.append(ACTIONS[i])
 return b,e,c,tuple(reversed(out))
def construct_candidates(ca,cb):
 stats={};gs=[]
 for l,kv in product(LAYERS,range(8)):
  g={}
  for a in ACTIONS:
   rs=[r for r in ca if r['layer']==l and r['kv_head']==kv and r['action']==a];g[a]={'relative_l2':statistics.mean(r['metrics']['relative_l2_error'] for r in rs),'cosine':statistics.mean(r['metrics']['cosine_similarity'] for r in rs),'total_kv_bytes':statistics.mean(r['traffic']['total_kv_bytes'] for r in rs)};stats[f'{l}:{kv}:{a}']=g[a]
  gs.append(g)
 mean=lambda m:statistics.mean(r['traffic']['total_kv_bytes'] for r in cb if r['method']==m);v,u=mean('frozen_cascadekv_v2'),mean('uniform_v1_10');ts={'T0':v,'T1':v+.5*(u-v),'T2':u};tabs={}
 for n,t in ts.items():
  z=optimize_calibration(gs,t);tabs[n]=None if z is None else {'name':'target_vaware_'+n,'target_kv':t,'table':{f'{l}:{h}':a for(l,h),a in zip(product(LAYERS,range(8)),z[3])},'calibration_objective':{'sum_relative_l2':z[1],'sum_cosine':z[2],'microbytes':z[0]}}
 return stats,ts,tabs
def summary(rs):
 def mean(where,key):return statistics.mean(r[where][key] for r in rs)
 rel=[r['metrics']['relative_l2_error']for r in rs];cos=[r['metrics']['cosine_similarity']for r in rs];q=lambda x,p:sorted(x)[min(len(x)-1,math.ceil(len(x)*p)-1)]
 return {'mean_cosine':statistics.mean(cos),'mean_relative_l2':statistics.mean(rel),'mean_absolute_l2':mean('metrics','absolute_l2_error'),'mean_attention_mass':mean('metrics','relative_exact_attention_mass'),'mean_top8_recall':mean('metrics','top8_recall'),'mean_total_k_bytes':mean('traffic','total_k_bytes'),'mean_selected_v_bytes':mean('traffic','selected_v_fp16_bytes'),'mean_total_kv_bytes':mean('traffic','total_kv_bytes'),'p5_cosine':q(cos,.05),'p95_relative_l2':q(rel,.95),'worst_relative_l2':max(rel)}
def direct_deltas(winner,bases):
 """Validation-only report deltas; never inputs to selection."""
 keys=('mean_cosine','mean_relative_l2','mean_absolute_l2','mean_total_kv_bytes')
 return {name:{key:winner['summary'][key]-bases[name][key] for key in keys} for name in ('original_frozen_cascadekv_v3_zero_shot','frozen_cascadekv_v2','uniform_v1_10','flat_q8k4_5')}
def merge(cache,shards,out):
 bad=[];aa=[];bb=[]
 for s,l in product(CAL+VAL,LAYERS):
  c,p=cache_path(cache,s,l),shard_path(shards,s,l);co=validate_cache(c,s,l)if c.exists()else(False,'missing');so=validate_shard(p,s,l)if p.exists()else(False,'missing')
  if not co[0]or not so[0]:bad.append({'source':s,'layer':l,'cache':co[1],'shard':so[1]})
  else:d=json.loads(p.read_text());aa+=d['rows'];bb+=d['baseline_rows']
 if bad:atomic_json({'status':'partial','classification':None,'invalid_artifacts':bad,'manifest_sha256':MANIFEST_SHA},out);return
 st,ts,tabs=construct_candidates([r for r in aa if r['role']=='calibration'],[r for r in bb if r['role']=='calibration']);va=[r for r in aa if r['role']=='validation'];vb=[r for r in bb if r['role']=='validation'];cand=[]
 for n,x in tabs.items():
  if x is None:raise RuntimeError('STOP infeasible candidate')
  cand.append({**x,'summary':summary([r for r in va if r['action']==x['table'][f"{r['layer']}:{r['kv_head']}"]])})
 bases={m:summary([r for r in vb if r['method']==m])for m in BASELINES}
 # Keep selection in exactly one frozen implementation.  Rich candidate
 # records remain report data; these three gate objects are the only inputs
 # to choose_winner.
 gates=[{'name':x['name'],'cosine':x['summary']['mean_cosine'],'relative_l2':x['summary']['mean_relative_l2'],'kv':x['summary']['mean_total_kv_bytes']} for x in cand]
 winner_gate=choose_winner(gates,bases['uniform_v1_10']['mean_total_kv_bytes'],bases['frozen_cascadekv_v2']['mean_relative_l2'],bases['original_frozen_cascadekv_v3_zero_shot']['mean_relative_l2'])
 by={x['name']:x for x in cand};winner=None if winner_gate is None else by[winner_gate['name']]
 atomic_json({'experiment':'cascadekv_v3_qwen3_1p7b_vaware_development_only','development_only':True,'status':'complete','classification':'1P7B-V-AWARE-STATIC-PROMISING' if winner else '1P7B-V-AWARE-STATIC-NOT-PROMISING','manifest_sha256':MANIFEST_SHA,'protocol_tag':{'name':TAG,'commit':TAG_COMMIT},'target_model':{'name':MODEL,'revision':MODEL_SHA,'config_sha256':CONFIG_SHA},'frozen_v1_sha256':V1_SHA,'frozen_v2_sha256':V2_SHA,'frozen_v3_sha256':V3_SHA,'sources':load_manifest()['sources'],'traffic_targets':ts,'calibration_action_statistics':st,'calibration_frozen_tables':tabs,'validation_candidates':cand,'validation_baselines':bases,'development_success_rule':'a candidate must meet the frozen validation gates; otherwise static V-aware is not promising','winner_rule':'first passing target in fixed T0,T1,T2 order','development_gates':{'min_mean_cosine':.985,'max_mean_relative_l2':.120,'max_mean_total_kv_bytes':'uniform_v1_10','strictly_better_mean_relative_l2_than':['frozen_cascadekv_v2','original_frozen_cascadekv_v3_zero_shot']},'winner_target_name':winner['name']if winner else None,'winner_table':winner['table']if winner else None,'winner_validation_deltas':direct_deltas(winner,bases) if winner else None},out)
def main():
 p=argparse.ArgumentParser();[p.add_argument(x,action='store_true')for x in('--preflight','--status','--capture','--evaluate','--merge')];p.add_argument('--source',choices=CAL+VAL);p.add_argument('--layer',type=int,choices=LAYERS);p.add_argument('--cache-dir',type=Path,default=Path('results/cascadekv_v3_qwen3_1p7b_vaware_dev_cache'));p.add_argument('--shard-dir',type=Path,default=Path('results/cascadekv_v3_qwen3_1p7b_vaware_dev_shards'));p.add_argument('--output',type=Path,default=Path('results/cascadekv_v3_qwen3_1p7b_vaware_dev.json'));a=p.parse_args()
 if a.preflight:load_manifest();print('frozen protocol manifest verified')
 elif a.status:print(json.dumps(status(a.cache_dir,a.shard_dir),indent=2))
 elif a.capture:
  if a.source is None or a.layer is None:p.error('--capture requires --source --layer')
  capture(a.source,a.layer,a.cache_dir)
 elif a.evaluate:
  if a.source is None or a.layer is None:p.error('--evaluate requires --source --layer')
  evaluate(a.source,a.layer,a.cache_dir,a.shard_dir)
 elif a.merge:merge(a.cache_dir,a.shard_dir,a.output)
 else:p.error('select phase')
if __name__=='__main__':main()
