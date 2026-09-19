#!/usr/bin/env python3
"""Future-only executor bound to the frozen wide V-aware protocol."""
from __future__ import annotations
import argparse, hashlib, json, math, statistics, subprocess, sys
from contextlib import contextmanager
from itertools import product
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments import cascadekv_v3_qwen3_1p7b_vaware_dev as _n

MODEL='Qwen/Qwen3-1.7B'; MODEL_SHA='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'; CONFIG_SHA='1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197'
TAG='cascadekv-v3-qwen3-1p7b-wide-vaware-dev-protocol-freeze'; TAG_COMMIT='f0c5b5666ad089531fca68862c5be959088d49ef'
MANIFEST=Path('results/cascadekv_v3_qwen3_1p7b_wide_vaware_dev_manifest.json'); MANIFEST_SHA='654596830b06bfc6e9740c5363273bcbc408b769fabd92a69ef80fb182a84fb9'
V1=Path('configs/cascadekv_v1.json');V1_SHA='2964e4295719e696587f9853177c38399b64edc753e0630fc0d44208c719559b'
V2=Path('configs/cascadekv_v2.json');V2_SHA='ef36c5bd1b4211083ac5680e554877281f0a09e37f3dea9e4eb877afba18259c';V3=Path('configs/cascadekv_v3.json');V3_SHA='5d364bc4e351061c243a9166c0a9ccdd68c7cefd74265e4135d937b323e6e505'
NARROW=Path('results/cascadekv_v3_qwen3_1p7b_vaware_dev.json');NARROW_SHA='9e847fd4fb97c6eaa996df9661ed80e8219d149c02cd311e83fcebff580789ac'
LAYERS=(0,7,14,21,27);POSITIONS=(2047,3071,4095);ACTIONS=('A0','A1','A2','A3','A4');ACTION={'A0':(.05,'hierarchy'),'A1':(.075,'hierarchy'),'A2':(.10,'hierarchy'),'A3':(.15,'hierarchy'),'A4':(.05,'flat')}
BASELINES=('dense_exact','flat_q8k4_5','flat_q8k4_10','uniform_v1_5','uniform_v1_10','frozen_cascadekv_v2','original_frozen_cascadekv_v3_zero_shot');CAL=('narrative_calibration','report_calibration','qa_calibration');VAL=('narrative_validation','report_validation','qa_validation')
METRICS=_n.METRICS;TRAFFIC_FIELDS=_n.TRAFFIC_FIELDS;TARGET_NAMES=tuple(f'target_wide_T{i}' for i in range(6))
ACTION_SEMANTICS={'A0':[.05,'hierarchy using frozen 5%-routing profile'],'A1':[.075,'hierarchy using frozen 5%-routing profile'],'A2':[.10,'hierarchy using frozen 5%-routing profile'],'A3':[.15,'hierarchy using frozen 5%-routing profile'],'A4':[.05,'flat']}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
identity=lambda x:(x['dataset'],x.get('config'),x['split'],x['index'],x.get('stable_example_id'))
def require_inputs():
 for p,h in ((MANIFEST,MANIFEST_SHA),(V1,V1_SHA),(V2,V2_SHA),(V3,V3_SHA),(NARROW,NARROW_SHA)):
  if not p.is_file() or sha(p)!=h:raise RuntimeError(f'STOP immutable input hash differs: {p}')
 if subprocess.check_output(['git','rev-parse',f'{TAG}^{{commit}}'],text=True).strip()!=TAG_COMMIT:raise RuntimeError('STOP protocol tag differs')
 n=json.loads(NARROW.read_text())
 if (n.get('status'),n.get('classification'),n.get('winner_target_name'))!=('complete','1P7B-V-AWARE-STATIC-NOT-PROMISING',None):raise RuntimeError('STOP narrow development provenance state differs')
 return json.loads(MANIFEST.read_text())
def load_manifest():
 require_inputs();d=json.loads(MANIFEST.read_text());ss=d.get('sources',{})
 if d.get('target_model')!={'name':MODEL,'resolved_commit_sha':MODEL_SHA,'config_sha256':CONFIG_SHA} or set(ss)!=set(CAL+VAL):raise RuntimeError('STOP frozen manifest binding')
 if any(ss[s].get('role')!=('calibration' if s in CAL else 'validation') for s in CAL+VAL):raise RuntimeError('STOP role binding')
 return d
# The narrow executor is deliberately reused for its deterministic mechanics.
# Never leave its module process-wide state pointed at this protocol: tests and
# callers can safely interleave narrow and wide operations, including failures.
_NARROW_BINDINGS=('MODEL','MODEL_SHA','CONFIG_SHA','TAG','TAG_COMMIT','MANIFEST',
 'MANIFEST_SHA','V1','V1_SHA','V2','V2_SHA','V3','V3_SHA','CAL','VAL','LAYERS',
 'POSITIONS','ACTIONS','ACTION','BASELINES','require_inputs','load_manifest')
def _wide_bindings():
 return dict(MODEL=MODEL,MODEL_SHA=MODEL_SHA,CONFIG_SHA=CONFIG_SHA,TAG=TAG,
  TAG_COMMIT=TAG_COMMIT,MANIFEST=MANIFEST,MANIFEST_SHA=MANIFEST_SHA,V1=V1,
  V1_SHA=V1_SHA,V2=V2,V2_SHA=V2_SHA,V3=V3,V3_SHA=V3_SHA,CAL=CAL,VAL=VAL,
  LAYERS=LAYERS,POSITIONS=POSITIONS,ACTIONS=ACTIONS,ACTION=ACTION,
  BASELINES=BASELINES,require_inputs=require_inputs,load_manifest=load_manifest)
@contextmanager
def bound_narrow():
 """Temporarily bind every narrow global used by the wide composition path."""
 old={k:getattr(_n,k) for k in _NARROW_BINDINGS}
 try:
  for k,v in _wide_bindings().items():setattr(_n,k,v)
  yield _n
 finally:
  for k,v in old.items():setattr(_n,k,v)
# Kept as a private compatibility helper for any out-of-tree callers.  New
# code must use bound_narrow(), so this cannot accidentally create a leak.
def _bind():
 raise RuntimeError('use bound_narrow() to avoid leaking narrow bindings')
cache_path=_n.cache_path;shard_path=_n.shard_path;atomic_json=_n.atomic_json
def validate_cache(p,s,l):
 with bound_narrow():ok,msg=_n.validate_cache(p,s,l)
 if not ok:return ok,msg
 try:
  import torch
  x=torch.load(p,map_location='cpu',weights_only=False)['metadata'].get('attention_scaling');return (isinstance(x,(int,float)) and math.isfinite(x),'attention scaling')
 except Exception as e:return False,str(e)
def validate_shard(p,s,l):
 with bound_narrow():return _n.validate_shard(p,s,l)
def capture(s,l,root):
 with bound_narrow():return _n.capture(s,l,root)
def evaluate(s,l,cache,shards):
 with bound_narrow():return _n.evaluate(s,l,cache,shards)
def wide_targets(v,u,f):
 if not v<u<f:raise RuntimeError('WIDE-TARGET-BASELINE-ORDER-INVALID')
 return {'T0':v,'T1':v+.5*(u-v),'T2':u,'T3':u+(f-u)/3,'T4':u+2*(f-u)/3,'T5':f}
def construct_candidates(aa,bb):
 stats={};groups=[]
 for l,h in product(LAYERS,range(8)):
  g={}
  for a in ACTIONS:
   rs=[r for r in aa if r['layer']==l and r['kv_head']==h and r['action']==a]
   if not rs:raise RuntimeError('STOP missing calibration action observations')
   g[a]={'relative_l2':statistics.mean(r['metrics']['relative_l2_error'] for r in rs),'cosine':statistics.mean(r['metrics']['cosine_similarity'] for r in rs),'total_kv_bytes':statistics.mean(r['traffic']['total_kv_bytes'] for r in rs)};stats[f'{l}:{h}:{a}']=g[a]
  groups.append(g)
 mean=lambda m:statistics.mean(r['traffic']['total_kv_bytes'] for r in bb if r['method']==m)
 ts=wide_targets(mean('frozen_cascadekv_v2'),mean('uniform_v1_10'),mean('flat_q8k4_5'));tabs={}
 with bound_narrow():
  for n,t in ts.items():
   z=_n.optimize_calibration(groups,t)
   if z is None:raise RuntimeError('STOP infeasible wide candidate')
   table={f'{l}:{h}':a for(l,h),a in zip(product(LAYERS,range(8)),z[3])}
   if len(table)!=40:raise RuntimeError('STOP malformed wide table')
   tabs[n]={'name':'target_wide_'+n,'target_kv':t,'table':table,'calibration_objective':{'sum_relative_l2':z[1],'sum_cosine':z[2],'microbytes':z[0]}}
 return stats,ts,tabs
summary=_n.summary
def passes(x,flat5,v2,v3):return x['cosine']>=.985 and x['relative_l2']<=.120 and x['relative_l2']<v2 and x['relative_l2']<v3 and x['kv']<flat5
def choose_winner(xs,flat5,v2,v3):
 if not isinstance(xs,list) or len(xs)!=6:return None
 by={}
 for x in xs:
  if not isinstance(x,dict) or x.get('name') in by or x.get('name') not in TARGET_NAMES:return None
  by[x['name']]=x
 if set(by)!=set(TARGET_NAMES):return None
 try:return next((by[n] for n in TARGET_NAMES if passes(by[n],flat5,v2,v3)),None)
 except (KeyError,TypeError):return None
def status(cache=Path('results/cascadekv_v3_qwen3_1p7b_wide_vaware_dev_cache'),shards=Path('results/cascadekv_v3_qwen3_1p7b_wide_vaware_dev_shards')):
 load_manifest();pairs=[];c=z=0
 for s,l in product(CAL+VAL,LAYERS):
  cp,sp=cache_path(cache,s,l),shard_path(shards,s,l);a=validate_cache(cp,s,l)if cp.exists()else(False,'absent');b=validate_shard(sp,s,l)if sp.exists()else(False,'absent');c+=a[0];z+=b[0];pairs.append({'source':s,'layer':l,'cache':a[1],'action_shard':b[1]})
 return {'manifest_sha256':MANIFEST_SHA,'protocol_tag':{'name':TAG,'commit':TAG_COMMIT},'valid_caches':c,'total_caches':30,'valid_action_shards':z,'total_action_shards':30,'result_exists':Path('results/cascadekv_v3_qwen3_1p7b_wide_vaware_dev.json').exists(),'pairs':pairs}
def require_complete(cache,shards):
 """Machine-readable fail-closed 30/30 barrier used before merge."""
 s=status(cache,shards)
 if (s['valid_caches'],s['valid_action_shards']) != (30,30):
  raise RuntimeError('STOP wide execution incomplete: require 30 valid caches and 30 valid action shards')
 return s
def merge(cache,shards,out):
 bad=[];aa=[];bb=[]
 for s,l in product(CAL+VAL,LAYERS):
  cp,sp=cache_path(cache,s,l),shard_path(shards,s,l);co=validate_cache(cp,s,l)if cp.exists()else(False,'missing');so=validate_shard(sp,s,l)if sp.exists()else(False,'missing')
  if not co[0]or not so[0]:bad.append({'source':s,'layer':l,'cache':co[1],'shard':so[1]})
  else:d=json.loads(sp.read_text());aa+=d['rows'];bb+=d['baseline_rows']
 if bad:atomic_json({'status':'partial','classification':None,'winner_target_name':None,'winner_table':None,'invalid_artifacts':bad,'manifest_sha256':MANIFEST_SHA},out);return
 st,ts,tabs=construct_candidates([r for r in aa if r['role']=='calibration'],[r for r in bb if r['role']=='calibration']);ca=[r for r in aa if r['role']=='calibration'];va=[r for r in aa if r['role']=='validation'];vb=[r for r in bb if r['role']=='validation']
 # Tables are fitted solely from calibration rows.  These summaries apply the
 # already-frozen table to those same rows; they never route or fit again.
 cal_summaries={x['name']:summary([r for r in ca if r['action']==x['table'][f"{r['layer']}:{r['kv_head']}"]]) for x in tabs.values()}
 cand=[{**x,'summary':summary([r for r in va if r['action']==x['table'][f"{r['layer']}:{r['kv_head']}"]])}for x in tabs.values()];bases={m:summary([r for r in vb if r['method']==m])for m in BASELINES};gates=[{'name':x['name'],'cosine':x['summary']['mean_cosine'],'relative_l2':x['summary']['mean_relative_l2'],'kv':x['summary']['mean_total_kv_bytes']}for x in cand];g=choose_winner(gates,bases['flat_q8k4_5']['mean_total_kv_bytes'],bases['frozen_cascadekv_v2']['mean_relative_l2'],bases['original_frozen_cascadekv_v3_zero_shot']['mean_relative_l2']);by={x['name']:x for x in cand};win=None if g is None else by[g['name']]
 delta=None if not win else {m:{k:win['summary'][k]-bases[m][k]for k in ('mean_cosine','mean_relative_l2','mean_absolute_l2','mean_total_kv_bytes')}for m in ('original_frozen_cascadekv_v3_zero_shot','frozen_cascadekv_v2','uniform_v1_10','flat_q8k4_5')}
 atomic_json({'experiment':'cascadekv_v3_qwen3_1p7b_wide_vaware_development_only','development_only':True,'status':'complete','classification':'1P7B-WIDE-V-AWARE-STATIC-PROMISING'if win else'1P7B-WIDE-V-AWARE-STATIC-NOT-PROMISING','manifest_sha256':MANIFEST_SHA,'protocol_tag':{'name':TAG,'commit':TAG_COMMIT},'target_model':{'name':MODEL,'revision':MODEL_SHA,'config_sha256':CONFIG_SHA},'frozen_v1_sha256':V1_SHA,'frozen_v2_sha256':V2_SHA,'frozen_v3_sha256':V3_SHA,'narrow_result_provenance_sha256':NARROW_SHA,'sources':load_manifest()['sources'],'fresh_calibration_baselines':{'B_v2':ts['T0'],'B_u10':ts['T2'],'B_flat5':ts['T5']},'target_formulas':{'T0':'B_v2','T1':'B_v2 + 0.5*(B_u10-B_v2)','T2':'B_u10','T3':'B_u10 + (B_flat5-B_u10)/3','T4':'B_u10 + 2*(B_flat5-B_u10)/3','T5':'B_flat5'},'traffic_targets':ts,'calibration_action_statistics':st,'calibration_frozen_tables':tabs,'calibration_summaries':cal_summaries,'validation_candidates':cand,'validation_baselines':bases,'development_gates':{'min_mean_cosine':.985,'max_mean_relative_l2':.120,'strictly_better_relative_l2_than':['frozen_cascadekv_v2','original_frozen_cascadekv_v3_zero_shot'],'strictly_less_total_kv_than':'flat_q8k4_5'},'gate_objects':gates,'winner_rule':'first passing target in fixed T0,T1,T2,T3,T4,T5 order','winner_target_name':None if not win else win['name'],'winner_table':None if not win else win['table'],'winner_validation_deltas':delta},out)
def main():
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
 for x in ('preflight','capture','evaluate','status','require-complete','merge'):g.add_argument('--'+x,action='store_true')
 p.add_argument('--source',choices=CAL+VAL);p.add_argument('--layer',type=int,choices=LAYERS);p.add_argument('--cache-dir',default='results/cascadekv_v3_qwen3_1p7b_wide_vaware_dev_cache');p.add_argument('--shard-dir',default='results/cascadekv_v3_qwen3_1p7b_wide_vaware_dev_shards');p.add_argument('--output',default='results/cascadekv_v3_qwen3_1p7b_wide_vaware_dev.json');a=p.parse_args()
 if a.preflight:print(json.dumps(load_manifest(),indent=2))
 elif a.status:print(json.dumps(status(Path(a.cache_dir),Path(a.shard_dir)),indent=2))
 elif a.require_complete:print(json.dumps(require_complete(Path(a.cache_dir),Path(a.shard_dir)),indent=2))
 elif a.capture:capture(a.source,a.layer,Path(a.cache_dir))
 elif a.evaluate:evaluate(a.source,a.layer,Path(a.cache_dir),Path(a.shard_dir))
 else:merge(Path(a.cache_dir),Path(a.shard_dir),Path(a.output))
if __name__=='__main__':main()
