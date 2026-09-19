import copy, hashlib, json, sys, types
from pathlib import Path
import pytest, torch
from experiments import cascadekv_v3_qwen3_1p7b_t1_4k_test as t

def doc(s='narrative',l=0):
 return {'metadata':{**t._meta(s,l),'tensor_shapes':{'q':[1,16,4096,128],'k':[1,8,4096,128],'v':[1,8,4096,128]},'attention_scaling':.125},'query':torch.empty((1,16,4096,128),dtype=torch.float16),'key':torch.empty((1,8,4096,128),dtype=torch.float16),'value':torch.empty((1,8,4096,128),dtype=torch.float16)}
def cache(root,s='narrative',l=0):
 p=t.cache_path(root,s,l);t.atomic_torch(doc(s,l),p);assert t.validate_cache(p,s,l)[0];return p
def row(s,l,p,h,m,cos=.99,l2=.1,kv=9):
 tr={x:0. for x in t.TRAFFIC_FIELDS};tr.update(total_k_bytes=kv/2,selected_v_fp16_bytes=kv/2,total_kv_bytes=kv)
 return {'source':s,'layer':l,'context_length':4096,'position':p,'q_head':h,'kv_head':h//2,'method':m,'metrics':{'relative_exact_attention_mass':1.,'top8_recall':1.,'cosine_similarity':cos,'relative_l2_error':l2,'absolute_l2_error':l2},'traffic':tr}
def shard(root,s='narrative',l=0,vals={}):
 rs=[row(s,l,p,h,m,*vals.get(m,(.99,.1,9)))for p in t.POSITIONS for h in range(16)for m in t.METHODS];q=t.shard_path(root,s,l);t.atomic_json({**t._meta(s,l),'rows':rs},q);assert t.validate_shard(q,s,l)[0];return q

def test_01_frozen_hashes_and_v2_v3_tuple_table_keys():
 c,m,d=t.require_inputs();v2=json.loads(t.V2.read_text())['exact_layer_head_action_table'];v3=json.loads(t.V3.read_text())['exact_layer_head_action_table']
 assert hashlib.sha256(t.CONFIG.read_bytes()).hexdigest()==t.CONFIG_SHA and hashlib.sha256(t.MANIFEST.read_bytes()).hexdigest()==t.MANIFEST_SHA
 assert all(k.startswith('(')for k in v2)and all(k.startswith('(')for k in v3)and all(':'in k for k in c['exact_layer_head_action_table'])
 assert t.frozen_table_action(v2,0,0,'v2')==v2['(0, 0)'] and t.frozen_table_action(v3,0,0,'v3')==v3['(0, 0)'] and d['winner_table']==t.EXPECTED_TABLE
def test_02_manifest_identities_and_sources():
 m=t.load_manifest();assert {s:m['sources'][s]['index']for s in t.SOURCES}=={'narrative':11,'report':12,'qa':11}
 assert [(m['sources'][s]['dataset'],m['sources'][s]['field'],m['sources'][s]['revision'])for s in t.SOURCES]==[('emozilla/pg19','text','c021754c8e01c5b1cc83a1f549c1f97fbbb756b8'),('ccdv/govreport-summarization','report','4e21184e01ae8017e2c036e180fe5e541fef60a0'),('zai-org/LongBench','context','75b6d5bffbcaa2cf4da85a9fa99939b13ee5b00b')]
def test_03_action_profiles_and_frozen_a2():
 x=json.loads(t.CONFIG.read_text())['action_semantics'];assert [t.action_profile(x,a)for a in ('A0','A1','A2','A3','A4')]==[(.05,.05,'hierarchy'),(.075,.05,'hierarchy'),(.1,.05,'hierarchy'),(.15,.05,'hierarchy'),(.05,None,'flat')]
def test_04_valid_cache_and_resume(tmp_path):
 cache(tmp_path);assert t.capture('narrative',0,tmp_path)is None
def test_05_cache_rejects_provenance_matrix(tmp_path):
 p=cache(tmp_path);b=torch.load(p,weights_only=False)
 for k,v in [('manifest_sha256','x'),('t1_config_sha256','x'),('protocol_tag','x'),('protocol_commit','x'),('model_name','x'),('model_revision','x'),('model_sha','x'),('target_config_sha256','x'),('source_identity',{}),('layer',7),('context_length',1)]:
  d=copy.deepcopy(b);d['metadata'][k]=v;t.atomic_torch(d,p);assert not t.validate_cache(p,'narrative',0)[0]
def test_06_cache_rejects_shape_dtype_scaling_matrix(tmp_path):
 p=cache(tmp_path);b=torch.load(p,weights_only=False)
 for k in ('query','key','value'):
  d=copy.deepcopy(b);d[k]=d[k][...,:127];t.atomic_torch(d,p);assert not t.validate_cache(p,'narrative',0)[0]
 d=copy.deepcopy(b);d['query']=d['query'].float();t.atomic_torch(d,p);assert not t.validate_cache(p,'narrative',0)[0]
 for z in (None,float('nan'),float('inf')):
  d=copy.deepcopy(b);d['metadata'].pop('attention_scaling',None)if z is None else d['metadata'].update(attention_scaling=z);t.atomic_torch(d,p);assert not t.validate_cache(p,'narrative',0)[0]
def test_07_valid_shard_schema_384(tmp_path):
 p=shard(tmp_path);assert t.validate_shard(p,'narrative',0)==(True,'valid');rs=json.loads(p.read_text())['rows'];assert len(rs)==384 and all(sum(r['method']==m for r in rs)==48 for m in t.METHODS)
def test_08_shard_rejects_provenance_and_identity(tmp_path):
 p=shard(tmp_path);b=json.loads(p.read_text())
 for k,v in [('manifest_sha256','x'),('t1_config_sha256','x'),('protocol_tag','x'),('protocol_commit','x'),('source_identity',{}),('layer',7),('context_length',1)]:
  d=copy.deepcopy(b);d[k]=v;t.atomic_json(d,p);assert not t.validate_shard(p,'narrative',0)[0]
def test_09_shard_rejects_rows_metrics_traffic(tmp_path):
 p=shard(tmp_path);b=json.loads(p.read_text());fs=[lambda d:d['rows'].pop(),lambda d:d['rows'].append(copy.deepcopy(d['rows'][0])),lambda d:d['rows'][0].update(method='bad'),lambda d:d['rows'][0].update(position=1),lambda d:d['rows'][0].update(q_head=17),lambda d:d['rows'][0].update(kv_head=7),lambda d:d['rows'][0]['metrics'].update(cosine_similarity=float('nan')),lambda d:d['rows'][0]['traffic'].update(total_k_bytes=float('inf')),lambda d:d['rows'][0]['metrics'].pop('top8_recall'),lambda d:d['rows'][0]['metrics'].update(x=1),lambda d:d['rows'][0]['traffic'].pop('total_k_bytes'),lambda d:d['rows'][0]['traffic'].update(x=1)]
 for f in fs:
  d=copy.deepcopy(b);f(d);t.atomic_json(d,p);assert not t.validate_shard(p,'narrative',0)[0]
def test_10_evaluate_resume_and_missing_cache_fail_closed(tmp_path):
 shard(tmp_path);assert t.evaluate('narrative',0,tmp_path,tmp_path)is None
 with pytest.raises(RuntimeError,match='invalid cache'):t.evaluate('narrative',0,tmp_path/'missing',tmp_path/'other')
def test_11_classification_absolute_gates():
 def p(c=.99,r=.1):return {'frozen_qwen3_1p7b_T1':{'metrics':{'cosine_similarity':{'mean':c},'relative_l2_error':{'mean':r}},'traffic':{'total_kv_bytes':{'mean':9}}},'frozen_cascadekv_v2':{'metrics':{'relative_l2_error':{'mean':.11}}},'original_frozen_cascadekv_v3_zero_shot':{'metrics':{'relative_l2_error':{'mean':.11}}},'flat_q8k4_5':{'traffic':{'total_kv_bytes':{'mean':10}}}}
 assert t.classify(p(.984))==t.LABELS['absolute_fail']and t.classify(p(.99,.121))==t.LABELS['absolute_fail']
def test_12_classification_comparative_and_uniform10_not_gate():
 base={'frozen_qwen3_1p7b_T1':{'metrics':{'cosine_similarity':{'mean':.99},'relative_l2_error':{'mean':.1}},'traffic':{'total_kv_bytes':{'mean':99}}},'frozen_cascadekv_v2':{'metrics':{'relative_l2_error':{'mean':.11}}},'original_frozen_cascadekv_v3_zero_shot':{'metrics':{'relative_l2_error':{'mean':.11}}},'flat_q8k4_5':{'traffic':{'total_kv_bytes':{'mean':100}}}}
 assert t.classify(base)==t.LABELS['pass'];base['frozen_cascadekv_v2']['metrics']['relative_l2_error']['mean']=.1;assert t.classify(base)==t.LABELS['comparative_fail']
def test_13_nonadaptivity_static_path():
 before=(t.CONFIG_SHA,t.MANIFEST_SHA,copy.deepcopy(t.EXPECTED_TABLE),t.METHODS,json.loads(t.CONFIG.read_text())['action_semantics'],t.LABELS);assert before==(t.CONFIG_SHA,t.MANIFEST_SHA,t.EXPECTED_TABLE,t.METHODS,json.loads(t.CONFIG.read_text())['action_semantics'],t.LABELS)
 src=Path(t.__file__).read_text().lower();assert all(x not in src for x in ('def optimize','def calibrat','construct_candidates','schedule fitting'))
def test_14_runner_order_without_execution():
 r=(t.ROOT/'scripts/run_cascadekv_v3_qwen3_1p7b_t1_4k_test.sh').read_text();assert 'uv run python3'in r and r.index('--preflight')<r.index('--capture')<r.index('--evaluate')<r.index('--require-complete')<r.index('--merge')
def test_15_real_status_safe_empty():
 x=t.status();assert(x['valid_caches'],x['valid_shards'],x['result_exists'])==(0,0,False)
def test_16_real_capture_wrapper_synthetic_external_boundaries(monkeypatch,tmp_path):
 import experiments.cascadekv_v3_qwen3_1p7b_vaware_dev as dev
 import experiments.qwen_partial_dot as qpd
 seen=[];monkeypatch.setattr(dev,'resolve',lambda s,tok,m:(seen.append((s,m['sources'][s]))or 'synthetic'))
 class Tok:
  def __call__(self,*a,**k):return types.SimpleNamespace(input_ids=torch.zeros((1,4096),dtype=torch.long))
 class Model:
  layers=[types.SimpleNamespace(self_attn=types.SimpleNamespace(scaling=.125))for _ in range(28)]
  def eval(self):return self
 fake=types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a,**k:Tok()),AutoModel=types.SimpleNamespace(from_pretrained=lambda *a,**k:Model()))
 monkeypatch.setitem(sys.modules,'transformers',fake);monkeypatch.setattr(qpd,'capture_target_layer_post_rope_qkv_low_memory',lambda *a:(doc()['query'],doc()['key'],doc()['value']))
 t.capture('report',7,tmp_path);p=t.cache_path(tmp_path,'report',7);assert t.validate_cache(p,'report',7)[0]
 d=torch.load(p,weights_only=False)['metadata'];assert seen==[('report',t.load_manifest()['sources']['report'])]and d['model_name']==t.MODEL and d['model_revision']==t.REVISION and d['t1_config_sha256']==t.CONFIG_SHA

# These helpers deliberately exercise the production wrappers.  The tensors have
# the required logical geometry but one-element backing stores, so the test never
# creates a holdout artifact or allocates a real 4K capture.
def tiny_doc(s='narrative',l=0):
 d=doc(s,l)
 d['query']=torch.zeros(1,dtype=torch.float16).expand(1,16,4096,128)
 d['key']=torch.zeros(1,dtype=torch.float16).expand(1,8,4096,128)
 d['value']=torch.zeros(1,dtype=torch.float16).expand(1,8,4096,128)
 return d
def tiny_cache(root,s='narrative',l=0):
 p=t.cache_path(root,s,l);t.atomic_torch(tiny_doc(s,l),p);assert t.validate_cache(p,s,l)[0];return p

def install_evaluate_boundaries(monkeypatch):
 """Replace only numerical/search boundaries and retain evaluate's dispatch."""
 import experiments.fresh_sequence_benchmark as fsb
 import experiments.confidence_routing_v2 as cr
 import cascadekv.adaptive_lifting as al
 calls=[]
 class Forest:
  def __init__(self,*a,**k):pass
  def append_atom(self,*a,**k):pass
 def budget(n,f):calls.append(('budget',round(f,3)));return max(1,int(n*f))
 def route_cfg(cfg,l,p):calls.append(('route',round(p,3)));return ({'sink':0,'local':0},0)
 def flat(q,k,b):calls.append(('flat',b));return [0]
 def init(f,q,*a):return {'ids':[0]}
 def advance(r,b):calls.append(('hier',b));return r
 def tr(x,n,b,**kw):return {'total_k_bytes':float(b),'selected_v_fp16_bytes':float(b)}
 monkeypatch.setattr(al,'StreamingLiftingForest',Forest)
 monkeypatch.setattr(fsb,'attention_reference',lambda *a:None)
 monkeypatch.setattr(fsb,'candidate_budget',budget)
 monkeypatch.setattr(fsb,'frozen_routing_config',route_cfg)
 monkeypatch.setattr(fsb,'load_frozen',lambda *a:{})
 monkeypatch.setattr(fsb,'output_metrics',lambda *a:{'relative_exact_attention_mass':1.,'top8_recall':1.,'cosine_similarity':.99,'relative_l2_error':.1,'absolute_l2_error':.1})
 monkeypatch.setattr(fsb,'quantized_flat',flat);monkeypatch.setattr(fsb,'reserve_ids',lambda *a:[]);monkeypatch.setattr(fsb,'traffic',tr)
 monkeypatch.setattr(cr,'initialize_route',init);monkeypatch.setattr(cr,'advance_until_budget',advance)
 return calls

def test_17_real_evaluate_384_rows_and_t1_action_dispatch(monkeypatch,tmp_path):
 tiny_cache(tmp_path);calls=install_evaluate_boundaries(monkeypatch)
 t.evaluate('narrative',0,tmp_path,tmp_path)
 p=t.shard_path(tmp_path,'narrative',0);rows=json.loads(p.read_text())['rows']
 assert t.validate_shard(p,'narrative',0)==(True,'valid') and len(rows)==384
 assert {m:sum(r['method']==m for r in rows) for m in t.METHODS}=={m:48 for m in t.METHODS}
 # T1 layer 0 contains all A0..A4; evaluate's actual calls preserve the
 # frozen candidate/routing semantics, including flat A4.
 assert {x for x in calls if x[0]=='budget'} >= {('budget',.05),('budget',.075),('budget',.1),('budget',.15)}
 assert ('route',.05) in calls and ('flat',max(1,int(2048*.05))) in calls
 assert ('route',.1) in calls # uniform_v1_10 is independently .10/.10

def test_18_real_evaluate_v2_v3_a2_uses_frozen_5pct_routing(monkeypatch,tmp_path):
 # Layer 0 has T1 and V3 A2; layer 14 has V2 A2.  The real wrapper must
 # dispatch A2 candidate .10 with routing .05, never silently uniform-.10.
 for layer in (0,14):
  tiny_cache(tmp_path,'narrative',layer);calls=install_evaluate_boundaries(monkeypatch)
  t.evaluate('narrative',layer,tmp_path,tmp_path)
  assert ('budget',.1) in calls and ('route',.05) in calls

def test_19_evaluate_corrupt_regenerates_and_atomic_failure(monkeypatch,tmp_path):
 tiny_cache(tmp_path);p=t.shard_path(tmp_path,'narrative',0);p.parent.mkdir(exist_ok=True);p.write_text('{bad')
 install_evaluate_boundaries(monkeypatch);t.evaluate('narrative',0,tmp_path,tmp_path);assert t.validate_shard(p,'narrative',0)[0]
 p.write_text('{bad')
 monkeypatch.setattr(t,'atomic_json',lambda *a,**k:(_ for _ in ()).throw(RuntimeError('forced')))
 with pytest.raises(RuntimeError,match='forced'):t.evaluate('narrative',0,tmp_path,tmp_path)
 assert not t.validate_shard(p,'narrative',0)[0]

@pytest.mark.parametrize('source',t.SOURCES)
def test_20_capture_resolves_each_frozen_source_and_regenerates(monkeypatch,tmp_path,source):
 import experiments.cascadekv_v3_qwen3_1p7b_vaware_dev as dev
 import experiments.qwen_partial_dot as qpd
 seen=[];monkeypatch.setattr(dev,'resolve',lambda s,tok,m:(seen.append((s,m['sources'][s])) or 'synthetic'))
 class Tok:
  def __call__(self,*a,**k):return types.SimpleNamespace(input_ids=torch.zeros((1,4096),dtype=torch.long))
 class Model:
  layers=[types.SimpleNamespace(self_attn=types.SimpleNamespace(scaling=.125)) for _ in range(28)]
  def eval(self):return self
 fake=types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a,**k:Tok()),AutoModel=types.SimpleNamespace(from_pretrained=lambda *a,**k:Model()))
 monkeypatch.setitem(sys.modules,'transformers',fake);monkeypatch.setattr(qpd,'capture_target_layer_post_rope_qkv_low_memory',lambda *a:tuple(tiny_doc()['query' if i==0 else 'key' if i==1 else 'value'] for i in range(3)))
 p=t.cache_path(tmp_path,source,0);p.parent.mkdir(exist_ok=True);p.write_text('corrupt')
 t.capture(source,0,tmp_path);assert t.validate_cache(p,source,0)[0] and seen==[(source,t.load_manifest()['sources'][source])]

def test_21_capture_atomic_failure_leaves_no_valid_cache(monkeypatch,tmp_path):
 import experiments.cascadekv_v3_qwen3_1p7b_vaware_dev as dev
 import experiments.qwen_partial_dot as qpd
 monkeypatch.setattr(dev,'resolve',lambda *a:'synthetic')
 class Tok:
  def __call__(self,*a,**k):return types.SimpleNamespace(input_ids=torch.zeros((1,4096),dtype=torch.long))
 class Model:
  layers=[types.SimpleNamespace(self_attn=types.SimpleNamespace(scaling=.125)) for _ in range(28)]
  def eval(self):return self
 monkeypatch.setitem(sys.modules,'transformers',types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a,**k:Tok()),AutoModel=types.SimpleNamespace(from_pretrained=lambda *a,**k:Model())))
 monkeypatch.setattr(qpd,'capture_target_layer_post_rope_qkv_low_memory',lambda *a:(tiny_doc()['query'],tiny_doc()['key'],tiny_doc()['value']))
 monkeypatch.setattr(t,'atomic_torch',lambda *a,**k:(_ for _ in ()).throw(RuntimeError('forced')))
 with pytest.raises(RuntimeError,match='forced'):t.capture('narrative',0,tmp_path)
 assert not t.cache_path(tmp_path,'narrative',0).exists()

def build_complete(root,vals):
 for s in t.SOURCES:
  for l in t.LAYERS:
   tiny_cache(root,s,l);shard(root,s,l,vals)
def merged(root,vals,missing=False):
 build_complete(root,vals)
 if missing:t.shard_path(root,'qa',27).unlink()
 out=root/'result.json';t.merge(root,root,out);return json.loads(out.read_text())
def values(t1=(.99,.1,9),v2=(.99,.11,11),v3=(.99,.11,11),flat=(.99,.2,10),u10=(.99,.2,20),dense=(1,0,30)):
 return {'frozen_qwen3_1p7b_T1':t1,'frozen_cascadekv_v2':v2,'original_frozen_cascadekv_v3_zero_shot':v3,'flat_q8k4_5':flat,'uniform_v1_10':u10,'dense_exact':dense}

@pytest.mark.parametrize('name,vs,label',[
 ('A',values(),t.LABELS['pass']),('B',values(t1=(.984,.1,9)),t.LABELS['absolute_fail']),
 ('C',values(t1=(.99,.121,9)),t.LABELS['absolute_fail']),('D',values(v2=(.99,.1,11)),t.LABELS['comparative_fail']),
 ('E',values(v3=(.99,.1,11)),t.LABELS['comparative_fail']),('F',values(flat=(.99,.2,9)),t.LABELS['comparative_fail']),
 ('G',values(u10=(.99,.2,8)),t.LABELS['pass'])])
def test_22_real_merge_cases_a_through_g(tmp_path,name,vs,label):
 r=merged(tmp_path,vs);assert r['status']=='complete' and r['classification']==label
 assert {m:x['row_count'] for m,x in r['pooled_summaries'].items()}=={m:720 for m in t.METHODS}

def test_23_real_merge_case_h_partial(tmp_path):
 r=merged(tmp_path,values(),missing=True);assert r['status']=='partial' and r['classification']==t.LABELS['invalid']

def test_24_complete_result_schema_and_nonadaptivity(tmp_path):
 a=merged(tmp_path/'a',values());b=merged(tmp_path/'b',values(t1=(.999,.02,2),v2=(.99,.5,11),v3=(.99,.4,11),flat=(.99,.2,10)))
 for r in (a,b):
  assert r['confirmatory_test'] and r['untouched_holdout'] and r['no_fitting_or_schedule_selection_occurred_on_test_data']
  assert r['test_manifest_sha256']==t.MANIFEST_SHA and r['t1_config_sha256']==t.CONFIG_SHA and r['protocol_tag']=={'name':t.TAG,'commit':t.TAG_COMMIT}
  assert r['development_result_sha256']==t.DEVELOPMENT_SHA and r['development_manifest_sha256']==t.DEVELOPMENT_MANIFEST_SHA
  assert (r['frozen_v1_sha256'],r['frozen_v2_sha256'],r['frozen_v3_sha256'])==(t.V1_SHA,t.V2_SHA,t.V3_SHA)
  assert r['target_model']=={'name':t.MODEL,'revision':t.REVISION,'config_sha256':t.MODEL_CONFIG_SHA} and r['methods']==list(t.METHODS)
  assert r['sources']==t.load_manifest()['sources'] and r['geometry']['layers']==list(t.LAYERS) and r['absolute_gates']=={'cosine_gte':.985,'relative_l2_lte':.120}
  assert r['comparative_gates']=={'relative_l2_lt':['frozen_cascadekv_v2','original_frozen_cascadekv_v3_zero_shot'],'total_kv_lt':'flat_q8k4_5'}
  assert set(r['t1_gate_margins'])=={'cosine','relative_l2'} and set(r['t1_deltas'])=={'frozen_cascadekv_v2','original_frozen_cascadekv_v3_zero_shot','uniform_v1_10','flat_q8k4_5','dense_exact'}
 frozen=lambda x:(x['test_manifest_sha256'],x['t1_config_sha256'],t.EXPECTED_TABLE,tuple(x['methods']),x['absolute_gates'],x['comparative_gates'],x['failure_label_rules'])
 assert frozen(a)==frozen(b)
 assert json.loads(t.CONFIG.read_text())['exact_layer_head_action_table']==t.EXPECTED_TABLE and len(t.EXPECTED_TABLE)==40
