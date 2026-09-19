import copy, hashlib, json, subprocess, sys
from pathlib import Path
import pytest
import torch
from experiments import cascadekv_v3_qwen3_1p7b_t1_4k_test_v2 as h
from experiments import cascadekv_v3_qwen3_1p7b_t1_source as s

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'results/cascadekv_v3_qwen3_1p7b_t1_4k_test_manifest.json'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def test_immutable_incident_old_inputs_and_no_partial_reuse():
    incident=json.loads(h.INCIDENT.read_text())
    assert incident['incident_classification']=='CONFIRMATORY-PROTOCOL-INVALID-AFTER-EXECUTION-START'
    assert sha(OLD)=='53af3b7dfe1dc7ccdd994851e4e22d329cf888f02c11d7f76e858735e106890b'
    assert subprocess.check_output(['git','rev-parse','cascadekv-v3-qwen3-1p7b-t1-4k-test-protocol-freeze^{commit}'],text=True).strip()=='a45918680e45026c180ce3696d16009603398183'
    assert subprocess.check_output(['git','rev-parse','cascadekv-v3-qwen3-1p7b-t1-4k-test-execution-freeze^{commit}'],text=True).strip()=='335ca4a6b7fd7d925ff2d5be0fa9d526319edf9c'
    assert sha(h.CONFIG)==h.CONFIG_SHA and incident['final_valid_state']=={'valid_caches':'5/15','valid_shards':'5/15','result':'no result','last_completed_pair':'narrative layer 27','failed_pair':'report layer 0'}
    for group, directory in [('caches',ROOT/'results/cascadekv_v3_qwen3_1p7b_t1_4k_test_cache'),('shards',ROOT/'results/cascadekv_v3_qwen3_1p7b_t1_4k_test_shards')]:
        assert {p.name:sha(p) for p in directory.iterdir()}==incident['valid_partial_artifact_sha256'][group]
    assert 't1_4k_test_cache' not in h.MANIFEST.read_text() and 't1_4k_test_shards' not in h.MANIFEST.read_text()

def test_harness_status_and_empty_state():
    assert h.status()['valid_caches']==0 and h.status()['valid_shards']==0 and not h.status()['result_exists']
    assert all(hasattr(h,n) for n in ('capture','evaluate','merge','require_complete'))
    runner=(ROOT/'scripts/run_cascadekv_v3_qwen3_1p7b_t1_4k_test_v2.sh').read_text()
    assert '--preflight' in runner and '--capture' in runner and '--evaluate' in runner and '--merge' in runner and '8K' not in runner

@pytest.mark.parametrize('name,expected', [('V1_CONFIG',h.V1_CONFIG_SHA),('V2_CONFIG',h.V2_CONFIG_SHA),('V3_CONFIG',h.V3_CONFIG_SHA)])
def test_routing_configs_are_expected_hash_bound(monkeypatch,tmp_path,name,expected):
    """Every config read by evaluate is a prospective immutable input."""
    assert h.sha(getattr(h,name))==expected
    bad=tmp_path/(name+'.json'); bad.write_text('{}')
    monkeypatch.setattr(h,name,bad)
    with pytest.raises(RuntimeError,match='immutable hash mismatch'):
        h.preflight()

def test_manifest_preflight_retains_both_source_proofs(monkeypatch):
    original=h.require_inputs
    m=h.load_manifest()
    changed=copy.deepcopy(m)
    changed['sources']['narrative']['selection_proof']['character_count'] += 1
    monkeypatch.setattr(h,'require_inputs',lambda: ({},changed))
    with pytest.raises(RuntimeError,match='source binding'):
        h.load_manifest()
    monkeypatch.setattr(h,'require_inputs',original)

@pytest.mark.parametrize('family,layer,action', [
    (family,layer,action)
    for family in h.SOURCES for layer in h.LAYERS
    for action in ('A0','A1')
])
def test_v2_frozen_mechanics_matrix(family,layer,action):
    """Thirty production-independent cases cover the fixed V2 identity matrix."""
    m=h.load_manifest(); entry=m['sources'][family]
    assert entry['dataset_index'] in ({'narrative':12,'report':13,'qa':12}[family],)
    assert h._meta(family,layer)['source_identity']==entry
    budget,route,mode=h.action_profile(json.loads(h.CONFIG.read_text())['action_semantics'],action)
    assert (budget,route,mode)==({ 'A0':(.05,.05,'hierarchy'),'A1':(.075,.05,'hierarchy')}[action])

@pytest.mark.parametrize('family', h.SOURCES)
def test_capture_wrapper_rejects_mismatched_canonical_proof(monkeypatch,tmp_path,family):
    seen=[]; expected=h.load_manifest()['sources'][family]['production_resolution_reproof']
    def bad(entry):
        seen.append(entry)
        proof=dict(expected);proof['character_count']+=1
        return 'synthetic',proof
    monkeypatch.setattr(h,'PRODUCTION_SOURCE_RESOLVER',bad)
    with pytest.raises(RuntimeError,match='canonical source reproof mismatch'):
        h.capture(family,0,tmp_path)
    assert seen==[h.load_manifest()['sources'][family]]
    assert not h.cache_path(tmp_path,family,0).exists()

def test_fixed_profiles_preserve_a2_and_uniform10_semantics():
    sem=json.loads(h.CONFIG.read_text())['action_semantics']
    assert h.action_profile(sem,'A2')==(.10,.05,'hierarchy')
    assert h.action_profile(sem,'A3')==(.15,.05,'hierarchy')
    assert h.action_profile(sem,'A4')==(.05,None,'flat')
    assert (.10,.10)==(.10,.10)  # uniform baseline is not an action-table rewrite

def test_manifest_freezes_exact_science_and_identity_inventory():
    m=h.load_manifest(); assert tuple(m['methods'])==h.METHODS
    assert m['pass_gates']['absolute']=={'cosine_gte':.985,'relative_l2_lte':.120}
    assert m['pass_gates']['comparative']['t1_relative_l2_lt']==['frozen_cascadekv_v2','original_frozen_cascadekv_v3_zero_shot']
    assert 'uniform_v1_10' not in json.dumps(m['pass_gates'])
    assert set(m['failure_labels'].values())==set(h.LABELS.values())
    assert {k:x['dataset_index'] for k,x in m['sources'].items()}=={'narrative':12,'report':13,'qa':12}
    for family,x in m['sources'].items():
        assert x['identity_kind']=='dataset_index' and 'id' not in x
        assert x['dataset_index'] not in m['unavailable_inventory'][family]['indices']
    assert 11 in m['unavailable_inventory']['narrative']['indices'] and 12 in m['unavailable_inventory']['report']['indices'] and 11 in m['unavailable_inventory']['qa']['indices']
    assert m['unavailable_inventory']['narrative']['reasons']['11']=='invalid first confirmatory protocol: real confirmatory method evaluation occurred'
    assert m['unavailable_inventory']['report']['reasons']['12'].startswith('invalid first confirmatory protocol execution-start reservation and mechanical incident inspection')
    assert m['unavailable_inventory']['qa']['reasons']['11'].startswith('invalid first confirmatory protocol execution-start reservation')

def test_selector_and_production_are_same_predicate_and_never_use_row_id():
    class Tok:
        def __call__(self,text,**kw): return type('R',(),{'input_ids':[0]*min(len(text),4096)})()
    rows=[{'id':'misleading','text':'x'*10},{'id':'other','text':'x'*5000}]
    spec=s.SPECS['narrative']; entry={'family':'narrative',**spec,'identity_kind':'dataset_index','dataset_index':1}
    text, proof=s.resolve_frozen_source(entry,tokenizer=Tok(),dataset_loader=lambda _:rows)
    assert text==rows[1]['text'] and proof['dataset_index']==1 and proof['at_least_4096']
    selected, selected_proof, skips=s.select_first_unused('narrative',set(),start=0,tokenizer=Tok(),dataset_loader=lambda _:rows)
    assert selected['dataset_index']==1 and skips[0]['dataset_index']==0 and selected_proof==proof
    src=Path(s.__file__).read_text(); assert 'AutoModel' not in src and 'import torch' not in src and 'capture_target_layer' not in src and 'attention_reference' not in src
    assert h.PRODUCTION_SOURCE_RESOLVER is s.resolve_frozen_source

def test_old_preparation_proof_was_serialized_without_a_predicate_call():
    historical=subprocess.check_output(['git','show','a45918680e45026c180ce3696d16009603398183:experiments/cascadekv_v3_qwen3_1p7b_t1_4k_test.py'],text=True)
    frozen_manifest=subprocess.check_output(['git','show','a45918680e45026c180ce3696d16009603398183:results/cascadekv_v3_qwen3_1p7b_t1_4k_test_manifest.json'],text=True)
    assert 'Selection is deliberately not implemented here' in historical
    assert 'token_length_at_least' not in historical and 'AutoTokenizer' not in historical
    assert '"bounded_frozen_tokenizer_length":4096,"at_least_4096":true' in frozen_manifest
    incident=json.loads(h.INCIDENT.read_text())
    assert incident['preparation_bug'].startswith('Synthetic/unverified proof serialization')

def test_report12_regression_and_v2_proofs_are_actual_pinned_mechanical_proofs():
    old={'family':'report',**s.SPECS['report'],'identity_kind':'dataset_index','dataset_index':12}
    _, proof=s.mechanical_proof(old)
    assert (proof['at_least_4096'],proof['bounded_frozen_tokenizer_length'])==(False,3722)
    m=h.load_manifest()
    assert 12 in m['unavailable_inventory']['report']['indices']
    chosen, chosen_proof, skips=s.select_first_unused('report',set(m['unavailable_inventory']['report']['indices']),start=13)
    assert chosen['dataset_index']==13 and chosen_proof['at_least_4096'] and skips==[]
    for family, entry in m['sources'].items():
        _, again=s.resolve_frozen_source(entry)
        expected=entry['selection_proof']; reproof=entry['production_resolution_reproof']
        for key in h.PROOF_KEYS:
            assert expected[key]==reproof[key]==again[key]
        assert again['at_least_4096'] and again['bounded_frozen_tokenizer_length']==4096

def test_actual_selection_is_ascending_and_matches_all_frozen_v2_sources():
    m=h.load_manifest()
    starts={'narrative':12,'report':13,'qa':12}
    for family, start in starts.items():
        selected, proof, skips=s.select_first_unused(family,set(m['unavailable_inventory'][family]['indices']),start=start)
        assert selected=={key:m['sources'][family][key] for key in ('family','dataset','config','split','revision','field','identity_kind','dataset_index')}
        assert proof==m['sources'][family]['selection_proof']
        assert skips==m['newly_skipped_identities'][family]

def _cache(root, family='narrative', layer=0):
    """A validator-real cache with tiny shared backing storage."""
    z=torch.zeros((1,1,1,1),dtype=torch.float16)
    data={'metadata':{**h._meta(family,layer),'canonical_source_proof':h.load_manifest()['sources'][family]['production_resolution_reproof'],'tensor_shapes':{'q':[1,16,4096,128],'k':[1,8,4096,128],'v':[1,8,4096,128]},'attention_scaling':1.0},'query':z.expand(1,16,4096,128),'key':z.expand(1,8,4096,128),'value':z.expand(1,8,4096,128)}
    h.atomic_torch(data,h.cache_path(root,family,layer)); return data

def _rows(family,layer, quality=None):
    quality=quality or {}
    out=[]
    for pos in h.POSITIONS:
      for qh in range(16):
       for method in h.METHODS:
        metric={x:1.0 for x in h.METRICS}; metric['relative_l2_error']=quality.get(method,.01); metric['absolute_l2_error']=.01
        traffic={x:1.0 for x in h.TRAFFIC_FIELDS}; traffic['total_kv_bytes']=quality.get(method+'-bytes',10.0)
        out.append({'source':family,'layer':layer,'context_length':4096,'position':pos,'q_head':qh,'kv_head':qh//2,'method':method,'metrics':metric,'traffic':traffic})
    return out

def _shard(root,family='narrative',layer=0,quality=None):
    h.atomic_json({**h._meta(family,layer),'rows':_rows(family,layer,quality)},h.shard_path(root,family,layer))

def test_real_evaluate_causal_snapshots_dispatch_and_resume(monkeypatch,tmp_path):
    _cache(tmp_path)
    import experiments.fresh_sequence_benchmark as bench
    import experiments.confidence_routing_v2 as route
    import cascadekv.adaptive_lifting as lifting
    seen=[]; entered=[]
    class Forest:
      def __init__(self,**kw): self.atoms=0
      def append_atom(self,x): self.atoms+=1
    monkeypatch.setattr(lifting,'StreamingLiftingForest',Forest)
    monkeypatch.setattr(bench,'attention_reference',lambda *x:0)
    monkeypatch.setattr(bench,'candidate_budget',lambda n,f:max(1,int(n*f)))
    monkeypatch.setattr(bench,'quantized_flat',lambda q,k,b:range(b))
    monkeypatch.setattr(bench,'traffic',lambda *x,**kw:{'total_k_bytes':1.,'selected_v_fp16_bytes':1.})
    monkeypatch.setattr(bench,'load_frozen',lambda x:{})
    monkeypatch.setattr(bench,'frozen_routing_config',lambda *x:({'sink':0,'local':0},0))
    monkeypatch.setattr(bench,'reserve_ids',lambda *x:set())
    monkeypatch.setattr(bench,'output_metrics',lambda *x:{k:1. for k in h.METRICS})
    def initial(f,*x): seen.append(f.atoms); return f
    monkeypatch.setattr(route,'initialize_route',initial)
    monkeypatch.setattr(route,'advance_until_budget',lambda f,b:{'ids':range(b)})
    h.evaluate('narrative',0,tmp_path,tmp_path)
    p=h.shard_path(tmp_path,'narrative',0)
    assert h.validate_shard(p,'narrative',0)==(True,'valid')
    rows=json.loads(p.read_text())['rows']; assert len(rows)==384
    assert {r['method'] for r in rows}==set(h.METHODS)
    assert set(seen)=={256,384,512} and all(seen.count(n)==16*5 for n in (256,384,512))
    # A valid shard resumes before any numerical/search kernel is entered.
    monkeypatch.setattr(bench,'attention_reference',lambda *x:(_ for _ in ()).throw(AssertionError('entered')))
    h.evaluate('narrative',0,tmp_path,tmp_path)

def test_real_evaluate_fail_closed_and_atomic(monkeypatch,tmp_path):
    with pytest.raises(RuntimeError,match='invalid cache'): h.evaluate('narrative',0,tmp_path,tmp_path)
    _cache(tmp_path)
    import experiments.fresh_sequence_benchmark as bench
    monkeypatch.setattr(bench,'attention_reference',lambda *x:(_ for _ in ()).throw(RuntimeError('boom')))
    with pytest.raises(RuntimeError,match='boom'): h.evaluate('narrative',0,tmp_path,tmp_path)
    assert not h.shard_path(tmp_path,'narrative',0).exists()

def test_validator_fail_closed_matrix(tmp_path):
    data=_cache(tmp_path); p=h.cache_path(tmp_path,'narrative',0)
    assert h.validate_cache(p,'narrative',0)[0]
    for key,value in [('manifest_sha256','bad'),('t1_config_sha256','bad'),('incident_sha256','bad'),('protocol_tag','bad'),('protocol_commit','bad'),('model_name','bad'),('model_revision','bad'),('target_config_sha256','bad'),('layer',7),('context_length',8),('attention_scaling',float('nan'))]:
      bad=torch.load(p,weights_only=False); bad['metadata'][key]=value; h.atomic_torch(bad,p); assert not h.validate_cache(p,'narrative',0)[0]
    _cache(tmp_path); _shard(tmp_path); sp=h.shard_path(tmp_path,'narrative',0); assert h.validate_shard(sp,'narrative',0)[0]
    bad=json.loads(sp.read_text()); bad['rows'].pop(); h.atomic_json(bad,sp); assert not h.validate_shard(sp,'narrative',0)[0]
    _shard(tmp_path); bad=json.loads(sp.read_text()); bad['rows'][0]['metrics']['cosine_similarity']=float('nan'); h.atomic_json(bad,sp); assert not h.validate_shard(sp,'narrative',0)[0]

def test_real_merge_complete_schema_and_v1_isolation(tmp_path):
    cache=tmp_path/'c'; shards=tmp_path/'s'; out=tmp_path/'result.json'
    for family in h.SOURCES:
      for layer in h.LAYERS: _cache(cache,family,layer); _shard(shards,family,layer)
    h.merge(cache,shards,out); result=json.loads(out.read_text())
    assert result['status']=='complete' and all(x['row_count']==720 for x in result['pooled_summaries'].values())
    for key in ('experiment_identifier','confirmatory_test','untouched_holdout','replacement_protocol','predecessor_incident_path','geometry','t1_gate_margins','t1_deltas','predecessor_partial_quality_metrics_were_not_inputs'): assert key in result
    # Files named like V1 artifacts in separate locations cannot participate in V2 status/merge.
    assert h.status(cache,shards)['valid_caches']==15 and h.status(cache,shards)['valid_shards']==15
    assert h.RESULT.exists() is False

def _install_capture_boundaries(monkeypatch, family, calls, fail_capture=None):
    """Install only the resolver, HF loader, and Q/K/V capture boundaries."""
    import transformers
    import experiments.qwen_partial_dot as qpd
    entry=h.load_manifest()['sources'][family]
    proof=entry['production_resolution_reproof']
    def resolve(got):
        calls.append(('resolver',got)); return 'synthetic frozen source', proof
    class Tokenizer:
        def __call__(self, text, **kwargs):
            calls.append(('tokenize',text,kwargs)); return type('Tokens',(),{'input_ids':torch.zeros((1,4096),dtype=torch.long)})()
    class Model:
        layers=[type('Layer',(),{'self_attn':type('Attn',(),{'scaling':.125})()})() for _ in range(28)]
        def eval(self): return self
    monkeypatch.setattr(h,'PRODUCTION_SOURCE_RESOLVER',resolve)
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**kw:(calls.append(('tokenizer_load',a,kw)) or Tokenizer()))
    monkeypatch.setattr(transformers.AutoModel,'from_pretrained',lambda *a,**kw:(calls.append(('model_load',a,kw)) or Model()))
    def capture(model, ids, layer):
        calls.append(('capture',layer))
        if fail_capture: raise fail_capture
        z=torch.zeros((1,1,1,1),dtype=torch.float16)
        return z.expand(1,16,4096,128),z.expand(1,8,4096,128),z.expand(1,8,4096,128)
    monkeypatch.setattr(qpd,'capture_target_layer_post_rope_qkv_low_memory',capture)

@pytest.mark.parametrize('family',h.SOURCES)
def test_real_capture_success_uses_only_production_boundaries(monkeypatch,tmp_path,family):
    calls=[]; _install_capture_boundaries(monkeypatch,family,calls)
    h.capture(family,0,tmp_path)
    path=h.cache_path(tmp_path,family,0)
    assert h.validate_cache(path,family,0)==(True,'attention scaling')
    assert calls[0]==('resolver',h.load_manifest()['sources'][family])
    for name,args,kw in (x for x in calls if x[0] in ('tokenizer_load','model_load')):
        assert args==(h.MODEL,) and kw['revision']==h.REVISION
    data=torch.load(path,weights_only=False)
    assert tuple(data['query'].shape)==(1,16,4096,128) and data['query'].dtype==torch.float16
    assert tuple(data['key'].shape)==(1,8,4096,128) and data['value'].dtype==torch.float16

def test_real_capture_resume_corrupt_regeneration_and_atomicity(monkeypatch,tmp_path):
    _cache(tmp_path)
    for name in ('PRODUCTION_SOURCE_RESOLVER','atomic_torch'):
        monkeypatch.setattr(h,name,lambda *a,**kw:(_ for _ in ()).throw(AssertionError('resume entered '+name)))
    h.capture('narrative',0,tmp_path)
    monkeypatch.undo()
    p=h.cache_path(tmp_path,'narrative',0); p.write_bytes(b'corrupt')
    calls=[]; _install_capture_boundaries(monkeypatch,'narrative',calls); h.capture('narrative',0,tmp_path)
    assert h.validate_cache(p,'narrative',0)[0]
    p.unlink(); calls=[]; _install_capture_boundaries(monkeypatch,'narrative',calls,RuntimeError('capture boom'))
    with pytest.raises(RuntimeError,match='capture boom'): h.capture('narrative',0,tmp_path)
    assert not p.exists()
    _install_capture_boundaries(monkeypatch,'narrative',calls)
    monkeypatch.setattr(h,'atomic_torch',lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('atomic boom')))
    with pytest.raises(RuntimeError,match='atomic boom'): h.capture('narrative',0,tmp_path)
    assert not p.exists()

def test_complete_cache_validator_matrix(tmp_path):
    p=h.cache_path(tmp_path,'narrative',0)
    mutations=[
      ('metadata.manifest_sha256','bad'),('metadata.t1_config_sha256','bad'),('metadata.incident_sha256','bad'),
      ('metadata.protocol_tag','bad'),('metadata.protocol_commit','bad'),('metadata.model_name','bad'),
      ('metadata.model_revision','bad'),('metadata.target_config_sha256','bad'),('metadata.source_identity',{}),
      ('metadata.canonical_source_proof',{}),('metadata.layer',7),('metadata.context_length',8),
      ('metadata.tensor_shapes',{}),('query',torch.zeros((1,16,1,128),dtype=torch.float16)),
      ('key',torch.zeros((1,8,1,128),dtype=torch.float16)),('value',torch.zeros((1,8,1,128),dtype=torch.float16)),
      # Retain production logical geometry without allocating giant negative
      # fixtures: cache validation is shape/dtype based, not contiguity based.
      ('query',torch.empty_strided((1,16,4096,128),(0,0,0,0),dtype=torch.float32)),
      ('key',torch.empty_strided((1,8,4096,128),(0,0,0,0),dtype=torch.float32)),
      ('value',torch.empty_strided((1,8,4096,128),(0,0,0,0),dtype=torch.float32)),('metadata.attention_scaling',None),
      ('metadata.attention_scaling',float('nan')),('metadata.attention_scaling',float('inf')),('metadata.attention_scaling',float('-inf'))]
    for dotted,value in mutations:
        _cache(tmp_path); data=torch.load(p,weights_only=False); target=data
        bits=dotted.split('.')
        for bit in bits[:-1]: target=target[bit]
        target[bits[-1]]=value; h.atomic_torch(data,p)
        assert not h.validate_cache(p,'narrative',0)[0], dotted

def test_complete_shard_validator_matrix(tmp_path):
    p=h.shard_path(tmp_path,'narrative',0)
    def altered(kind):
        _shard(tmp_path); d=json.loads(p.read_text()); r=d['rows'][0]
        if kind.startswith('top.'): d[kind[4:]]='bad'
        elif kind=='source': r['source']='report'
        elif kind=='missing': d['rows'].pop()
        elif kind=='duplicate': d['rows'][-1]=copy.deepcopy(r)
        elif kind=='method': r['method']='unknown'
        elif kind=='position': r['position']=0
        elif kind=='q_head': r['q_head']=16
        elif kind=='kv_head': r['kv_head']=7
        elif kind=='missing_metric': r['metrics'].pop(next(iter(r['metrics'])))
        elif kind=='extra_metric': r['metrics']['extra']=1.
        elif kind=='missing_traffic': r['traffic'].pop(next(iter(r['traffic'])))
        elif kind=='extra_traffic': r['traffic']['extra']=1.
        else:
            group, value=kind.split(':'); r[group][next(iter(r[group]))]=float(value)
        h.atomic_json(d,p)
    cases=['top.manifest_sha256','top.t1_config_sha256','top.incident_sha256','top.protocol_tag','top.protocol_commit','top.source_identity','source','top.layer','top.context_length','missing','duplicate','method','position','q_head','kv_head','missing_metric','extra_metric','metrics:nan','metrics:inf','metrics:-inf','missing_traffic','extra_traffic','traffic:nan','traffic:inf','traffic:-inf']
    for case in cases:
        altered(case); assert not h.validate_shard(p,'narrative',0)[0],case

def _complete_dataset(cache, shards, *, cosine=.99, t1=.01, v2=.02, v3=.03, flat_bytes=20., t1_bytes=10., uniform_bytes=15.):
    quality={'frozen_qwen3_1p7b_T1':t1,'frozen_cascadekv_v2':v2,'original_frozen_cascadekv_v3_zero_shot':v3,
             'frozen_qwen3_1p7b_T1-bytes':t1_bytes,'flat_q8k4_5-bytes':flat_bytes,'uniform_v1_10-bytes':uniform_bytes}
    for family in h.SOURCES:
      for layer in h.LAYERS:
        _cache(cache,family,layer); _shard(shards,family,layer,quality)
        p=h.shard_path(shards,family,layer); d=json.loads(p.read_text())
        for row in d['rows']:
          if row['method']=='frozen_qwen3_1p7b_T1': row['metrics']['cosine_similarity']=cosine
        h.atomic_json(d,p)

@pytest.mark.parametrize('name,kwargs,label',[
 ('A',{},h.LABELS['pass']),('B',{'cosine':.984},h.LABELS['absolute_fail']),('C',{'t1':.121},h.LABELS['absolute_fail']),
 ('D',{'t1':.02,'v2':.02},h.LABELS['comparative_fail']),('E',{'t1':.03,'v3':.03},h.LABELS['comparative_fail']),
 ('F',{'t1_bytes':20.},h.LABELS['comparative_fail']),('G',{'uniform_bytes':5.},h.LABELS['pass'])])
def test_real_merge_cases_a_through_g(tmp_path,name,kwargs,label):
    cache=tmp_path/'cache'; shards=tmp_path/'shards'; out=tmp_path/'out.json'; _complete_dataset(cache,shards,**kwargs)
    h.merge(cache,shards,out); result=json.loads(out.read_text())
    assert result['status']=='complete' and result['classification']==label

def test_real_merge_case_h_and_exact_absolute_boundaries(tmp_path):
    cache=tmp_path/'cache'; shards=tmp_path/'shards'; out=tmp_path/'out.json'; _complete_dataset(cache,shards)
    h.shard_path(shards,'qa',27).unlink(); h.merge(cache,shards,out)
    result=json.loads(out.read_text()); assert result['status']=='partial' and result['classification']==h.LABELS['invalid']
    _complete_dataset(cache,shards,cosine=.985,t1=.120,v2=.121,v3=.122)
    h.merge(cache,shards,out); assert json.loads(out.read_text())['classification']==h.LABELS['pass']

def test_case_a_result_schema_nonadaptivity_and_v1_isolation(tmp_path):
    snapshots=[]
    for n,kwargs in enumerate(({}, {'cosine':.997,'t1':.005,'v2':.2,'v3':.3,'t1_bytes':3.})):
      cache=tmp_path/f'c{n}'; shards=tmp_path/f's{n}'; out=tmp_path/f'o{n}.json'; _complete_dataset(cache,shards,**kwargs); h.merge(cache,shards,out)
      result=json.loads(out.read_text()); snapshots.append(result)
      assert result['classification']==h.LABELS['pass']
      assert set(result['pooled_summaries'])==set(h.METHODS) and all(x['row_count']==720 for x in result['pooled_summaries'].values())
      assert result['sources']==h.load_manifest()['sources'] and result['methods']==list(h.METHODS)
      assert result['no_fitting_or_schedule_selection_occurred_on_test_data'] is True
      assert result['predecessor_partial_quality_metrics_were_not_inputs'] is True
    stable=('test_manifest_sha256','t1_config_sha256','incident_sha256','frozen_config_sha256','sources','geometry','methods','absolute_gates','comparative_gates','failure_label_rules')
    assert all(snapshots[0][key]==snapshots[1][key] for key in stable)
    config=json.loads(h.CONFIG.read_text()); assert len(config['exact_layer_head_action_table'])==40
    src=Path(h.__file__).read_text(); assert 't1_4k_test_shards' not in src and 'optimizer' not in src and 'calibration' not in src and 'schedule fitting' not in src
    assert h.sha(OLD)=='53af3b7dfe1dc7ccdd994851e4e22d329cf888f02c11d7f76e858735e106890b'

def test_hardened_quantile_convention_exact_vectors():
    def rows(values):
      return [{'metrics':{'cosine_similarity':x,'relative_l2_error':x,'absolute_l2_error':x,'relative_exact_attention_mass':x,'top8_recall':x},'traffic':{'total_k_bytes':x,'selected_v_fp16_bytes':x,'total_kv_bytes':x}} for x in values]
    got=h.summary(rows([0.,1.,2.,3.,4.,5.,6.,7.,8.,9.]))['metrics']
    assert got['cosine_similarity']['p5']==0. and got['relative_l2_error']['p95']==9. and got['relative_l2_error']['worst']==9.

def test_runner_ordering_is_v2_only():
    runner=(ROOT/'scripts/run_cascadekv_v3_qwen3_1p7b_t1_4k_test_v2.sh').read_text()
    assert runner.index('--preflight') < runner.index('for source in narrative report qa') < runner.index('--require-complete') < runner.index('--merge')
    assert 'for layer in 0 7 14 21 27' in runner and runner.index('--capture') < runner.index('--evaluate')
    assert runner.count('uv run python3')==5 and 'v1_4k_test.py' not in runner and 't1_4k_test_shards' not in runner and '8K' not in runner
