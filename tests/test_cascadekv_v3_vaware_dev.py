import hashlib, itertools, json, random
from pathlib import Path
import pytest
from experiments import cascadekv_v3_vaware_dev as v

def test_consumed_indices_and_split_are_rejected(tmp_path, monkeypatch):
    d={'schema_version':2,'model':{'resolved_commit_sha':v.MODEL_SHA},'context_length':4096,
       'prior_explicit_identity_inventory':[], 'frozen_inputs':{'cascadekv_v1':{'sha256':v.V1_SHA}},
       'sources':{k:{'index':3,'split_assignment':k.rsplit('_',1)[1],'dataset':v.SPECS[k.rsplit('_',1)[0]]['dataset']} for k in v.CAL+v.VAL}}
    p=tmp_path/'m.json';p.write_text(json.dumps(d));monkeypatch.setattr(v,'MANIFEST',p);monkeypatch.setattr(v,'immutable',lambda:None)
    monkeypatch.setattr(v,'prior_manifest_identity_inventory',lambda:[])
    monkeypatch.setattr(v,'V3_MANIFEST_SHA',hashlib.sha256(p.read_bytes()).hexdigest())
    assert v.load_manifest()['sources']['narrative_calibration']['index']==3
    d['sources']['qa_validation']['index']=2;p.write_text(json.dumps(d))
    monkeypatch.setattr(v,'V3_MANIFEST_SHA',hashlib.sha256(p.read_bytes()).hexdigest())
    with pytest.raises(ValueError):v.load_manifest()

def test_selection_is_two_per_domain_and_split_is_predeclared():
    assert len(v.CAL)==len(v.VAL)==3
    assert all(x.endswith('calibration') for x in v.CAL)
    assert all(x.endswith('validation') for x in v.VAL)

def _brute(groups,target):
    brute=[]
    for xs in itertools.product(v.ACTIONS,repeat=len(groups)):
        b=sum(groups[i][a]['total_kv_bytes'] for i,a in enumerate(xs));e=sum(groups[i][a]['relative_l2'] for i,a in enumerate(xs));c=sum(groups[i][a]['cosine'] for i,a in enumerate(xs))
        micro=sum(round(groups[i][a]['total_kv_bytes']*1000) for i,a in enumerate(xs))
        if micro<=round(target*len(groups)*1000):brute.append((micro,e,c,xs))
    return None if not brute else min(brute,key=lambda x:(x[1],-x[2],x[0],x[3]))

def test_multiple_choice_optimizer_matches_exhaustive_bruteforce_instances():
    instances=[
      ([[(1,.4,.9),(2,.1,.8),(9,.01,.99),(10,.01,.99),(11,.01,.99)]]*2,2.0),
      ([[(1,.2,.70),(2,.2,.95),(3,.1,.60),(4,.1,.99),(5,.05,.50)]]*3,3.0),
      ([[(1,.3,.9),(3,.2,.8),(4,.1,.7),(6,.1,.99),(7,.05,.6)]]*4,3.5),
    ]
    for choices,target in instances:
        groups=[{a:{'total_kv_bytes':n,'relative_l2':e,'cosine':c} for a,(n,e,c) in zip(v.ACTIONS,choice)} for choice in choices]
        z=v.optimize(groups,target); want=_brute(groups,target)
        assert (z[0],z[1],z[2],z[3])==want

def test_optimizer_matches_bruteforce_many_small_deterministic_instances():
    rng=random.Random(731)
    for group_count in range(1,6):
      for _ in range(15):
        groups=[]
        for _ in range(group_count):
          groups.append({a:{'total_kv_bytes':rng.randint(1,9),
                            'relative_l2':rng.randint(0,12)/10,
                            'cosine':rng.randint(0,10)/10}
                         for a in v.ACTIONS})
        for target in (1,2.5,4,6,9):
          want=_brute(groups,target)
          got=v.optimize(groups,target)
          assert (None if got is None else (got[0],got[1],got[2],got[3]))==want

def test_lower_error_higher_traffic_wins_when_feasible():
    g=[{a:{'total_kv_bytes':n,'relative_l2':e,'cosine':c} for a,n,e,c in
        [('A0',1,.4,.99),('A1',2,.1,.1),('A2',9,.9,.1),('A3',10,.9,.1),('A4',11,.9,.1)]}]
    assert v.optimize(g,2)[3]==('A1',)

def test_equal_error_higher_cosine_survives_pareto_pruning():
    # A1 costs more than A0 at equal error but must survive for its cosine,
    # because the final budget admits it and the stated objective prefers it.
    g=[{a:{'total_kv_bytes':n,'relative_l2':e,'cosine':c} for a,n,e,c in
        [('A0',1,.10,.80),('A1',2,.10,.99),('A2',9,.9,.1),('A3',10,.9,.1),('A4',11,.9,.1)]}]
    assert v.optimize(g,2)[3]==('A1',)

def test_tiebreak_traffic_then_serialized_table():
    g=[{a:{'total_kv_bytes':n,'relative_l2':1,'cosine':1} for a,n in zip(v.ACTIONS,(1,2,3,4,5))}]
    assert v.optimize(g,5)[3]==('A0',)
    g=[{a:{'total_kv_bytes':1,'relative_l2':1,'cosine':1} for a in v.ACTIONS}]
    assert v.optimize(g,1)[3]==('A0',)

def test_traffic_cap_pruning_preserves_bruteforce_optimum():
    groups=[{a:{'total_kv_bytes':n,'relative_l2':e,'cosine':c} for a,n,e,c in
             [('A0',1,.5,.5),('A1',3,.3,.6),('A2',5,.2,.7),('A3',9,.1,.8),('A4',12,.01,.99)]}
            for _ in range(4)]
    want=_brute(groups,2)
    got=v.optimize(groups,2)
    assert (got[0],got[1],got[2],got[3])==want

def test_large_monotonic_workload_has_bounded_exact_frontier():
    # This used to make the quadratic all-pairs dominance loop pathological.
    groups=[{a:{'total_kv_bytes':i+1,'relative_l2':1/(i+1),'cosine':.8+i*.01}
             for i,a in enumerate(v.ACTIONS)} for _ in range(300)]
    diagnostics={}; got=v.optimize(groups,3,diagnostics)
    assert got is not None and got[0]<=round(3*len(groups)*1000)
    assert diagnostics['expanded_state_count'] > 0
    # There are only 1,201 possible integer-byte totals in this fixture;
    # the frontier is linear in those totals rather than quadratically scanned.
    assert diagnostics['max_frontier_size']<=1201

def _schedule_rows(sequence, multiplier=1):
    rows=[]
    for l in v.LAYERS:
      for kv in range(8):
       for method in (*v.ACTIONS,'frozen_cascadekv_v2','uniform_v1_10'):
        action_index=list(v.ACTIONS).index(method) if method in v.ACTIONS else 0
        rows.append({'sequence':sequence,'layer':l,'kv_head':kv,'method':method,
          'metrics':{'relative_l2_error':multiplier*(.1+action_index*.01),'cosine_similarity':.9-action_index*.001,'relative_exact_attention_mass':.8},
          'traffic':{'total_k_bytes':1,'total_kv_bytes':2+action_index if method in v.ACTIONS else (3 if method=='frozen_cascadekv_v2' else 5)}})
    return rows

def test_validation_cannot_change_frozen_calibration_targets_or_tables():
    cal=sum((_schedule_rows(s) for s in v.CAL),[])
    first=v.construct_calibration_schedule(cal)
    validation_v1=sum((_schedule_rows(s,1) for s in v.VAL),[])
    validation_v2=sum((_schedule_rows(s,1000000) for s in v.VAL),[])
    assert v.construct_calibration_schedule(cal, validation_v1)==first
    assert v.construct_calibration_schedule(cal, validation_v2)==first
    assert validation_v1 != validation_v2

def test_validation_cannot_enter_calibration_aggregation():
    rows=[]
    for l in v.LAYERS:
      for kv in range(8):
       for a in v.ACTIONS:
        rows.append({'sequence':'narrative_calibration','layer':l,'kv_head':kv,'method':a,'metrics':{'relative_l2_error':.1,'cosine_similarity':.9,'relative_exact_attention_mass':.8},'traffic':{'total_k_bytes':1,'total_kv_bytes':2}})
    rows.append({'sequence':'narrative_validation','layer':0,'kv_head':0,'method':'A0','metrics':{'relative_l2_error':99,'cosine_similarity':0,'relative_exact_attention_mass':0},'traffic':{'total_k_bytes':99,'total_kv_bytes':99}})
    assert v.aggregate([x for x in rows if x['sequence'] in v.CAL])['0:0:A0']['relative_l2']==.1

def test_oracles_are_never_deployable_and_classification_exact():
    assert 'NONDEPLOYABLE' in Path(v.__file__).read_text()
    x={'relative_l2':.12,'cosine':.985,'kv':1,'v2_relative_l2':.13}
    assert v.classify([x],{'oracle_headroom':False},1)=='V-AWARE-STATIC-PROMISING'
    assert v.classify([],{'oracle_headroom':True},1)=='V-AWARE-STATIC-NOT-GENERALIZING'
    assert v.classify([],{'oracle_headroom':False},1)=='ACTION-MENU-LIMITED'

def test_frozen_hashes_and_consumed_test_not_fit():
    assert hashlib.sha256(v.V1.read_bytes()).hexdigest()==v.V1_SHA
    assert hashlib.sha256(v.V2.read_bytes()).hexdigest()==v.V2_SHA
    assert hashlib.sha256(v.V2_TEST.read_bytes()).hexdigest()==v.V2_TEST_SHA
    assert 'V2_TEST' not in v.aggregate.__code__.co_names

def test_ordinary_merge_never_writes_v3_freeze_files(tmp_path):
    # A partial merge is sufficient to exercise the ordinary merge writer.
    merged=tmp_path/'ordinary-merge.json'
    v.merge(tmp_path/'absent-shards',merged)
    assert merged.exists()
    assert not (tmp_path/'cascadekv_v3.json').exists()
    assert not (tmp_path/'cascadekv_v3_vaware_dev_frozen.json').exists()

def test_explicit_freeze_rejects_wrong_result_sha(tmp_path):
    bad=tmp_path/'result.json'; bad.write_bytes(v.Path('results/cascadekv_v3_vaware_dev.json').read_bytes()+b' ')
    config,frozen=tmp_path/'cascadekv_v3.json',tmp_path/'frozen.json'
    with pytest.raises(RuntimeError,match='result hash'):
        v.freeze_v3(bad,config,frozen)
    assert not config.exists() and not frozen.exists()

def test_explicit_freeze_rejects_wrong_manifest_sha(tmp_path):
    bad=tmp_path/'manifest.json'; bad.write_bytes(v.MANIFEST.read_bytes()+b' ')
    config,frozen=tmp_path/'cascadekv_v3.json',tmp_path/'frozen.json'
    with pytest.raises(RuntimeError,match='manifest'):
        v.freeze_v3(config_path=config,frozen_result_path=frozen,manifest_path=bad)
    assert not config.exists() and not frozen.exists()

def test_explicit_freeze_rejects_modified_t0_table(tmp_path,monkeypatch):
    result=json.loads(v.Path('results/cascadekv_v3_vaware_dev.json').read_text())
    original=result['calibration_frozen_tables']['T0']['table']['0:0']
    result['calibration_frozen_tables']['T0']['table']['0:0']='A1' if original!='A1' else 'A0'
    altered=tmp_path/'altered-result.json'; altered.write_text(json.dumps(result))
    monkeypatch.setattr(v,'V3_RESULT_SHA',hashlib.sha256(altered.read_bytes()).hexdigest())
    config,frozen=tmp_path/'cascadekv_v3.json',tmp_path/'frozen.json'
    with pytest.raises(RuntimeError,match='reconstruction'):
        v.freeze_v3(altered,config,frozen)
    assert not config.exists() and not frozen.exists()

def test_reconstruction_path_uses_calibration_only(tmp_path, monkeypatch):
    seen=[]
    def fake_schedule(rows, validation_rows=None):
        seen.extend(row['sequence'] for row in rows)
        return {},{}, {'T0':{'table':{f'{layer}:{kv}':'A0' for layer in v.LAYERS for kv in range(8)}}}
    monkeypatch.setattr(v,'construct_calibration_schedule',fake_schedule)
    monkeypatch.setattr(v,'validate_shard',lambda *args:(True,None))
    monkeypatch.setattr(v,'shard_path',lambda root,s,l:tmp_path/f'{s}-{l}.json')
    for s in v.CAL:
        for l in v.LAYERS: (tmp_path/f'{s}-{l}.json').write_text(json.dumps({'rows':[{'sequence':s}]}))
    assert len(v.reconstruct_t0_table(tmp_path))==40
    assert set(seen)==set(v.CAL)

def test_explicit_freeze_copies_source_bytes_and_writes_40_actions(tmp_path):
    config,frozen=tmp_path/'cascadekv_v3.json',tmp_path/'frozen.json'
    outcome=v.freeze_v3(config_path=config,frozen_result_path=frozen)
    assert frozen.read_bytes()==v.Path('results/cascadekv_v3_vaware_dev.json').read_bytes()
    assert outcome['frozen_result_sha256']==v.V3_RESULT_SHA
    table=json.loads(config.read_text())['exact_layer_head_action_table']
    assert len(table)==40 and set(table.values())<=set(v.ACTIONS)

def test_explicit_freeze_is_idempotent_and_never_touches_production(tmp_path):
    config,frozen=tmp_path/'cascadekv_v3.json',tmp_path/'frozen.json'
    first=v.freeze_v3(config_path=config,frozen_result_path=frozen)
    first_bytes=config.read_bytes(),frozen.read_bytes()
    second=v.freeze_v3(config_path=config,frozen_result_path=frozen)
    assert first['config_sha256']==second['config_sha256']
    assert first_bytes==(config.read_bytes(),frozen.read_bytes())
