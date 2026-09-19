import hashlib, json, subprocess
from pathlib import Path
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

def test_harness_only_preflight_status_and_empty_state():
    assert h.status()['valid_caches']==0 and h.status()['valid_shards']==0 and not h.status()['result_exists']
    assert not any(hasattr(h,n) for n in ('capture','evaluate','merge','require_complete'))
    runner=(ROOT/'scripts/run_cascadekv_v3_qwen3_1p7b_t1_4k_test_v2.sh').read_text()
    assert '--preflight' in runner and '--status' in runner and all(x not in runner for x in ('--capture','--evaluate','--merge','8K'))

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
