import copy, random
import pytest
from cascadekv import phi35_8k_c6 as c6
from cascadekv import phi35_8k_c7 as c7

def stat(loss,traffic=1.,cosine=.99): return {"robust_relative_l2":loss,"relative_l2":loss,"total_kv_bytes":traffic,"cosine":cosine}
def groups(): return [{a:stat(1.) for a in c7.ACTIONS} for _ in range(160)]
def rows():
 return [{"source":f"{f}:{i}:development","layer":l,"position":p,"head":h,"metrics":{"relative_l2_error":.1,"cosine_similarity":.99},"traffic":{"total_kv_bytes":1.}} for f in c7.c4.FAMILIES for i in c7.DEVELOPMENT[f] for l in c7.c4.LAYERS for p in c7.c4.POSITIONS for h in c7.c4.HEADS]

def test_actions_mask_inventory_and_protocol():
 assert c7.action_definitions()==c6.action_definitions(); assert c7.allowed_actions(0)==c7.ACTIONS
 assert all(c7.allowed_actions(l)==c7.BASE_ACTIONS for l in (8,16,24,31)); assert c7.validate_protocol()
 assert len(c7.validate_observation_inventory(rows(),action="A0"))==8640

@pytest.mark.parametrize("mut",["delete","duplicate","report20","nan"])
def test_inventory_fails_closed(mut):
 value=rows()
 if mut=="delete":value.pop()
 if mut=="duplicate":value[-1]=copy.deepcopy(value[0])
 if mut=="report20":value[0]["source"]="report:20:development"
 if mut=="nan":value[0]["metrics"]["relative_l2_error"]=float("nan")
 with pytest.raises(c7.C7Error):c7.validate_observation_inventory(value,action="A0")

def test_family_aggregation_and_row_order_invariance():
 data={a:rows() for a in c7.ACTIONS}; before=c7.family_cell_statistics(data)
 random.Random(7).shuffle(data["A0"]); after=c7.family_cell_statistics(data)
 assert before==after and before["0:0:A0"]["family_observation_count"]==18 and before["0:0:A0"]["robust_relative_l2"]==.1

def test_equal_family_c6_parity_and_ties():
 data=groups()
 for row in data:
  for action in c7.ACTIONS: row[action]=stat(2,2,.1)
  row["A0"]=stat(1,2,.8);row["A1"]=stat(1,1,.9)
 c7out=c7.optimize_action_cells(data,160000,domains=(c7.BASE_ACTIONS,)*160)
 c6data=[{a:{"relative_l2":x["relative_l2"],"total_kv_bytes":x["total_kv_bytes"],"cosine":x["cosine"]} for a,x in row.items() if a in c6.ACTIONS} for row in data]
 assert c7out==c6.optimize_action_cells(c6data,160000)
 assert c7out[-1][0]=="A1" # cosine after equal robust loss

def test_robust_divergence_mask_layer0_and_integer_cap():
 data=groups();data[0]["A0"]=stat(.9,1,.99);data[0]["A1"]=stat(.2,2,.9)
 assert c7.optimize_action_cells(data,160000,domains=(c7.BASE_ACTIONS,)*160)[-1][0]=="A0"
 assert c7.optimize_action_cells(data,161000,domains=c7.action_domain_by_cell())[-1][0]=="A1"
 data[32]["A10"]=stat(0,0,.99);assert c7.optimize_action_cells(data,160000,domains=c7.action_domain_by_cell())[-1][32]!="A10"
 data[0]["A10"]=stat(0,1000,.99);assert c7.optimize_action_cells(data,160000,domains=c7.action_domain_by_cell())[-1][0]!="A10"

def test_runtime_closure_and_global_gate_contract():
 assert c7.verify_runtime_closure()
 # Robust loss is optimizer-only: this intentionally does not turn .2 global
 # relL2 into a gate pass.
 assert c7.validate_protocol()["quality_gates"]["mean_relative_l2_lte"]==.12
