"""The search adapter returns a real information-set DCFR average policy."""
from types import SimpleNamespace
import numpy as np
from alphanl.stage2.wizard import fuse
from alphanl.transformer.schema import HandSnapshot
from alphanl.stage2.context import MatchCtx


class FlatBlueprint:
    weights = {'cfv_river.0.weight': np.zeros((32,200))}
    settings = {'search_v1': {'iterations':8,'sample_hands':3,'runouts':2}}
    def raw_observation(self,snapshot,context):
        return {'legal_mask': np.array([[a in snapshot.legal_action_ids for a in range(5)]])}
    def logits(self,batch):
        return np.where(batch['legal_mask'],0.,-1e9)


def test_preflop_uses_corrected_tree_and_rejects_legacy_values():
    snap=HandSnapshot((),('SA','HA'),(),1,1,0,150.,(19900.,19950.),50.,(0,1,2,3,4),100.,20000.)
    agent=FlatBlueprint()
    out=fuse(agent,snap,MatchCtx(),np.full(5,.2),{0:0,1:50,2:150,3:150,4:19950})
    assert agent.search_audit['status']=='active'
    assert agent.search_audit['legacy_rejected']
    assert agent.search_audit['output']=='dcfr_average'
    assert agent.search_audit['zero_sum_error']<1e-10
    assert out.sum()==1.


def test_river_no_leaf_network_uses_full_public_domain():
    snap=HandSnapshot((),('S5','H6'),('S2','H3','D4','C9','SK'),1,1,3,1000.,(100.,100.),0.,(1,4),100.,20000.)
    agent=FlatBlueprint()
    out=fuse(agent,snap,MatchCtx(),np.array([0,.5,0,0,.5]),{1:0,4:100})
    assert agent.search_audit['value']=='river_full1081'
    assert out[0]==out[2]==out[3]==0
    assert abs(out.sum()-1)<1e-12
