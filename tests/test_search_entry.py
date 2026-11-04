"""Actual generated runtime/respond path; no search bypass when lock is enabled."""
import importlib.util
import json
import sys
import numpy as np
import pytest
import torch
from alphanl.cfr.config import CFRConfig
from alphanl.cfr.distillation import PolicyNetwork
from alphanl.stage2.policy import ResidualExploiterPolicy
from alphanl.stage2.train import save_checkpoint
from alphanl.stage2.export import export_agent
from alphanl.transformer.config import ModelConfig


@pytest.fixture
def live(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(4)
    config=CFRConfig(model=ModelConfig(d_model=16,nhead=2,num_layers=1,dim_feedforward=32,max_seq_len=32,dropout=0.))
    model=ResidualExploiterPolicy(PolicyNetwork(config)).eval()
    checkpoint=tmp_path/'tiny.pt';save_checkpoint(checkpoint,model)
    output=tmp_path/'agent.py';export_agent(checkpoint,output)
    spec=importlib.util.spec_from_file_location('acceptance_agent',output)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    agent=module.Agent()
    agent.settings['search_v1']={'iterations':8,'sample_hands':3,'runouts':2}
    agent.deployment_guard.update(enabled=True,victory_blind_lock_enabled=True,prune_below=0.)
    agent.force_argmax=True
    return module,agent,sys.modules['_stage2_export.runtime'],sys.modules['_stage2_export.wizard']


def request(river=False):
    history=[dict(round=s,player_id=p,action=0) for s in range(3) for p in (1,0)] if river else []
    return dict(num_players=2,dealer_id=0,my_id=1,my_chips=19900 if river else 19950,
                my_cards=[51,23],public_cards=[0,5,10,31,44] if river else [],history=history,
                hand=0,max_hand=50,total_win_chips=[0,0],total_win_games=[0,0])


def test_respond_and_direct_act_call_one_search_one_guard(live,monkeypatch):
    mod,agent,runtime,wizard=live
    calls=[];guards=[]
    real=wizard.fuse;guard=runtime.apply_guard
    def search(*a,**kw):
        calls.append(True);return real(*a,**kw)
    def guarded(*a,**kw):
        guards.append(True);return guard(*a,**kw)
    monkeypatch.setattr(wizard,'fuse',search);monkeypatch.setattr(runtime,'apply_guard',guarded)
    req=request(river=True);payload={'requests':[req],'responses':[]}
    result=mod.respond(agent,payload,safety=False)
    assert len(calls)==len(guards)==1
    assert result['data']['search_audit']['status']=='active'
    assert result['data']['search_audit']['value']=='river_full1081'
    snap,mapping=mod.request_snapshot(req)
    ctx,_=runtime.context_from_payload(payload)
    amounts={a:snap.me_stack if r==-2 else min(snap.to_call,snap.me_stack) if a==1 else max(r,0) for a,r in mapping.items()}
    action=agent.act(snap,ctx,amounts=amounts,request=req)
    assert mapping[action]==result['response']
    assert len(calls)==len(guards)==2
    cached=mod.respond(agent,dict(payload,data=result['data']),safety=False)
    assert cached['response']==result['response'] and len(calls)==2
    assert agent.search_audit['status']=='cached'
    changed=dict(req,total_win_chips=[-10000,10000])
    locked=mod.respond(agent,{'requests':[changed],'responses':[],'data':result['data']},safety=False)
    assert locked['response']==-1
    assert agent.search_audit=={'status':'skipped','reason':'blind_lock_own'}
    assert len(calls)==2


def test_hard_guard_is_final_and_cannot_be_revived(live,monkeypatch):
    mod,agent,runtime,wizard=live
    monkeypatch.setattr(wizard,'fuse',lambda *a,**kw:np.array([0.,0.,0.,0.,1.]))
    seen=[]
    def force_call(*a,**kw):
        seen.append(1);return np.array([0.,1.,0.,0.,0.]),{'forced_call':True}
    monkeypatch.setattr(runtime,'apply_guard',force_call)
    result=mod.respond(agent,{'requests':[request()],'responses':[]},safety=False)
    assert result['response']==0 and seen==[1]


def test_numeric_error_is_explicit_and_strict_mode_raises(live,monkeypatch):
    mod,agent,runtime,wizard=live
    from _stage2_export.search_belief import RangeError
    def failure(*a,**kw):
        raise RangeError('zero posterior')
    monkeypatch.setattr(wizard,'fuse',failure)
    result=mod.respond(agent,{'requests':[request()],'responses':[]},safety=False)
    assert result['data']['search_audit']['status']=='error_blueprint'
    assert 'zero posterior' in agent.last_wizard
    assert result['data']['search_audit']['final_output']!='dcfr_average'
    agent.settings['strict_search']=True
    with pytest.raises(RangeError):
        mod.respond(agent,{'requests':[request()],'responses':[]},safety=False)


def test_release_packaging_binds_names_and_rejects_reuse(tmp_path):
    import zipfile
    from alphanl.stage2.export import pack_botzone_zip
    # Internal temporary packaging fixture, not an allocated upload release.
    out=tmp_path/'V999_SOURCE.zip'
    pack_botzone_zip(out,release_id='V999_INTERNAL_TEST')
    with zipfile.ZipFile(out) as z:
        entry=z.read('botzone_entry.py').decode()
        assert "V999_MODEL.npz" in entry and "V999_INTERNAL_TEST" in entry
        for name in ('search_math','search_game','search_belief','search_values','blind_lock'):
            assert 'stage2_runtime/'+name+'.py' in z.namelist()
    with pytest.raises(FileExistsError):
        pack_botzone_zip(out,release_id='V999_INTERNAL_TEST')
    with pytest.raises(ValueError):
        pack_botzone_zip(tmp_path/'source.zip',release_id='V999_INTERNAL_TEST')


def test_versioned_weights_metadata_and_exclusive_create(live,tmp_path):
    from alphanl.stage2.export import export_botzone_weights
    target=tmp_path/'V999_MODEL.npz'
    export_botzone_weights(tmp_path/'tiny.pt',target,release_id='V999_INTERNAL_TEST')
    with np.load(target,allow_pickle=False) as z:
        meta=json.loads(z['meta_json'].tobytes().decode())
    assert meta['deployment_release']=='V999_INTERNAL_TEST'
    assert meta['search_contract']=='public_tree_dcfr_v1'
    with pytest.raises(FileExistsError):
        export_botzone_weights(tmp_path/'tiny.pt',target,release_id='V999_INTERNAL_TEST')
    with pytest.raises(ValueError):
        export_botzone_weights(tmp_path/'tiny.pt',tmp_path/'model.npz',release_id='V999_INTERNAL_TEST')


def test_sampling_roundoff_cannot_select_zero_or_illegal_tail(live,monkeypatch):
    mod,agent,runtime,wizard=live
    monkeypatch.setattr(agent,'policy',lambda *a,**kw:np.array([0.,.2,.3,.4999999999999998,0.]))
    agent.force_argmax=False
    rng=type('Rng',(),{'random':lambda self:np.nextafter(1.,0.)})()
    assert agent.act(None,None,rng=rng)==3


def test_raw_positive_undercall_preserves_judge_raise_state(live):
    mod,agent,runtime,wizard=live
    from _stage2_export.search_game import PublicState
    # Legal but unusual platform history: a positive undercall is semantically
    # labelled call, yet the referee still updates round_raise from 102 to 150.
    history=[dict(round=0,player_id=p,action=x) for p,x in zip((1,0,1,0,1,0,1),(0,100,101,102,3,100,150))]
    req=dict(request(),my_id=0,my_chips=19598,history=history)
    snap,mapping=mod.request_snapshot(req)
    state=PublicState.from_request(req,snap)
    assert snap.events[-1].action_type=='call'
    assert state.view.round_raise==150
    assert state.view.street_put==[402,354]
    assert state.view.chips==[19598,19646]
    out=mod.respond(agent,{'requests':[req],'responses':[]},safety=False)
    assert out['data']['search_audit']['status']=='active'
