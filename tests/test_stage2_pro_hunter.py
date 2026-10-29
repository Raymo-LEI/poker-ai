from dataclasses import replace
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

from alphanl.cfr.config import CFRConfig
from alphanl.cfr.distillation import PolicyNetwork
from alphanl.cfr.gauntlet import ModelActor, play_block
from alphanl.cfr.pro_bots import ProScriptActor, JitteredProScriptActor, PRO_DEFAULTS
from alphanl.cfr.evaluation import RuleActor
from alphanl.stage2.blend import ResidualOutputBlend
from alphanl.stage2.context import MatchCtx, CONTEXT_FIELDS, PRO_CONTEXT_FIELDS
from alphanl.stage2.env import BotzoneMatchEnv
from alphanl.stage2.policy import ResidualExploiterPolicy
from alphanl.stage2.pool import MatchOpponentPool, PRO_WEIGHTS, DEFENSE
from alphanl.stage2.train import save_checkpoint, load_checkpoint, warm_start_policy, collect_match, train_match
from alphanl.transformer.config import ModelConfig
from alphanl.transformer.schema import ActionEvent
from alphanl.transformer.hunl import HunlTable

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from eval_pro_slumbot import safe_post


@pytest.fixture(autouse=True)
def rng():
    torch.set_num_threads(1);torch.manual_seed(47)


def policy(fields=CONTEXT_FIELDS):
    config=CFRConfig(equity_features=True,equity_samples=2,model=ModelConfig(
        d_model=16,nhead=2,num_layers=1,dim_feedforward=32,max_seq_len=24,dropout=0.))
    return ResidualExploiterPolicy(PolicyNetwork(config),context_fields=fields).eval()


def event(street,me,kind,owed=0,added=0,legal=(0,1,2,3,4),stack=19000):
    return ActionEvent(street,1 if me else 0,'SB' if me else 'BB',me,None,kind,600,
        added,added/600,0,owed,stack,legal)


def test_context_counts_opportunities_not_checks_or_donks():
    ctx=MatchCtx();ctx.begin_hand()
    ctx.observe_action(event(0,False,'raise',50,250))
    ctx.observe_action(event(0,True,'call',200,200))
    ctx.observe_action(event(1,False,'check'))
    ctx.observe_action(event(1,True,'bet',0,300))
    ctx.observe_action(event(1,False,'raise',300,900))
    assert ctx.trap_opportunities=={'check_raise_freq':1,'cbet_freq':1}
    assert ctx.trap_successes=={'check_raise_freq':1.,'cbet_freq':0.}
    assert ctx.tensor().shape==(1,12) and ctx.tensor(PRO_CONTEXT_FIELDS).shape==(1,14)
    assert MatchCtx.from_state_dict(json.loads(json.dumps(ctx.state_dict()))).state_dict()==ctx.state_dict()
    ctx.finish_hand(10);ctx.begin_hand()
    ctx.observe_action(event(0,False,'raise',50,250))
    ctx.observe_action(event(1,True,'bet',0,300))
    ctx.observe_action(event(1,False,'raise',300,900))
    assert ctx.trap_opportunities=={'check_raise_freq':1,'cbet_freq':1} # A donk response is neither.
    ctx.finish_hand(0);ctx.begin_hand()
    ctx.observe_action(event(0,False,'raise',50,250))
    ctx.observe_action(event(1,True,'check'))
    ctx.observe_action(event(1,False,'bet',0,300))
    assert ctx.trap_successes['cbet_freq']==1 and ctx.trap_opportunities['cbet_freq']==2


def test_no_checkraise_opportunity_when_raise_is_illegal():
    ctx=MatchCtx();ctx.begin_hand()
    ctx.observe_action(event(1,False,'check'))
    ctx.observe_action(event(1,True,'all_in',0,19000))
    ctx.observe_action(event(1,False,'call',19000,19000,legal=(0,1),stack=19000))
    assert ctx.trap_opportunities['check_raise_freq']==0


def test_pro_pool_initial_weights_and_stable_jitter():
    pool=MatchOpponentPool('unused',weights=PRO_WEIGHTS)
    assert len(pool.weights)==11 and pool.probabilities()==pytest.approx(PRO_WEIGHTS)
    assert pool.probabilities()['TAG_Pro']==pytest.approx(.15)
    assert pool.probabilities()['Trapper_CR']==pytest.approx(.15)
    for name in ('TAG_Pro','Trapper_CR'):
        actors=[JitteredProScriptActor(name,seed=s,equity_samples=2) for s in (1,2)]
        params=dict(actors[0].params)
        assert params!=actors[1].params and params!=PRO_DEFAULTS
        play_block(actors[0],RuleActor('call',seed=1),np.arange(50),1)
        assert params==actors[0].params
        assert ProScriptActor(name,seed=1).params==PRO_DEFAULTS


def test_eleven_family_sampling_quotas_and_pool_resume(tmp_path):
    from collections import Counter
    from alphanl.cfr.model import RegretNetwork
    from alphanl.transformer.model import PokerTransformer
    from alphanl.transformer.hunl import RULES_VERSION
    model=policy(PRO_CONTEXT_FIELDS);config=model.config
    directory=tmp_path/'strategies';directory.mkdir()
    network=RegretNetwork(PokerTransformer(**config.to_dict()['model']),fallback=config.fallback)
    for t in (100,110,140,160,240,260,320,350):
        torch.save(dict(format='sd_cfr_strategy_v2',iteration=t,weight=t,rules_version=RULES_VERSION,
            numeric_version=config.numeric_version,config=config.to_dict(),models={'shared':network.state_dict()}),
            directory/f'iteration_{t:07d}.pt')
    pool=MatchOpponentPool(directory,seed=7,weights=PRO_WEIGHTS,equity_samples=2,history_cache_size=8)
    pool.initialize(model);seen=Counter()
    for _ in range(20):
        block=[pool.sample() for _ in range(20)]
        counts=Counter(name for name,_ in block);seen.update(counts)
        assert {n:counts[n] for n in DEFENSE}=={'reference':2,'history':3,'self':2}
        assert all(isinstance(actor,JitteredProScriptActor) for n,actor in block if n in ('TAG_Pro','Trapper_CR'))
    assert set(seen)==set(PRO_WEIGHTS)
    restored=MatchOpponentPool(directory,seed=99,weights=PRO_WEIGHTS,equity_samples=2,history_cache_size=8)
    restored.initialize(model);restored.load_state_dict(pool.state_dict())
    assert [pool.sample()[0] for _ in range(20)]==[restored.sample()[0] for _ in range(20)]


def test_output_blend_and_identical_reference_guard():
    a=policy();b=policy();b.reference.load_state_dict(a.reference.state_dict())
    with torch.no_grad():
        a.delta_head.bias.fill_(.3);b.delta_head.bias.copy_(torch.arange(5)/3)
    table=HunlTable(seed=8);table.reset()
    actor=ModelActor(a,seed=1);actor.begin_hand(1)
    batch=actor.tokenizer.encode(actor.guard.apply(table.snapshot(1)));batch['match_ctx']=actor.ctx.tensor()
    for weights in ((.4,.6),(.5,.5)):
        blend=ResidualOutputBlend([a,b],weights)
        expected=weights[0]*a(batch,mask_illegal=False)[0]+weights[1]*b(batch,mask_illegal=False)[0]
        torch.testing.assert_close(blend(batch,mask_illegal=False)[0],expected,rtol=0,atol=0)
        assert blend(batch)[0].softmax(-1)[batch['legal_mask']==0].sum()==0
    with torch.no_grad():next(b.reference.parameters()).add_(1)
    with pytest.raises(ValueError,match='references differ'):ResidualOutputBlend([a,b],[.4,.6])


def test_warm_migration_exact_behavior_checkpoint_and_ppo(tmp_path):
    old=policy()
    with torch.no_grad():old.delta_head.weight.normal_(0,.1)
    source=tmp_path/'old.pt';save_checkpoint(source,old,update=200)
    new=ResidualExploiterPolicy(old.reference,**old.settings,context_fields=PRO_CONTEXT_FIELDS).eval()
    info=warm_start_policy(new,source)
    assert info['source_update']==200 and not info['optimizer_restored'] and not info['pool_restored']
    assert torch.count_nonzero(new.context_encoder[0].weight[:,12:])==0
    left,hands=play_block(ModelActor(old,seed=7,residual=True),RuleActor('call',seed=1),np.arange(50),1)
    right,other=play_block(ModelActor(new,seed=7,residual=True),RuleActor('call',seed=1),np.arange(50),1)
    assert left==right and hands==other
    path=tmp_path/'new.pt';save_checkpoint(path,new,update=0)
    restored,blob=load_checkpoint(path)
    assert restored.context_fields==PRO_CONTEXT_FIELDS and blob['context_fields']==list(PRO_CONTEXT_FIELDS)
    frozen={k:v.clone() for k,v in new.reference.state_dict().items()}
    env=BotzoneMatchEnv(new.config,JitteredProScriptActor('Trapper_CR',seed=31,equity_samples=2),context_fields=PRO_CONTEXT_FIELDS)
    rollout=collect_match(new,env)
    assert rollout.batch['match_ctx'].shape[1]==14 and env.ctx.hands_played==50
    optimizer=torch.optim.Adam([p for p in new.parameters() if p.requires_grad],lr=.001)
    metrics=train_match(new,optimizer,rollout,epochs=1,minibatch_size=256)
    assert all(np.isfinite(x) for x in metrics.values())
    assert all(torch.equal(v,new.reference.state_dict()[k]) for k,v in frozen.items())
    assert torch.count_nonzero(new.context_encoder[0].weight[:,12:])>0
    from alphanl.stage2.export import export_agent
    import importlib.util
    save_checkpoint(path,new,update=1)
    destination=tmp_path/'portable.py';export_agent(path,destination)
    spec=importlib.util.spec_from_file_location('pro_hunter_portable',destination)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    agent=module.Agent()
    assert len(agent.context_fields)==14
    table=HunlTable(seed=19);table.reset();snapshot=table.snapshot(1)
    ctx=MatchCtx();ctx.begin_hand()
    batch=agent.observation(snapshot,ctx)
    actor=ModelActor(new,seed=1,residual=True)
    tensor_batch=actor.tokenizer.encode(actor.guard.apply(snapshot));tensor_batch['match_ctx']=ctx.tensor(PRO_CONTEXT_FIELDS)
    np.testing.assert_allclose(agent.logits(batch),new(tensor_batch)[0].detach().numpy(),rtol=2e-4,atol=2e-4)


def test_only_presend_dns_is_retried(monkeypatch):
    import requests
    from urllib3.exceptions import NameResolutionError,MaxRetryError
    import eval_pro_slumbot
    monkeypatch.setattr(eval_pro_slumbot.time,'sleep',lambda _:None)
    class Response:
        def raise_for_status(self):pass
        def json(self):return {'ok':True}
    class Session:
        calls=0
        def post(self,*a,**kw):
            self.calls+=1
            if self.calls==1:
                raise requests.ConnectionError(MaxRetryError(None,'/api/act',NameResolutionError('slumbot.com',None,OSError('dns'))))
            return Response()
    session=Session();assert safe_post(session,'act',{})=={'ok':True};assert session.calls==2
    class Ambiguous:
        calls=0
        def post(self,*a,**kw):
            self.calls+=1;raise requests.ReadTimeout('response lost')
    session=Ambiguous()
    with pytest.raises(requests.ReadTimeout):safe_post(session,'act',{})
    assert session.calls==1


def test_fourteen_field_botzone_external_weights_and_shim(tmp_path):
    import os,subprocess
    from alphanl.stage2.export import export_botzone_weights,pack_botzone_zip
    model=policy(PRO_CONTEXT_FIELDS)
    with torch.no_grad():model.delta_head.weight.normal_(0,.1)
    checkpoint=tmp_path/'v3.pt';save_checkpoint(checkpoint,model)
    weights=tmp_path/'v3.npz';export_botzone_weights(checkpoint,weights)
    archive=tmp_path/'v3.zip';pack_botzone_zip(archive)
    code=("import sys,json\nsys.path.insert(0,%r)\n" % str(archive)
        +"import dataclasses_compat\nsys.modules['dataclasses']=dataclasses_compat\n"
        +"import botzone_entry as entry\nfrom stage2_runtime import runtime\n"
        +"agent,_,_=entry.load_agent(runtime)\n"
        +"request=dict(num_players=2,dealer_id=0,my_id=1,my_chips=19950,my_cards=[51,23],public_cards=[],history=[],hand=0,max_hand=50,total_win_chips=[0,0],total_win_games=[0,0])\n"
        +"snap,_=runtime.request_snapshot(request)\nctx,_=runtime.context_from_payload({'requests':[request]})\n"
        +"batch=agent.observation(snap,ctx)\nprint(json.dumps({'batch':{k:v.tolist() for k,v in batch.items()},'logits':agent.logits(batch).tolist()}))\n")
    result=subprocess.run([sys.executable,'-c',code],env=dict(os.environ,POKER_STAGE2_WEIGHTS=str(weights)),
        capture_output=True,text=True,check=True)
    row=json.loads(result.stdout)
    types={'token_ids':torch.long,'type_ids':torch.long,'pad_mask':torch.bool}
    batch={k:torch.tensor(v,dtype=types.get(k,torch.float32)) for k,v in row['batch'].items()}
    assert batch['match_ctx'].shape==(1,14)
    np.testing.assert_allclose(row['logits'],model(batch)[0].detach().numpy(),rtol=2e-4,atol=2e-4)
