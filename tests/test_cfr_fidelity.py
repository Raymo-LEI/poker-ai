from dataclasses import asdict
import json
import pytest
import torch
from alphanl.cfr.config import CFRConfig
from alphanl.cfr.distillation import PolicyNetwork,prepare_archive,save_student,load_student
from alphanl.cfr.fidelity import (ExactTeacher,collect_pool_states,posterior_mixture,
    prune_targets,importance_weights,weighted_kl,fidelity_report,FidelityTrainer,postprocess)
from alphanl.cfr.model import RegretNetwork
from alphanl.transformer.config import ModelConfig
from alphanl.transformer.model import PokerTransformer
from alphanl.transformer.hunl import RULES_VERSION


@pytest.fixture
def teacher(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(17)
    config=CFRConfig(model=ModelConfig(d_model=16,nhead=2,num_layers=1,dim_feedforward=32,max_seq_len=16,dropout=0.))
    for t in (1,2,3):
        model=RegretNetwork(PokerTransformer(**asdict(config.model)))
        with torch.no_grad():
            model.regret_head.weight.zero_()
            model.regret_head.bias.copy_(torch.tensor([t,4-t,1.,2.,3.]))
        torch.save(dict(format='sd_cfr_strategy_v2',iteration=t,weight=t,config=config.to_dict(),
                        rules_version=RULES_VERSION,numeric_version=config.numeric_version,
                        models={'shared':model.state_dict()}),tmp_path/f'iteration_{t:07d}.pt')
    (tmp_path/'index.json').write_text(json.dumps(dict(format='sd_cfr_archive_v1',iterations=[1,2,3])))
    _,manifest=prepare_archive(tmp_path)
    return ExactTeacher(tmp_path,config,manifest),config,manifest


def test_exact_posterior_uses_own_reach_not_local_iteration_average():
    probs=torch.tensor([[[1.,0.]],[[0.,1.]]])
    result=posterior_mixture(probs,torch.tensor([[.1],[.9]]),torch.tensor([1.,3.]))
    torch.testing.assert_close(result,torch.tensor([[1/28,27/28]],dtype=torch.float64))
    assert not torch.allclose(result,torch.tensor([[.25,.75]],dtype=torch.float64))
    with pytest.raises(ValueError,match='unreachable'):
        posterior_mixture(probs,torch.zeros(2,1),torch.tensor([1.,3.]))


def test_collection_reproducible_splits_and_exact_labels_match_manual_reach(teacher):
    ensemble,config,_=teacher
    student=PolicyNetwork(config).eval()
    data,report=collect_pool_states(ensemble,hands=20,seed=9,student=student)
    again,_=collect_pool_states(ensemble,hands=20,seed=9,student=student)
    assert all(torch.equal(value,again[key]) for key,value in data.items())
    labelled=ensemble.label(data,batch_size=7)
    assert labelled['teacher_reachable'].all()
    component=torch.stack([m.strategy(data).detach().double() for m in ensemble.models])
    reach=torch.ones(len(ensemble.models),len(data['parent']),dtype=torch.float64)
    for i,parent in enumerate(data['parent']):
        if parent>=0:reach[:,i]=reach[:,parent]*component[:,parent,data['action'][parent]]
    expected=posterior_mixture(component,reach,ensemble.weights)
    torch.testing.assert_close(labelled['target_probs'].double(),expected,atol=1e-6,rtol=1e-6)
    assert (labelled['target_probs'][data['legal_mask']==0]==0).all()
    validation,_=collect_pool_states(ensemble,hands=4,seed=9,hand_offset=20)
    assert not set(data['hand_id'].tolist())&set(validation['hand_id'].tolist())
    assert sum(report['sources'].values())==20


def test_unreachable_student_paths_are_flagged_without_fabricated_labels():
    class Fixed:
        def strategy(self,batch):return torch.tensor([[1.,0.,0.,0.,0.]]).expand(len(batch['legal_mask']),-1)
    ensemble=ExactTeacher.__new__(ExactTeacher)
    ensemble.models=[Fixed()];ensemble.weights=torch.tensor([1.]);ensemble.device='cpu'
    data={key:torch.zeros(2,1) for key in ('token_ids','type_ids','numeric','pad_mask')}
    data.update(parent=torch.tensor([-1,0]),action=torch.tensor([1,1]),hand_id=torch.tensor([0,0]),legal_mask=torch.ones(2,5))
    labelled=ensemble.label(data)
    assert labelled['teacher_reachable'].tolist()==[True,False]
    assert labelled['target_probs'][1].sum()==0
    data['hand_id'][1]=1
    with pytest.raises(ValueError,match='cross hands'):ensemble.label(data)


def test_weighted_kl_pruning_cap_and_illegal_gradients():
    weights=importance_weights(torch.tensor([0,1,2,3,2]),torch.tensor([2.,21.,4.,50.,21.]))
    assert weights.tolist()==[1.,2.,2.,4.,4.]
    target=torch.tensor([[.01,.49,.50,0.,0.],[.04,.46,.50,0.,0.]])
    mask=torch.tensor([[1.,1.,1.,0.,0.]]).expand(2,-1)
    pruned=prune_targets(target,mask,.03)
    assert pruned[0,0]==0 and pruned[1,0]>0
    torch.testing.assert_close(pruned.sum(-1),torch.ones(2))
    logits=torch.tensor([[.1,.2,.3,100.,100.],[.7,.2,-.3,100.,100.]],requires_grad=True)
    loss=weighted_kl(logits,pruned,mask,torch.tensor([1.,4.]))
    expected=torch.distributions.kl_divergence(torch.distributions.Categorical(probs=pruned[:,:3]),
                                              torch.distributions.Categorical(logits=logits[:,:3]))
    torch.testing.assert_close(loss,(expected*torch.tensor([1.,4.])).sum()/5)
    loss.backward();assert logits.grad[:,3:].abs().sum()==0
    extreme=postprocess(torch.tensor([[.02,.04,.94,0.,0.]]),temperature=.9,cutoff=.99)
    assert extreme.tolist()==[[0.,0.,1.,0.,0.]]
    original=torch.tensor([[.2,.8,0.,0.,0.]])
    assert postprocess(original) is original


def test_fit_exact_labels_improves_and_checkpoint_stays_stage2_compatible(teacher,tmp_path):
    ensemble,config,manifest=teacher
    data,_=collect_pool_states(ensemble,hands=20,seed=4)
    data=ensemble.label(data)
    model=PolicyNetwork(config).eval()
    before=fidelity_report(model,data)['slices']['all']['teacher_to_student_kl']
    trainer=FidelityTrainer(model,data,learning_rate=.003,target_prune=0.)
    trainer.fit(60,batch_size=32)
    after=fidelity_report(model,data)
    assert after['slices']['all']['teacher_to_student_kl']<before
    path=tmp_path/'student.pt';save_student(path,model,manifest,after)
    restored,_=load_student(path)
    from alphanl.stage2.policy import ResidualExploiterPolicy
    residual=ResidualExploiterPolicy(restored).eval()
    batch={**{key:value[:2] for key,value in data.items()},'match_ctx':torch.zeros(2,12)}
    with torch.no_grad():torch.testing.assert_close(residual(batch)[0].softmax(-1),restored.strategy(batch))


def test_acceptance_rejects_small_budgets_missing_bots_and_runtime_cutoffs():
    import copy,importlib.util,sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'scripts'))
    spec=importlib.util.spec_from_file_location('fidelity_acceptance',root/'scripts/train_distillation_fidelity.py')
    script=importlib.util.module_from_spec(spec);spec.loader.exec_module(script)
    scores=dict(random=5000.,call=2200.,maniac=5000.,overbet=6000.,nit=100.,equity=-3000.)
    result=dict(complete=True,inference=dict(temperature=1.,cutoff=0.),
                student_uniform_mean_mbb_per_hand=sum(scores.values())/6,
                opponents={name:dict(student=dict(pairs=2500,hands=5000,mbb_per_hand=value),
                                     identical_deals_verified=True) for name,value in scores.items()})
    assert script.acceptance_gate(result)
    for mutate in (lambda r:r['opponents']['call']['student'].update(pairs=500),
                   lambda r:r['opponents'].pop('nit'),
                   lambda r:r['inference'].update(cutoff=.05),
                   lambda r:r.update(student_uniform_mean_mbb_per_hand=9999.),
                   lambda r:r['opponents']['equity']['student'].update(mbb_per_hand=float('nan'))):
        bad=copy.deepcopy(result);mutate(bad);assert not script.acceptance_gate(bad)
    for temperature in (.85,.9):
        model=PolicyNetwork(CFRConfig(model=ModelConfig(d_model=16,nhead=2,num_layers=1,dim_feedforward=32,max_seq_len=16)))
        features=torch.randn(3,16)
        baked=script.bake_temperature(model,temperature)
        with torch.no_grad():
            torch.testing.assert_close(baked.backbone.policy_head(features).softmax(-1),
                postprocess(model.backbone.policy_head(features).softmax(-1),temperature=temperature),atol=1e-6,rtol=1e-6)
