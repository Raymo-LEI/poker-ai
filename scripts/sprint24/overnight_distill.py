"""Expanded residual Policy+V_root distillation and independently gated Astra.

The Transformer stays frozen. A 256-wide residual replaces the 64-wide branch;
the original champion's independent 64-wide preflop branch is never trained.
"""
import copy
from pathlib import Path
import numpy as np
from alphanl.stage2 import cfv
from evening_cd import _batch
from overnight_data import collect_teacher,TREE,paired_gain


def bind_champion(model,champion):
    """Bind the initial champion, even when the hot start is an evening child."""
    import torch
    if model.settings['alpha']!=champion.settings['alpha']:raise ValueError('preflop alpha differs from original champion')
    for k,value in champion.reference.state_dict().items():
        if not torch.equal(value.cpu(),model.reference.state_dict()[k].cpu()):
            raise ValueError('warm-start reference differs from initial champion: '+k)
    model._evening_preflop_anchor={k:v.detach().cpu().clone() for k,v in champion.state_dict().items()
        if k.startswith(('context_encoder.','fusion.','delta_head.'))}
    return model


def expand(model,width=256):
    import torch
    from torch import nn
    old=model.settings['hidden']
    if old>=width:return model
    device=next(model.parameters()).device
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(90211)
        ctx=nn.Linear(len(model.context_fields),width).to(device)
        fusion=nn.Linear(model.config.model.d_model+width,width).to(device)
        delta=nn.Linear(width,5).to(device);value=nn.Linear(width,1).to(device)
    with torch.no_grad():
        ctx.weight[:old].copy_(model.context_encoder[0].weight);ctx.bias[:old].copy_(model.context_encoder[0].bias)
        fusion.weight[:old].zero_();fusion.weight[:old,:model.config.model.d_model+old].copy_(model.fusion[0].weight)
        fusion.bias[:old].copy_(model.fusion[0].bias)
        delta.weight.zero_();delta.weight[:,:old].copy_(model.delta_head.weight);delta.bias.copy_(model.delta_head.bias)
        value.weight.zero_();value.weight[:,:old].copy_(model.value_head.weight);value.bias.copy_(model.value_head.bias)
    model.context_encoder[0]=ctx;model.fusion[0]=fusion;model.delta_head=delta;model.value_head=value
    model.settings['hidden']=width
    return model


def prepare(model,items):
    import torch
    device=next(model.parameters()).device
    result=[]
    with torch.inference_mode():
        for i in range(0,len(items),256):
            batch=model.prepare(_batch([item['row'] for item in items[i:i+256]],device))
            for j in range(len(batch['legal_mask'])):
                result.append({k:batch[k][j:j+1].cpu().numpy().copy() for k in ('poker_features','ref_logits','legal_mask','match_ctx','token_ids','type_ids')})
    return result


def measures(model,items):
    import torch
    device=next(model.parameters()).device;ces=[];vs=[];ps=[]
    with torch.inference_mode():
        for i in range(0,len(items),256):
            sub=items[i:i+256];logits,value=model(_batch([r['row'] for r in sub],device))
            p=logits.softmax(-1).cpu().numpy();ps.append(p)
            target=np.stack([r.get('target',r['prior']) for r in sub])
            ces.extend(-(target*np.log(np.maximum(p,1e-12))).sum(1))
            vs.extend(np.abs(value.cpu().numpy()-np.array([r.get('value',0.) for r in sub])))
    return np.asarray(ces),np.asarray(vs),np.concatenate(ps)


def fit(model,items,epochs,kl=.01):
    import torch
    from torch.nn import functional as F
    cached=prepare(model,items);device=next(model.parameters()).device
    target=torch.as_tensor(np.stack([r['target'] for r in items]),device=device)
    values=torch.as_tensor(np.array([r['value'] for r in items]),device=device)
    prior=torch.as_tensor(np.stack([r['prior'] for r in items]),device=device)
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=3e-4)
    model.settings.update(beta_kl=kl,late_beta_kl=kl)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=3e-6)
    # Inference tensors are copied into ordinary training tensors by _batch.
    model.train()
    for epoch in range(epochs):
        order=np.random.default_rng(92110+epoch).permutation(len(items))
        for ids in np.array_split(order,max(1,(len(items)+255)//256)):
            logits,value=model(_batch([cached[i] for i in ids],device));lp=logits.log_softmax(-1)
            ce=-(target[ids]*lp).sum(-1).mean()
            reg=(prior[ids]*(prior[ids].clamp_min(1e-12).log()-lp)).sum(-1).mean()
            # V_root is in centered root BB, exactly as returned by the solver.
            vl=F.smooth_l1_loss(value,values[ids])
            loss=ce+kl*reg+.1*vl
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite joint distillation')
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
        schedule.step()
        print('OVERNIGHT_POLICY_VALUE_EPOCH',epoch+1,'/',epochs,'policy',float(ce.detach()),'value',float(vl.detach()),flush=True)
    model.eval()
    return dict(loss=float(loss.detach()),kl=kl,value_weight=.1,hidden=model.settings['hidden'])


def paired_improvement(before,after,minimum=1e-5,groups=None):
    return paired_gain(np.asarray(before)-np.asarray(after),groups,minimum)


def gate(model,baseline,items,anchors,strict=False):
    ce,ve,p=measures(model,items);bc,bv,bp=measures(baseline,items)
    drift=float((.5*np.abs(measures(model,anchors)[2]-measures(baseline,anchors)[2]).sum(1)).max())
    result=dict(candidate_ce=float(ce.mean()),baseline_ce=float(bc.mean()),candidate_value_mae=float(ve.mean()),
                baseline_value_mae=float(bv.mean()),preflop_max_tv=drift,postflop_max_tv=float((.5*np.abs(p-bp).sum(1)).max()),
                significance=paired_improvement(bc,ce,groups=[r['rollout_seed'] for r in items]))
    result['accepted']=bool(np.isfinite(ce).all() and np.isfinite(ve).all() and ce.mean()<=bc.mean()+1e-6 and
        ve.mean()<=bv.mean()+1e-4 and drift<=1e-12 and (not strict or result['significance']['accepted']))
    return result


def surgery_fn(model,contrast,tuning):
    """Actual activation-contrast Astra rank-one candidates; no gate-data access."""
    import torch
    from surgery import direction,column_scores,apply_edit
    ce,_,_=measures(model,contrast)
    order=np.argsort(ce);n=max(2,len(order)//3);good=order[:n];bad=order[-n:]
    device=next(model.parameters()).device
    sites=[('context_encoder.0.weight',model.context_encoder[0]),('fusion.0.weight',model.fusion[0]),('delta_head.weight',model.delta_head)]
    records={};hooks=[]
    for path,layer in sites:
        def hook(module,args,out,key=path):
            records[key]=(args[0].detach().cpu().numpy(),out.detach().cpu().numpy())
        hooks.append(layer.register_forward_hook(hook))
    try:
        with torch.inference_mode():model(_batch([r['row'] for r in contrast],device))
    finally:
        for h in hooks:h.remove()
    best=model;score=float(measures(model,tuning)[0].mean());grid=[];winner=None
    original=dict(model.named_parameters())
    for path,_ in sites:
        a,h=records[path]
        unit,info=direction(h[bad],h[good])
        scores=column_scores(a[bad],a[good],unit,original[path])
        for alpha in (-.10,.10,.25):
            for k in (16,32):
                edited=copy.deepcopy(model)
                touched=apply_edit(dict(edited.named_parameters())[path],unit,scores,k,alpha)
                value=float(measures(edited,tuning)[0].mean())
                row=dict(site=path,alpha=alpha,k=k,touched=touched,tuning_ce=value,contrast=info)
                grid.append(row)
                if touched and np.isfinite(value) and value<score:best,score,winner=edited,value,row
    return best,dict(called=True,candidates=grid,winner=winner,rank=1)


def one_generation(source,dest,champion,nets,seed,plan,workers,device,smoke=False,surgeon=surgery_fn):
    from alphanl.stage2.train import load_checkpoint,save_checkpoint
    model,_=load_checkpoint(source,device=device)
    initial,_=load_checkpoint(champion,device='cpu')
    bind_champion(model,initial);baseline=copy.deepcopy(model).eval();model=expand(model)
    dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    splits={};hashes={}
    # Each split occupies a different 10-billion namespace; no gate data tune edits.
    for j,name in enumerate(('train','validation','surgery_tune','surgery_gate','final','anchor')):
        n=plan['rebel_n'] if name=='train' else plan['holdout_n']
        items=collect_teacher(source,n,seed+j*10000000000,nets,workers,2 if smoke else {'1':32,'2':32,'3':64},1 if smoke else 2,name=='anchor')
        # Preserve raw observations and precise supervision for an independent audit.
        payload={k:np.concatenate([r['row'][k] for r in items]) for k in items[0]['row']}
        payload.update(cluster_ids=np.array([r['rollout_seed'] for r in items],dtype=np.int64),target=np.stack([r.get('target',r['prior']) for r in items]),
                       value=np.array([r.get('value',0.) for r in items]),prior=np.stack([r['prior'] for r in items]))
        path=dest.with_name(dest.stem+'.'+name+'.npz')
        cfv.publish(path,payload,n,0,'joint_distillation',dict(tree=TREE,identities=[r['identity'] for r in items],teacher_sha256=cfv.digest(nets)))
        hashes[name]=cfv.digest(path);splits[name]=items
    training=fit(model,splits['train'],plan['policy_epochs'])
    distill_gate=gate(model,baseline,splits['validation'],splits['anchor'])
    rollback=[]
    if not distill_gate['accepted']:model=copy.deepcopy(baseline);rollback.append('distillation')
    edited,surgery=surgeon(copy.deepcopy(model),splits['train'][:min(512,len(splits['train']))],splits['surgery_tune'])
    sg=gate(edited,model,splits['surgery_gate'],splits['anchor'],strict=True)
    if sg['accepted']:model=edited
    else:rollback.append('surgery')
    final=gate(model,baseline,splits['final'],splits['anchor'])
    if not final['accepted']:model=baseline;rollback.append('final_holdout')
    metadata=dict(training=training,tree=TREE,source_checkpoint=str(source),source_sha256=cfv.digest(source),
        original_champion_sha256=cfv.digest(champion),split_sha256=hashes,distillation_gate=distill_gate,
        surgery=dict(search=surgery,gate=sg),final=final,rollback=rollback,
        accepted=bool(final['accepted'] and final['postflop_max_tv']>1e-8 and 'final_holdout' not in rollback),
        preflop_route='original_champion_frozen_residual',teacher='pure_signed_dcfr_policy_and_centered_root_bb')
    save_checkpoint(dest,model,metadata=metadata)
    cfv.atomic_json(str(dest)+'.manifest.json',dict(metadata,sha256=cfv.digest(dest),complete=True))
    return metadata
