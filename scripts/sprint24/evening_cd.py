"""Two gated distillation generations on the same strict deployed search path."""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

REPO=Path(__file__).resolve().parents[2]
if str(REPO/'src') not in sys.path:sys.path.insert(0,str(REPO/'src'))
from alphanl.stage2 import cfv
from alphanl.stage2.evening_strategy import search_policy

OUT=REPO/'eval/evening_v2_repair/smoke'
NETS=OUT/'cfv_nets.npz'
REBEL_N,EPOCHS=200000,15
_NET_CACHE={}


def wait_nets(path=None):
    """A complete validated artifact is required, not merely an existing file."""
    return cfv.load_nets(NETS if path is None else path)


def search_label(snapshot,context,prior,nets=None,amounts=None,iterations=None,chance_samples=2):
    if nets is None:
        path=Path(NETS)
        key=(str(path.resolve()),cfv.digest(path),cfv.digest(str(path)+'.manifest.json'))
        if key not in _NET_CACHE:_NET_CACHE[key]=wait_nets(path)[0]
        nets=_NET_CACHE[key]
    budget=iterations.get(str(int(snapshot.street))) if isinstance(iterations,dict) else iterations
    out,trace=search_policy(snapshot,context,prior,nets,amounts,budget,chance_samples)
    search_label.last_trace=trace
    return out.astype(np.float32)


def _batch(rows,device):
    import torch
    return {k:torch.as_tensor(np.concatenate([r[k] for r in rows],axis=0),device=device) for k in rows[0]}


def collect(model,count,seed0,nets=None,preflop=False,iterations=None,chance_samples=2):
    import torch
    from alphanl.cfr.evaluation import RuleActor
    from alphanl.stage2.env import BotzoneMatchEnv
    from alphanl.transformer.tables import standard_deck
    from ladder_actors import SlayerActor
    model.eval()
    device=next(model.parameters()).device
    rows,targets,ids=[],[],[]
    match_id=0
    while len(rows)<count:
        seed=seed0+match_id
        rng=np.random.RandomState(seed)
        decks=[rng.permutation(standard_deck()).tolist() for _ in range(50)]
        name=('call','TAG_Pro','Trapper_CR','Bot17_Probe3Bet','NearShove_19949')[(seed0+match_id)%5]
        opp=SlayerActor(name,seed=seed,equity_samples=4) if name in ('Bot17_Probe3Bet','NearShove_19949') else RuleActor(name,seed=seed,equity_samples=4)
        env=BotzoneMatchEnv(model.config,opp,decks=decks,first_hero_seat=(seed0+match_id)%2,seed=seed,
                            context_fields=model.context_fields,allin_mask=model.settings.get('botzone_native_allin_mask'))
        env.reset()
        while not env.done and len(rows)<count:
            snap=env._hero_decision_snapshot()
            obs={k:v.to(device) for k,v in env.observe().items()}
            if snap.street>0 and not preflop:
                prior=np.zeros(5,dtype=np.float32);prior[list(snap.legal_action_ids)]=1;prior/=prior.sum()
            else:
                with torch.no_grad():prior=model(obs)[0][0].softmax(-1).cpu().numpy()
            if (snap.street==0)==preflop:
                target=prior if preflop else search_label(snap,env.ctx,prior,nets,amounts={a:env.table.action_chips(a) for a in snap.legal_action_ids},iterations=iterations,chance_samples=chance_samples)
                rows.append({k:v.detach().cpu().numpy() for k,v in obs.items()})
                targets.append(target)
                ids.append((seed,env.ctx.hands_played,len(snap.events)))
            behaviour=target if snap.street>0 and not preflop else prior
            env.step(int(rng.choice(5,p=behaviour.astype(float)/behaviour.astype(float).sum())))
        match_id+=1
        if match_id>max(100,count*5):raise RuntimeError('insufficient postflop opportunities')
    collect.last_ids=ids
    return rows,np.stack(targets)


def _probabilities(model,rows):
    import torch
    device=next(model.parameters()).device
    with torch.no_grad():
        return np.concatenate([model(_batch(rows[i:i+256],device))[0].softmax(-1).cpu().numpy() for i in range(0,len(rows),256)])


def _metrics(model,rows,targets):
    p=_probabilities(model,rows)
    ce=float(-np.sum(targets*np.log(np.maximum(p,1e-12)),axis=1).mean())
    return {'ce':ce,'max_tv':float((.5*np.abs(p-targets).sum(axis=1)).max())}


def fit(model,anchor,rows,targets,anchor_rows,anchor_targets,epochs=EPOCHS):
    import torch
    device=next(model.parameters()).device
    model.eval()
    def prepare_rows(items):
        result=[]
        with torch.no_grad():
            for start in range(0,len(items),256):
                batch=model.prepare(_batch(items[start:start+256],device))
                for j in range(len(batch['legal_mask'])):
                    result.append({k:batch[k][j:j+1].detach().cpu().numpy() for k in ('poker_features','ref_logits','legal_mask','match_ctx','token_ids','type_ids')})
        return result
    rows=prepare_rows(rows);anchor_rows=prepare_rows(anchor_rows)
    model.train()
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=3e-5)
    for epoch in range(epochs):
        order=np.random.RandomState(20261104+epoch).permutation(len(rows))
        for start in range(0,len(order),256):
            ids=order[start:start+256]
            batch=_batch([rows[i] for i in ids],device)
            t=torch.as_tensor(targets[ids],device=device)
            lp=model(batch)[0].log_softmax(-1)
            ce=-(t*lp).sum(-1).mean()
            # Dedicated preflop data constrain the shared residual, independently
            # of the postflop training distribution.
            ab=_batch(anchor_rows[:256],device)
            ap=model(ab)[0].log_softmax(-1)
            at=torch.as_tensor(anchor_targets,device=device)
            anchor_ce=-(at*ap).sum(-1).mean()
            loss=ce+5*anchor_ce
            if not torch.isfinite(loss):raise FloatingPointError("nonfinite distillation loss")
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        print('DISTILL_EPOCH',epoch+1,'/',epochs,'last_batch_loss',float(loss.detach()),flush=True)
    model.eval()
    return float(loss.detach())


def validation_gate(candidate,baseline,validation,anchors,max_preflop_tv=1e-3):
    rows,targets=validation
    anchor_rows,anchor_targets=anchors
    cm,bm=_metrics(candidate,rows,targets),_metrics(baseline,rows,targets)
    drift=float((.5*np.abs(_probabilities(candidate,anchor_rows)-_probabilities(baseline,anchor_rows)).sum(axis=1)).max())
    return {'accepted':bool(cm['ce']<=bm['ce']+1e-6 and drift<=max_preflop_tv),
            'candidate':cm,'baseline':bm,'preflop_max_tv':drift,'preflop_limit':max_preflop_tv}


def one_generation(source,dest,seed,nets_path=None,count=REBEL_N,epochs=EPOCHS,
                   device='cpu',iterations=None,chance_samples=2,holdout_n=64,surgery_fn=None,workers=1):
    from alphanl.stage2.train import load_checkpoint,save_checkpoint
    nets_path=NETS if nets_path is None else Path(nets_path)
    nets,manifest=wait_nets(nets_path)
    model,source_blob=load_checkpoint(source,device=device)
    inherited=source_blob.get('metadata',{}).get('preflop_anchor')
    anchor_state=({k:v.detach().cpu().clone() for k,v in model.state_dict().items()
                  if k.startswith(('context_encoder.','fusion.','delta_head.'))} if inherited is None else inherited)
    model._evening_preflop_anchor=anchor_state
    baseline=copy.deepcopy(model).eval()
    def gather(n,s,preflop=False):
        if workers>1:
            return collect_parallel(source,n,s,nets_path,workers,preflop,iterations,chance_samples)
        value=collect(baseline,n,s,nets,preflop,iterations,chance_samples)
        return value+(list(collect.last_ids),)
    rows,targets,train_ids=gather(count,seed)
    ar,at,anchor_ids=gather(holdout_n,seed+1000000,True)
    vr,vt,valid_ids=gather(holdout_n,seed+2000000)
    fr,ft,final_ids=gather(holdout_n,seed+3000000)
    far,fat,final_anchor_ids=gather(holdout_n,seed+4000000,True)
    dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    data={k:np.concatenate([r[k] for r in rows],axis=0) for k in rows[0]}
    data['target']=targets
    dataset=dest.with_suffix('.labels.npz')
    cfv.publish(dataset,data,len(rows),0,'distillation',{'teacher_cfv_sha256':manifest['sha256'],
                  'identities':{'train':train_ids,'anchor':anchor_ids,'validation':valid_ids,'final':final_ids,'final_anchor':final_anchor_ids}})
    split_hashes={}
    for name,split_rows,split_targets,identities in (
        ('anchor',ar,at,anchor_ids),('validation',vr,vt,valid_ids),
        ('final',fr,ft,final_ids),('final_anchor',far,fat,final_anchor_ids)):
        split_path=dest.with_name(dest.stem+'.'+name+'.npz')
        payload={k:np.concatenate([r[k] for r in split_rows],axis=0) for k in split_rows[0]}
        payload['target']=split_targets
        cfv.publish(split_path,payload,len(split_rows),0,'distillation_'+name,
                    {'identities':identities,'teacher_cfv_sha256':manifest['sha256']})
        split_hashes[name]=cfv.digest(split_path)
    loss=fit(model,source,rows,targets,ar,at,epochs)
    gate=validation_gate(model,baseline,(vr,vt),(ar,at))
    rollback=[]
    if not gate['accepted']:
        model.load_state_dict(baseline.state_dict());rollback.append('distillation_validation')
    if surgery_fn is not None:
        before=copy.deepcopy(model)
        # Surgeon may tune on its own data, never on these reserved validation deals.
        edited=surgery_fn(copy.deepcopy(model))
        sg=validation_gate(edited,before,(vr,vt),(ar,at))
        if sg['accepted']:model=edited
        else:rollback.append('surgery_validation')
    else:sg={'applied':False}
    final=validation_gate(model,baseline,(fr,ft),(far,fat))
    if not final['accepted']:
        model.load_state_dict(baseline.state_dict());rollback.append('independent_final_holdout')
    metadata={'loss':loss,'samples':len(rows),'epochs':epochs,'cfv_sha256':manifest['sha256'],
              'teacher_iterations':iterations,'teacher_chance_samples':chance_samples,
              'schema':cfv.SCHEMA,'sources':dict(cfv.source_hashes(),evening_cd=cfv.digest(__file__),
                    evening_strategy=cfv.digest(REPO/'src/alphanl/stage2/evening_strategy.py')),'dataset_sha256':cfv.digest(dataset),'split_sha256':split_hashes,
              'source_checkpoint_sha256':cfv.digest(source),'collection_policy':'search_average_postflop',
              'preflop_route':'frozen_source_residual','gate':gate,'surgery':sg,'final':final,'rollback':rollback}
    save_checkpoint(dest,model,metadata=dict(metadata,preflop_anchor=anchor_state))
    cfv.atomic_json(str(dest)+'.manifest.json',dict(metadata,sha256=cfv.digest(dest),complete=True))
    return metadata


_COLLECT_MODEL=None
_COLLECT_NETS=None


def _init_collect_worker(source,nets_path):
    global _COLLECT_MODEL,_COLLECT_NETS
    from alphanl.stage2.train import load_checkpoint
    _COLLECT_MODEL,_=load_checkpoint(source,device='cpu')
    _COLLECT_NETS,_=wait_nets(nets_path)


def _collect_worker(payload):
    import time
    started=time.monotonic()
    index,count,seed,preflop,iterations,chance_samples=payload
    rows,target=collect(_COLLECT_MODEL,count,seed,_COLLECT_NETS,preflop,iterations,chance_samples)
    if any(ident[0]>=seed+128 for ident in collect.last_ids):raise RuntimeError('collection exhausted its disjoint match-seed segment')
    return index,rows,target,list(collect.last_ids),started,time.monotonic()


def collect_parallel(source,count,seed,nets_path,workers,preflop=False,iterations=None,chance_samples=2):
    from evening_parallel import jobs
    if count>200000:raise ValueError('reserved seed namespace supports at most 200000 rows per split')
    chunk=32
    payloads=[(i,min(chunk,count-i*chunk),seed+i*129,preflop,iterations,chance_samples) for i in range((count+chunk-1)//chunk)]
    results=sorted(jobs(_collect_worker,payloads,len(payloads),workers,_init_collect_worker,(str(source),str(nets_path)),timeout=240,label='DISTILL'))
    collect_parallel.last_compute_seconds=max(r[5] for r in results)-min(r[4] for r in results)
    return ([row for item in results for row in item[1]],np.concatenate([item[2] for item in results]),[ident for item in results for ident in item[3]])


def phase_c(source,out=OUT,nets_path=None,**kwargs):
    out=Path(out)
    g1,g2=out/'u250_deep_rebel_g1.pt',out/'u250_deep_rebel_g2.pt'
    stats={'g1':one_generation(source,g1,20261104,nets_path,**kwargs)}
    stats['g2']=one_generation(g1,g2,120261104,nets_path,**kwargs)
    cfv.atomic_json(out/'cd_progress.json',stats)
    return stats


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--nets',type=Path,required=True);p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--smoke',action='store_true');p.add_argument('--device',default='cpu')
    a=p.parse_args()
    print(json.dumps(phase_c(a.source,a.out,a.nets,count=4 if a.smoke else REBEL_N,
                epochs=1 if a.smoke else EPOCHS,device=a.device,iterations=1 if a.smoke else None,
                chance_samples=1 if a.smoke else 2,holdout_n=2 if a.smoke else 64)),flush=True)

if __name__=='__main__':main()
