"""Range-conditioned public CFV generation and residual training.

Production counts are explicit CLI options. The smoke preset is deliberately
small and writes to its own directory; it never reuses legacy label artifacts.
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO / 'src') not in sys.path:
    sys.path.insert(0, str(REPO / 'src'))
from alphanl.stage2 import deep_solver_v2 as v2
from alphanl.stage2 import cfv

OUT = REPO / 'eval/evening_v2_repair/smoke'
RIVER_N, TURN_N, EPOCHS_NET = 300000, 200000, 30


def _snap(hero, board, pot, call, stack, street, opp_stack=None):
    return SimpleNamespace(hole_cards=tuple(hero), board_cards=tuple(board),
        pot=float(pot), to_call=float(call), me_stack=float(stack),
        opp_stack=float(stack if opp_stack is None else opp_stack), big_blind=100.,
        street=street, legal_action_ids=(0,1,2,3,4) if call else (1,2,3,4),
        events=(), me_seat=0)


def sample_spot(seed, street):
    rng = np.random.RandomState(int(seed))
    deck = list(v2.DECK)
    rng.shuffle(deck)
    board, hero = deck[:street+2], deck[street+2:street+4]
    pot, stack, opp = rng.uniform(400,6000), rng.uniform(800,18000), rng.uniform(800,18000)
    call = 0. if rng.rand()<.5 else min(stack*.7,pot*.6)
    rh, ro = rng.dirichlet(np.full(32,.7)), rng.dirichlet(np.full(32,.7))
    snap=_snap(hero,board,pot,call,stack,street,opp)
    snap.first_player=0 if call else int(rng.randint(2))
    snap.me_seat=int(rng.randint(2)) if call else snap.first_player
    return snap,rh,ro


def label_spot(snap, rh, ro, weights=None, iterations=None, chance_samples=2, seed=0):
    # Map is fixed before solving. Exactly these ranges enter both feature and target.
    bucket_map = v2.build_public_bucket_map(snap.board_cards)
    sol = v2.solve_public_subgame(snap,rh,ro,weights=weights,iterations=iterations,
                                  chance_samples=chance_samples,seed=seed,utility=False,first_player=getattr(snap,"first_player",0))
    assert all(bucket_map[h] == b for h,b in zip(sol.hands,sol.bins))
    target = v2.bucket_values(sol.state, sol.rh*sol.rootmass[0], sol.bins)
    feat = cfv.spot_features(snap.board_cards,rh,ro,snap.pot,snap.to_call,snap.me_stack,
                            snap.big_blind,snap.street,snap.opp_stack,
                            0 if getattr(snap,"me_seat",0)==getattr(snap,"postflop_first_seat",0) else 1)
    return feat.astype(np.float32),target.astype(np.float32),sol


def _river_job(seed):
    snap,rh,ro=sample_spot(seed,3)
    return label_spot(snap,rh,ro)[:2]


def _turn_job(payload):
    seed,path=payload
    weights,_=cfv.load_artifact(path,'river_net',min_epochs=1)
    cfv.validate_weights(weights,('cfv_river.',))
    snap,rh,ro=sample_spot(seed,2)
    return label_spot(snap,rh,ro,weights)[:2]


def _predict(weights,feat,prefix='cfv_river.'):
    return cfv.predict_residual_cfv(weights,prefix,feat)


def _train(feats, targets, prefix, epochs=30, seed=0, device='cpu', valid=None,cosine=False):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if feats.shape != (len(feats),136) or targets.shape != (len(feats),32):
        raise ValueError('bad CFV training contract')
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp=nn.Linear(136,256)
            self.blocks=nn.ModuleList([nn.Linear(256,256) for _ in range(3)])
            self.out=nn.Linear(256,32)
        def forward(self,x):
            h=nn.functional.gelu(self.inp(x))
            for layer in self.blocks:
                h=h+nn.functional.gelu(layer(h))
            return self.out(h)
    model=Net().to(device)
    x=torch.as_tensor(feats,device=device)
    y=torch.as_tensor(targets,device=device)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=1e-5) if cosine else None
    for epoch in range(epochs):
        order=torch.randperm(len(x),device=device)
        for ids in order.split(256):
            loss=nn.functional.smooth_l1_loss(model(x[ids]),y[ids])
            if not torch.isfinite(loss):raise FloatingPointError("nonfinite CFV loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        print('CFV_EPOCH',prefix,epoch+1,'/',epochs,'last_batch_loss',float(loss.detach()),flush=True)
        if scheduler is not None:scheduler.step()
    layers=[model.inp]+list(model.blocks)+[model.out]
    weights={'cfv_schema':np.frombuffer(cfv.SCHEMA.encode(),dtype=np.uint8)}
    for i,layer in enumerate(layers):
        weights[prefix+'%d.weight'%i]=layer.weight.detach().cpu().numpy().copy()
        weights[prefix+'%d.bias'%i]=layer.bias.detach().cpu().numpy().copy()
    with torch.no_grad():
        ref=model(x[:64]).cpu().numpy()
    error=float(np.max(np.abs(ref-cfv.predict_residual_cfv(weights,prefix,feats[:64]))))
    if not np.isfinite(error) or error > 1e-4: raise ValueError('Torch / portable residual mismatch')
    stats={'loss':float(loss.detach()),'forward_max_error':error}
    if valid is not None:
        vx,vy=valid
        errors=np.abs(cfv.predict_residual_cfv(weights,prefix,vx)-vy)
        stats['heldout_mae']=float(np.mean(errors))
        stats['heldout_constant_mae']=float(np.mean(np.abs(targets.mean(axis=0)-vy)))
        stats['heldout_mae_pot_ratio']=float(np.mean(errors/np.maximum(vx[:,119:120]*200.,1.)))
    return weights,stats


_WORKER_NET=None


def _init_label_worker(path):
    global _WORKER_NET
    if path is not None:
        _WORKER_NET,_=cfv.load_artifact(path,'river_net',min_epochs=1)
        cfv.validate_weights(_WORKER_NET,('cfv_river.',))


def _label_worker(payload):
    import time
    started=time.monotonic()
    seed,street,iterations,chance_samples=payload
    snap,rh,ro=sample_spot(seed,street)
    x,y,_=label_spot(snap,rh,ro,_WORKER_NET,iterations,chance_samples,seed)
    return seed,x,y,started,time.monotonic()


def make_dataset(out,street,count,seed,weights=None,iterations=None,chance_samples=2,workers=1,nets_path=None):
    import time
    started=time.monotonic()
    if workers>1:
        from evening_parallel import jobs
        payloads=((seed+i,street,iterations,chance_samples) for i in range(count))
        rows=sorted(jobs(_label_worker,payloads,count,workers,_init_label_worker,(None if nets_path is None else str(nets_path),),label='CFV_'+str(street)))
        x=np.stack([r[1] for r in rows]);y=np.stack([r[2] for r in rows])
    else:
        rows=[]
        for i in range(count):
            snap,rh,ro=sample_spot(seed+i,street)
            rows.append(label_spot(snap,rh,ro,weights,iterations,chance_samples,seed+i)[:2])
        x,y=map(np.stack,zip(*rows))
    active_s=(max(r[4] for r in rows)-min(r[3] for r in rows)) if workers>1 else time.monotonic()-started
    cfv.publish(out,{'x':x,'y':y,'seeds':np.arange(seed,seed+count)},count,0,'dataset',
                {'street':street,'seed_start':seed,'iterations':iterations,'chance_samples':chance_samples,'workers':workers,
                 'elapsed_s':time.monotonic()-started,'active_compute_s':active_s,'teacher_cfv':None if weights is None else _array_hash(weights)})
    return x,y


def _array_hash(weights):
    import hashlib
    h=hashlib.sha256()
    for k in sorted(weights):
        h.update(k.encode());h.update(np.ascontiguousarray(weights[k]).tobytes())
    return h.hexdigest()


def phase_b(out=OUT, river_n=RIVER_N, turn_n=TURN_N, epochs=EPOCHS_NET,
            iterations=None, chance_samples=2, device='cpu', heldout_n=2,workers=1,river_iters=None,turn_iters=None):
    out=Path(out)
    # Fail closed: no existence-only reuse and no mixing old schema/data.
    ri=river_iters or iterations;ti=turn_iters or iterations
    rx,ry=make_dataset(out/'river_ds.npz',3,river_n,110000,None,ri,chance_samples,workers)
    rv=make_dataset(out/'river_holdout.npz',3,heldout_n,910000,None,ri,chance_samples,workers)
    rw,rs=_train(rx,ry,'cfv_river.',epochs,1,device,rv)
    cfv.publish(out/'cfv_river.npz',rw,river_n,epochs,'river_net',{'validation':rs,'dataset_sha256':cfv.digest(out/'river_ds.npz')})
    tx,ty=make_dataset(out/'turn_ds.npz',2,turn_n,210000,rw,ti,chance_samples,workers,out/'cfv_river.npz')
    tv=make_dataset(out/'turn_holdout.npz',2,heldout_n,920000,rw,ti,chance_samples,workers,out/'cfv_river.npz')
    tw,ts=_train(tx,ty,'cfv_turn.',epochs,2,device,tv)
    both=dict(rw);both.update(tw)
    cfv.publish(out/'cfv_nets.npz',both,river_n+turn_n,epochs,'nets',
                {'river_samples':river_n,'turn_samples':turn_n,'validation':{'river':rs,'turn':ts},
                 'dataset_sha256':{'river':cfv.digest(out/'river_ds.npz'),'turn':cfv.digest(out/'turn_ds.npz')}})
    stats={'river':rs,'turn':ts,'river_n':river_n,'turn_n':turn_n,'epochs':epochs,'schema':cfv.SCHEMA}
    cfv.atomic_json(out/'progress.json',stats)
    return stats


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--river-n',type=int,default=RIVER_N)
    p.add_argument('--turn-n',type=int,default=TURN_N)
    p.add_argument('--epochs',type=int,default=EPOCHS_NET)
    p.add_argument('--device',default='cpu')
    a=p.parse_args()
    print(json.dumps(phase_b(a.out,4 if a.smoke else a.river_n,3 if a.smoke else a.turn_n,
                             1 if a.smoke else a.epochs,2 if a.smoke else None,
                             1 if a.smoke else 2,a.device)),flush=True)

if __name__=='__main__':main()
