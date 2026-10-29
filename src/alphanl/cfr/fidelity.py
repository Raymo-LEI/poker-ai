"""Own-reach exact mixture labels, opponent coverage and weighted distillation.

All retained profiles contribute to each label. The posterior at an infoset
is proportional to iteration weight times that profile's own action reach.
"""
from collections import Counter
from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import numpy as np
import torch
from .actions import CFRActionGuard
from .distillation import semantics_match
from .evaluation import RuleActor
from .model import RegretNetwork
from alphanl.transformer.hunl import HunlTable, HunlTokenizer
from alphanl.transformer.model import PokerTransformer

OBS_KEYS = ('token_ids','type_ids','numeric','pad_mask','legal_mask')
RULES = ('random','call','maniac','overbet','nit','equity')


def postprocess(probs,*,temperature=1.,cutoff=0.):
    if not math.isfinite(temperature) or temperature<=0 or not 0<=cutoff<1:
        raise ValueError('invalid inference processing')
    if temperature==1. and cutoff==0.:return probs
    sharpened=probs.clamp_min(0).pow(1/temperature)
    keep=sharpened/sharpened.sum(-1,keepdim=True)>=cutoff
    keep.scatter_(1,sharpened.argmax(-1,keepdim=True),True)
    sharpened=sharpened*keep
    return sharpened/sharpened.sum(-1,keepdim=True)


class ExactTeacher:
    def __init__(self, directory, config, manifest, *, device='cpu', progress=None):
        if not semantics_match(manifest['semantics'],config):
            raise ValueError('teacher semantics differ')
        self.config, self.manifest, self.device = config, manifest, torch.device(device)
        self.models = []
        self.weights = torch.tensor([r['sampling_weight'] for r in manifest['snapshots']],dtype=torch.float64)
        for index,record in enumerate(manifest['snapshots']):
            path = Path(directory)/record['file']
            if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
                raise ValueError('teacher snapshot changed')
            blob = torch.load(path,map_location='cpu',weights_only=True)
            with torch.random.fork_rng(devices=[]):
                model = RegretNetwork(PokerTransformer(**asdict(config.model)),fallback=config.fallback)
            model.load_state_dict(blob['models']['shared'],strict=True)
            self.models.append(model.to(device).eval().requires_grad_(False))
            if progress and (index+1)%25==0: progress('load',index+1,len(manifest['snapshots']))

    @torch.inference_mode()
    def label(self, data, *, batch_size=256, progress=None):
        n = len(data['parent'])
        if n == 0: raise ValueError('empty states')
        parent, action = data['parent'].long(), data['action'].long()
        if ((parent < -1)|(parent >= torch.arange(n))).any():
            raise ValueError('parents must precede their children')
        linked = parent >= 0
        if not torch.equal(data['hand_id'][parent[linked]],data['hand_id'][linked]):
            raise ValueError('own reach cannot cross hands')
        depth = torch.zeros(n,dtype=torch.long)
        for i in range(n):
            if parent[i] >= 0: depth[i] = depth[parent[i]]+1
        layers = [(depth==d).nonzero().flatten() for d in range(1,int(depth.max())+1)]
        numerator, denominator = torch.zeros(n,5,dtype=torch.float64),torch.zeros(n,dtype=torch.float64)
        for index,model in enumerate(self.models):
            probs = torch.cat([model.strategy({k:data[k][s:s+batch_size].to(self.device) for k in OBS_KEYS}).cpu()
                               for s in range(0,n,batch_size)])
            reach = torch.ones(n,dtype=torch.float64)
            for ids in layers:
                ancestors = parent[ids]
                reach[ids] = reach[ancestors]*probs[ancestors,action[ancestors]].double()
            weight = self.weights[index]*reach
            numerator += weight[:,None]*probs.double()
            denominator += weight
            if progress: progress('label',index+1,len(self.models))
        valid = denominator > 0
        targets = torch.zeros(n,5)
        targets[valid] = (numerator[valid]/denominator[valid,None]).float()
        targets[valid] /= targets[valid].sum(-1,keepdim=True)
        return {**data,'target_probs':targets,'teacher_reachable':valid,
                'teacher_own_reach_mass':denominator/self.weights.sum()}


def posterior_mixture(component_probs, own_reach, iteration_weights):
    """Pure exact-label kernel; undefined all-zero own reach is rejected."""
    weights = own_reach.double()*iteration_weights.double()[:,None]
    total = weights.sum(0)
    if (total<=0).any(): raise ValueError('teacher-unreachable infoset')
    return (weights[:,:,None]*component_probs.double()).sum(0)/total[:,None]


@torch.inference_mode()
def collect_pool_states(teacher, *, hands, seed, hand_offset=0, student=None,
                        max_decisions=200000, progress=None):
    config,device = teacher.config,teacher.device
    tokenizer = HunlTokenizer(config.model.max_seq_len,**config.tokenizer_kwargs())
    guard = CFRActionGuard.from_config(config)
    rows, actions, parents, hand_ids, streets, pots, seats, opponents, origins, sprs = [],[],[],[],[],[],[],[],[],[]
    counts = Counter()
    weights = teacher.weights.numpy(); weights = weights/weights.sum()
    for hand in range(hand_offset,hand_offset+hands):
        rng = np.random.default_rng(np.random.SeedSequence([seed,hand,1]))
        hero = hand%2
        profile = int(rng.choice(len(teacher.models),p=weights))
        # Keep uniform-state coverage; augment with real standard rule paths.
        name = 'uniform' if rng.random()<.20 else str(rng.choice(RULES,p=[.05,.30,.15,.10,.05,.35]))
        use_student = student is not None and name!='uniform' and rng.random()<.35
        opponent = None if name=='uniform' else RuleActor(name,seed=int(rng.integers(2**31)),equity_samples=64)
        table = HunlTable(seed=(seed<<64)+hand);table.reset()
        parent = -1
        counts[name+('/student' if use_student else '/teacher')] += 1
        steps = 0
        while not table.is_over():
            seat = table.current_player()
            if seat == hero:
                snapshot = guard.apply(table.snapshot(hero))
                obs = tokenizer.encode(snapshot)
                batch = {k:v.to(device) for k,v in obs.items()}
                probs = (student.strategy(batch) if use_student else teacher.models[profile].strategy(batch))[0].cpu().double().numpy()
                action = int(rng.choice(5,p=probs/probs.sum()))
                index = len(rows)
                rows.append(obs);actions.append(action);parents.append(parent);hand_ids.append(hand)
                streets.append(snapshot.street);pots.append(snapshot.pot/100.);seats.append(hero)
                opponents.append(-1 if name=='uniform' else RULES.index(name));origins.append(int(use_student))
                sprs.append(guard.all_in_pot_ratio(snapshot))
                parent = index
                if len(rows)>max_decisions: raise ValueError('max_decisions exceeded')
            else:
                snapshot = table.snapshot(seat)
                action = int(rng.choice(guard.apply(snapshot).legal_action_ids)) if opponent is None else opponent.act(snapshot)
            table.step(action);steps+=1
            if steps>1000: raise RuntimeError('hand action limit')
        if sum(table.payoffs())!=0 or table.chips_in_play()!=40000: raise RuntimeError('chip conservation')
        if progress and (hand-hand_offset+1)%500==0:progress('collect',hand-hand_offset+1,hands)
    if not rows: raise ValueError('empty collection')
    data = {k:torch.cat([r[k] for r in rows]) for k in OBS_KEYS}
    for key,value in [('action',actions),('parent',parents),('hand_id',hand_ids),('street',streets),
                      ('pot_bb',pots),('hero_seat',seats),('opponent_id',opponents),('origin',origins),('spr',sprs)]:
        data[key] = torch.tensor(value,dtype=torch.float32 if key in ('pot_bb','spr') else torch.long)
    return data,dict(hands=hands,decisions=len(rows),hand_range=[hand_offset,hand_offset+hands],
                     sources=dict(counts),street_counts=torch.bincount(data['street'],minlength=4).tolist())


def prune_targets(targets, legal_mask, threshold=.03):
    if not 0<=threshold<1: raise ValueError('invalid pruning threshold')
    if not torch.isfinite(targets).all() or (targets<0).any(): raise ValueError('invalid targets')
    legal = legal_mask>.5
    if (targets[~legal]!=0).any() or not torch.allclose(targets.sum(-1),torch.ones(len(targets),device=targets.device),atol=1e-5):
        raise ValueError('invalid target distribution')
    keep = (targets>=threshold)&legal
    keep.scatter_(1,targets.argmax(-1,keepdim=True),True)
    pruned = targets*keep
    return pruned/pruned.sum(-1,keepdim=True)


def importance_weights(street,pot_bb):
    # Single cap: late AND large states receive at most 4x, not 16x.
    if not torch.isfinite(pot_bb).all() or (pot_bb<0).any():raise ValueError('invalid pot')
    weights = torch.ones_like(pot_bb)
    weights = torch.where(street==2,weights*2,weights)
    weights = torch.where(street==3,weights*3,weights)
    return (weights*torch.where(pot_bb>20,2.,1.)).clamp_max(4.)


def weighted_kl(logits,targets,legal_mask,weights):
    legal = legal_mask>.5
    if (not legal.any(-1).all() or not torch.isfinite(logits[legal]).all()
            or not torch.isfinite(weights).all() or (weights<=0).any()):raise ValueError('invalid weighted KL input')
    if (targets[~legal]!=0).any() or not torch.allclose(targets.sum(-1),torch.ones(len(targets),device=targets.device),atol=1e-5):
        raise ValueError('invalid KL targets')
    logp = logits.masked_fill(~legal,-torch.inf).log_softmax(-1).masked_fill(~legal,0.)
    row = (targets*(targets.clamp_min(1e-30).log()-logp)).sum(-1)
    return (row*weights).sum()/weights.sum()


@torch.inference_mode()
def fidelity_report(model,data,*,batch_size=256):
    valid = data['teacher_reachable']
    if not valid.any():raise ValueError('no reachable audit states')
    device = next(model.parameters()).device
    probs = torch.cat([model.strategy({k:data[k][s:s+batch_size].to(device) for k in OBS_KEYS}).cpu()
                       for s in range(0,len(valid),batch_size)])
    q = data['target_probs']
    kl = (q*(q.clamp_min(1e-30).log()-probs.clamp_min(1e-30).log())).sum(-1).clamp_min(0.)
    teacher_entropy=-(q*q.clamp_min(1e-30).log()).sum(-1)
    student_entropy=-(probs*probs.clamp_min(1e-30).log()).sum(-1)
    agreement = q.argmax(-1)==probs.argmax(-1)
    leakage = (probs*(q<.001)).sum(-1)
    def stats(mask):
        mask = mask&valid
        if not mask.any():return None
        high_risk = (q[:,4]<.001)&(probs[:,4]>=.02)&(data['spr']>3)&(data['legal_mask'][:,4]>.5)
        call_leak = (q[:,1]<.001)&(probs[:,1]>=.02)
        return dict(states=int(mask.sum()),teacher_to_student_kl=float(kl[mask].mean()),
                    top1_agreement=float(agreement[mask].float().mean()),
                    teacher_entropy=float(teacher_entropy[mask].mean()),student_entropy=float(student_entropy[mask].mean()),
                    total_variation=float((q[mask]-probs[mask]).abs().sum(-1).mean()/2),
                    mean_mass_on_teacher_near_zero=float(leakage[mask].mean()),
                    deep_shove_leak_fraction=float(high_risk[mask].float().mean()),
                    call_leak_fraction=float(call_leak[mask].float().mean()),
                    tail_2_to_5pct_fraction=float((((q<.001)&(probs>=.02)&(probs<=.05)).any(-1))[mask].float().mean()))
    slices={'all':stats(torch.ones_like(valid))}
    for street in range(4):
        slices[f'street_{street}']=stats(data['street']==street)
        slices[f'street_{street}/pot_gt20']=stats((data['street']==street)&(data['pot_bb']>20))
    for label,mask in [('pot_le5',data['pot_bb']<=5),('pot_5_20',(data['pot_bb']>5)&(data['pot_bb']<=20)),
                       ('pot_20_50',(data['pot_bb']>20)&(data['pot_bb']<=50)),('pot_gt50',data['pot_bb']>50)]:
        slices[label]=stats(mask)
    for i,name in enumerate(RULES):
        slices['opponent/'+name]=stats(data['opponent_id']==i)
        if name in ('call','equity'):
            for street in range(4):
                mask=(data['opponent_id']==i)&(data['street']==street)
                slices[f'opponent/{name}/street_{street}']=stats(mask)
                slices[f'opponent/{name}/street_{street}/pot_gt20']=stats(mask&(data['pot_bb']>20))
    return dict(unreachable_states=int((~valid).sum()),slices={k:v for k,v in slices.items() if v is not None},
                scope='Exact own-reach teacher, raw unpruned targets; KL teacher||student; near-zero threshold 0.001')


class FidelityTrainer:
    def __init__(self,model,data,*,learning_rate=1e-4,seed=17,target_prune=.03):
        self.model = model.eval()
        self.device = next(model.parameters()).device
        valid = data['teacher_reachable'].nonzero().flatten()
        self.data = {k:v[valid] for k,v in data.items() if k in OBS_KEYS or k in ('target_probs','street','pot_bb')}
        self.targets = prune_targets(self.data['target_probs'],self.data['legal_mask'],target_prune)
        self.weights = importance_weights(self.data['street'],self.data['pot_bb'])
        self.generator = torch.Generator().manual_seed(seed)
        self.optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=1e-4)
        self.steps = 0

    def fit(self,steps,*,batch_size=256):
        if steps<1 or batch_size<1:raise ValueError('positive fit budget required')
        total=0.
        for _ in range(steps):
            ids=torch.randint(len(self.targets),(batch_size,),generator=self.generator)
            batch={k:self.data[k][ids].to(self.device) for k in OBS_KEYS}
            loss=weighted_kl(self.model(batch),self.targets[ids].to(self.device),batch['legal_mask'],self.weights[ids].to(self.device))
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite fidelity loss')
            self.optimizer.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.)
            if not torch.isfinite(norm):raise FloatingPointError('nonfinite fidelity gradient')
            self.optimizer.step();self.steps+=1;total+=float(loss.detach())
        return total/steps
