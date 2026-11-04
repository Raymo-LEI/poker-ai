"""Bounded public match priors. No private cards, network columns or tokens."""
import numpy as np


def profile(context, street, snapshot=None, eta_max=.35):
    successes = getattr(context, 'successes', {})
    chances = getattr(context, 'opportunities', {})
    fold_key = 'fold_to_preflop_raise' if street == 0 else 'fold_to_postflop_bet'
    aggro_key = ('pfr', 'flop_aggression', 'turn_aggression', 'river_aggression')[street]
    folds, actions = chances.get(fold_key, 0), chances.get(aggro_key, 0)
    if not 0 <= eta_max <= .35:raise ValueError("invalid HUD eta bound")
    sample_eta = min(eta_max, max(0., float(getattr(context, 'hands_played', 0))) / 40.)
    risk, scale = 0., 20.
    if snapshot is not None:
        risk = max(float(snapshot.pot), float(snapshot.to_call)) / snapshot.big_blind
        lead = abs(float(getattr(context, 'hero_net_bb', 0.)))
        if lead > 0:
            scale = max(2., min(scale, lead / 2.))
    eta = sample_eta * float(np.exp(-risk / scale)) if risk < 40. else 0.
    # A hand count alone supplies no evidence about an unobserved response.
    eta = eta if folds or actions else 0.
    return dict(eta=eta, eta_sample=sample_eta, pot_risk_bb=risk, risk_scale_bb=scale,
                street=street, fold=(successes.get(fold_key, 0.) + 3.) / (folds + 6.),
                aggression=(successes.get(aggro_key, 0.) + .35 * 6.) / (actions + 6.),
                fold_observations=folds, action_observations=actions)


def response_prior(labels, strength, info, context=None, fraction=None):
    """A strength-aware Fold/Call/Raise model, projected onto legal children."""
    width = len(strength)
    out = np.zeros((width, len(labels)))
    fold_rate, aggression = info['fold'], info['aggression']
    if fraction is not None:
        bucket = 'small' if fraction <= .4 else 'overbet' if fraction > 1. else 'regular'
        n, folds, raises = getattr(context, 'response_by_size', {}).get('%d:%s' % (info['street'], bucket), (0, 0, 0))
        fold_rate = (folds + 4. * fold_rate) / (n + 4.)
        aggression = (raises + 4. * aggression) / (n + 4.)
    fold = np.clip(fold_rate - .45 * (strength - .5), .02, .96)
    raise_p = np.clip((1. - fold) * aggression + .15 * (strength - .5), .01, .8)
    raise_p = np.minimum(raise_p, .98 - fold)
    if 'fold' in labels:
        for i, label in enumerate(labels):
            out[:, i] = fold if label == 'fold' else 1. - fold - raise_p if label == 'call' else raise_p
    else:
        # After a check, the other legal children are bet sizes.
        for i, label in enumerate(labels):
            out[:, i] = 1. - info['aggression'] if label == 'check' else info['aggression'] / max(1, len(labels) - 1)
    out /= out.sum(axis=1, keepdims=True)
    return out


def condition_range(snapshot, context, base, strength, info):
    """RNR mixture on full combo support, preserving the current public Bayes.

    With a uniform base this is exactly (1-eta)*uniform + eta*context_size.
    An existing history posterior is retained as the baseline, rather than
    erased or counted twice. Size fingerprints describe style, not card truth.
    """
    eta = info['eta']
    if eta == 0:
        return base.copy()
    likelihood = np.ones(len(base))
    for event in getattr(snapshot, 'events', ()):
        if event.actor_is_me or event.chips_added <= event.to_call_before:
            continue
        size = max(0., float(event.pot_fraction))
        if event.street == 0:
            count = getattr(context, 'sb_open_count', 0)
            if not count or event.position != 'SB':
                continue
            mean = getattr(context, 'sb_open_mean_bb', 0.)
            variance = getattr(context, 'sb_open_m2_bb', 0.) / count
            z = (event.chips_added / snapshot.big_blind - mean) / max(1., variance**.5)
            tilt = np.clip(z, -2., 2.) * .4
        else:
            count = getattr(context, 'postflop_size_count', 0)
            common = ((getattr(context, 'postflop_overbets', 0) + 1.) / (count + 2.) if size > 1.
                      else (getattr(context, 'postflop_small_bets', 0) + 1.) / (count + 2.))
            tilt = (.5 + min(size, 3.)) * (info['fold'] - info['aggression']) * (1. - .5 * common)
        likelihood *= np.exp(np.clip(tilt * (2. * strength - 1.), -2., 2.))
    target = base * likelihood
    target /= target.sum()
    return (1. - eta) * base + eta * target


class ContextPriors:
    """One public MatchCtx snapshot per resolve; no live dictionary references."""
    def __init__(self, context, street, snapshot, eta_max):
        self.profile=profile(context,street,snapshot,eta_max)
        p=self.profile
        self.eta=float(p['eta']);self.sample_eta=float(p['eta_sample'])
        self.risk_scale=float(p['risk_scale_bb']);self.observed=bool(p['fold_observations'] or p['action_observations'])
        self.binding=(int(street),float(snapshot.pot),float(snapshot.to_call),float(snapshot.big_blind),float(eta_max))
        responses=getattr(context,'response_by_size',{})
        rates=[]
        for bucket in ('small','regular','overbet'):
            n,folds,raises=responses.get('%d:%s'%(street,bucket),(0,0,0))
            rates.append(((folds+4.*p['fold'])/(n+4.),(raises+4.*p['aggression'])/(n+4.)))
        rates.append((p['fold'],p['aggression']))
        self.rates=np.asarray(rates,dtype=np.float64)
        count=float(getattr(context,'sb_open_count',0))
        self.style=np.array([count,float(getattr(context,'sb_open_mean_bb',0.)),
            float(getattr(context,'sb_open_m2_bb',0.))/max(count,1.),
            float(getattr(context,'postflop_size_count',0)),float(getattr(context,'postflop_overbets',0)),
            float(getattr(context,'postflop_small_bets',0)),p['fold'],p['aggression']],dtype=np.float64)
        self.rates.setflags(write=False);self.style.setflags(write=False)
    def node_eta(self,risk_bb):
        if not self.observed or risk_bb>=40.:return 0.
        return min(self.eta,self.sample_eta*float(np.exp(-risk_bb/self.risk_scale)))


def prepare_context(context,street,snapshot,eta_max=.35):
    return ContextPriors(context,int(street),snapshot,float(eta_max))


class SigmaPriors:
    """All size buckets and opponent action schemas, vectorized exactly once."""
    def __init__(self,cache,strength):
        x=np.asarray(strength,dtype=np.float64)[None,:]
        fold=np.clip(cache.rates[:,0,None]-.45*(x-.5),.02,.96)
        raise_p=np.clip((1.-fold)*cache.rates[:,1,None]+.15*(x-.5),.01,.8)
        raise_p=np.minimum(raise_p,.98-fold)
        full=np.stack((fold,1.-fold-raise_p,raise_p),axis=-1)
        self.fcr=full/full.sum(axis=2,keepdims=True)
        pair=full[:,:,:2];self.fc=pair/pair.sum(axis=2,keepdims=True)
        aggression=float(cache.style[7]);n=len(strength)
        self.check2=np.tile([1.-aggression,aggression],(n,1))
        self.check3=np.tile([1.-aggression,aggression/2.,aggression/2.],(n,1))
        self.check4=np.tile([1.-aggression,aggression/3.,aggression/3.,aggression/3.],(n,1))
        for a in (self.fcr,self.fc,self.check2,self.check3,self.check4):a.setflags(write=False)
    def at(self,labels,fraction):
        if labels[0]=='check':
            if len(labels)==2:return self.check2
            if len(labels)==3:return self.check3
            if len(labels)==4:return self.check4
        bucket=3 if fraction is None else 0 if fraction<=.4 else 2 if fraction>1. else 1
        if labels==['fold','call']:return self.fc[bucket]
        if labels==['fold','call','raise_all_in']:return self.fcr[bucket]
        raise ValueError('unsupported cached opponent action schema: '+str(labels))


def condition_cached_range(snapshot,base,strength,cache):
    if cache.eta==0:return base.copy()
    count,mean,variance,post,over,small,fold,aggression=cache.style
    likelihood=np.ones(len(base))
    for event in getattr(snapshot,'events',()):
        if event.actor_is_me or event.chips_added<=event.to_call_before:continue
        size=max(0.,float(event.pot_fraction))
        if event.street==0:
            if not count or event.position!='SB':continue
            z=(event.chips_added/snapshot.big_blind-mean)/max(1.,variance**.5)
            tilt=np.clip(z,-2.,2.)*.4
        else:
            common=(over+1.)/(post+2.) if size>1. else (small+1.)/(post+2.)
            tilt=(.5+min(size,3.))*(fold-aggression)*(1.-.5*common)
        likelihood*=np.exp(np.clip(tilt*(2.*strength-1.),-2.,2.))
    target=base*likelihood;target/=target.sum()
    return (1.-cache.eta)*base+cache.eta*target
