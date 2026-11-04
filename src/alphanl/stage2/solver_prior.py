"""Bounded public match priors. No private cards, network columns or tokens."""
import numpy as np


def profile(context, street, snapshot=None):
    successes = getattr(context, 'successes', {})
    chances = getattr(context, 'opportunities', {})
    fold_key = 'fold_to_preflop_raise' if street == 0 else 'fold_to_postflop_bet'
    aggro_key = ('pfr', 'flop_aggression', 'turn_aggression', 'river_aggression')[street]
    folds, actions = chances.get(fold_key, 0), chances.get(aggro_key, 0)
    sample_eta = min(.35, max(0., float(getattr(context, 'hands_played', 0))) / 40.)
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
