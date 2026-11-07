"""Evaluation-only adapters on the SHA-pinned V22 policy and Round 3 CFV."""
import ast
from contextlib import contextmanager
from dataclasses import replace
import inspect
import os
import sys
from zipfile import ZipFile

import numpy as np

from alphanl.stage2 import cfv, deep_solver_v2 as exact
from alphanl.stage2.deployment_guard import apply_surgical_guard
from alphanl.stage2.search_belief import history_combo_ranges
from alphanl.stage2.solver_prior import profile, response_prior

V22_RELEASE = 'V22_SEARCH_LOCK_20261104'
V22_SHA = (
    'ba170fa399efeec704e7e3185d0b07e14ee60e82ebf37f66f5c414bb36e9b2e0',
    'e90dee519c4b98be049a59c3331b62a54361ba20990d048be4312d9193ec1e05',
)
DEPTH = {1: 8, 2: 8, 3: 16}
DESCRIPTION = {
    'E0': 'Exact V22 runtime, policy weights and vleaf',
    'E1': 'Replace nonterminal vleaf payoffs only. Round 3 heads via a next-board CFV adapter',
    'E2': 'V22 DCFR iteration counts 8/8/16 instead of 24/24/24',
    'E3': 'River-only 1,081-combo solver backend (its public ranges and betting tree), V22 KL fuse and 24 iterations',
    'E4': 'Use the Surgical guard at the V22 guard boundary, before search',
    'E5': 'Blend bounded HUD into V22 opponent response initialization, eta_max=0.10',
    'E6': 'V22 DCFR average postflop, no KL or empirical guard, network only initializes root strategy',
    'E7': 'E6 with uniform root initialization and no postflop network forward',
    'V26': 'Exact archived gen-1 A bundle',
    'V27': 'Exact archived gen-1 C bundle',
    'V28': 'Exact archived gen-1 D bundle',
    'L_no_pot_control': 'V26 Legacy minus _pot_control only',
    'L_no_blind_protection': 'V26 Legacy minus _r5_blind_protection only',
    'L_no_anti_punt_caps': 'V26 Legacy minus active-shove cap and forced-fold overbet/big-call fuses',
}
PURE = {'E6', 'E7'}


@contextmanager
def override(obj, name, value):
    original = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, original)


def uniform(snapshot, amounts):
    legal = set(snapshot.legal_action_ids) & set(amounts)
    if snapshot.to_call == 0:
        legal.discard(0)
    mask = np.array([a in legal for a in range(5)], dtype=np.float64)
    if not mask.any():
        raise ValueError('empty executable DCFR root')
    return mask, mask / mask.sum()


def pure_dcfr(wizard):
    # Keep V22's exact regret/average implementation. Replace the neural
    # zero-regret fallback with uniform so the network only initializes hero.
    source = inspect.getsource(wizard._dcfr)
    original = 'hero = np.where(total > 1e-8, hero_regret / np.maximum(total, 1e-8), prior)'
    initial_sum = 'hero_sum = hero.copy()'
    if source.count(original) != 1 or source.count(initial_sum) != 1:
        raise ValueError('V22 DCFR fallback or averaging site changed')
    source = source.replace(original, original.replace(', prior)', ', legal / legal.sum())'))
    source = source.replace(initial_sum, 'hero_sum = np.zeros_like(hero)')
    scope = dict(vars(wizard))
    exec(compile(source, '<V22 DCFR with uniform fallback>', 'exec'), scope)
    return scope['_dcfr']


def kl_fuse(wizard, agent, snapshot, prior, legal, values):
    center = float(np.dot(prior, values))
    pot_bb = max(float(snapshot.pot) / float(snapshot.big_blind), 1.)
    logits = np.log(prior + 1e-12) + wizard._beta(agent.weights) * (values - center) / pot_bb
    logits = np.where(legal > 0, logits, -1e9)
    fused = np.exp(logits - logits.max())
    return fused / fused.sum()


def continuation_payoffs(wizard, snapshot, pbs, amounts, old_leaf, nets):
    """Project range-conditioned next-street CFV onto V22's sampled hero rows."""
    pay, opp0 = wizard._payoffs(snapshot, pbs, amounts, old_leaf)
    if snapshot.street == 3 or old_leaf is None:
        return pay, opp0
    # The frozen Botzone judge's postflop first actor is canonical seat 1.
    object.__setattr__(snapshot, 'postflop_first_seat', 1)
    rh, ro = history_combo_ranges(snapshot, eta=0.)
    solver = exact.PublicSolver(snapshot, rh, ro, weights=nets, chance_samples=2,
                                context_prior=False, context_range_prior=False)
    # Recover the exact sampled hero rows and their rank order from V22.
    board = [wizard._norm(c) for c in snapshot.board_cards]
    hero = [wizard._norm(c) for c in snapshot.hole_cards]
    deck = [c for c in wizard.DECK if c not in board + hero]
    wizard._rng(snapshot).shuffle(deck)
    sampled = [hero] + [deck[2*i:2*i+2] for i in range(wizard.HANDS-1)]
    lookup = {frozenset(h): i for i, h in enumerate(solver.hands)}
    positions = np.array([lookup[frozenset(map(cfv.card, h))] for h in sampled])
    # The rows of pbs['matrix'] are already in V22 hero-strength order.
    # Reconstruct the original row order with the same deterministic equity.
    order = np.argsort(wizard._sample_hands(snapshot).mean(axis=1))
    positions = positions[order]
    blind, pot = float(snapshot.big_blind), float(snapshot.pot)
    me, opp = float(snapshot.me_stack), float(snapshot.opp_stack)

    def next_value(h, o):
        settled_pot, hero, opponent = exact.settle(pot, float(snapshot.to_call), h, o)
        hv, _ = solver.chance(('show', hero, opponent, settled_pot), solver.rh, solver.ro)
        conditional = np.divide(hv, solver.rootmass[0], out=np.zeros_like(hv), where=solver.rootmass[0] > 0)
        return (conditional[positions] + pot / (2 * blind))[:, None]

    for action in snapshot.legal_action_ids:
        if action == 0:
            continue
        invest = wizard._chips(action, snapshot, amounts)
        matched = min(invest, opp)
        if invest >= me - 1 or matched >= opp - 1:
            continue
        if action == 1 and snapshot.to_call == 0:
            pay[action, :, :, 1] = next_value(0., 0.)
        elif action == 1:
            pay[action] = next_value(invest, matched)[:, :, None]
        else:
            showdown = wizard._showdown(pot + matched, invest, pbs['matrix'], blind)
            pay[action, :, :, 1] = .5 * (showdown + next_value(invest, matched))
    return pay, opp0


def hud_response_prior(snapshot, context, opp0):
    info = profile(context, snapshot.street, snapshot, eta_max=.10)
    if info['eta'] == 0:
        return opp0
    out = opp0.copy()
    strength = (np.arange(out.shape[1]) + .5) / out.shape[1]
    if snapshot.to_call == 0:
        out[1, :, 1:] = (1-info['eta'])*out[1, :, 1:] + info['eta']*response_prior(
            ('check', 'bet'), strength, info, context)
    for action in (2, 3, 4):
        out[action] = (1-info['eta'])*out[action] + info['eta']*response_prior(
            ('fold', 'call', 'raise'), strength, info, context)
    return out


class V22:
    def __init__(self, source, model, arm='E0', nets=None):
        # Workers import only this archived stage2_runtime under its original
        # name. Modern Round 3 bundles use their own digest-namespaced modules.
        import importlib
        self.arm, self.nets = arm, nets
        self.source = str(source)
        sys.path.insert(0, self.source)
        try:
            entry = importlib.import_module('botzone_entry')
            self.runtime = importlib.import_module('stage2_runtime.runtime')
            self.wizard = importlib.import_module('stage2_runtime.wizard')
        finally:
            sys.path.pop(0)
        if (entry.RELEASE_ID != V22_RELEASE or
                not str(entry.__file__).startswith(self.source + os.sep)):
            raise ValueError('wrong V22 ZIP import')
        old_model = os.environ.get('POKER_STAGE2_WEIGHTS')
        old_guard = os.environ.pop('POKER_STAGE2_GUARD_CONFIG', None)
        os.environ['POKER_STAGE2_WEIGHTS'] = str(model)
        try:
            self.agent, self.safety, _ = entry.load_agent(self.runtime)
        finally:
            if old_model is None:
                os.environ.pop('POKER_STAGE2_WEIGHTS')
            else:
                os.environ['POKER_STAGE2_WEIGHTS'] = old_model
            if old_guard is not None:
                os.environ['POKER_STAGE2_GUARD_CONFIG'] = old_guard
        self.original_fuse = self.wizard.fuse
        self.pure_solver = pure_dcfr(self.wizard) if arm in PURE else None
        self.fires = 0

    def fuse(self, agent, snapshot, context, probs, amounts):
        w = self.wizard
        street = int(snapshot.street)
        if self.arm in PURE:
            if street == 0:
                return probs
            legal, flat = uniform(snapshot, amounts)
            warm = probs * legal
            warm = warm / warm.sum() if warm.sum() else flat
            pbs = w.build_pbs(snapshot, context, flat)
            leaf = w.leaf_vector(agent.weights, pbs['feat'])
            pay, opp0 = w._payoffs(snapshot, pbs, amounts, leaf)
            _, average = self.pure_solver(pay, pbs['r_hero'], pbs['r_opp'], legal, warm, opp0)
            self.fires += 1
            return average[pbs['hero_bucket']]
        if not w._active(snapshot):
            return self.original_fuse(agent, snapshot, context, probs, amounts)
        if self.arm == 'E2':
            with override(w, 'DCFR_ITERS', DEPTH[street]):
                self.fires += 1
                return self.original_fuse(agent, snapshot, context, probs, amounts)
        if self.arm == 'E3' and street == 3:
            legal, prior = w._legal(snapshot, probs)
            river_amounts = {a: amount for a, amount in amounts.items() if legal[a]}
            snapshot = replace(snapshot, legal_action_ids=tuple(river_amounts))
            object.__setattr__(snapshot, 'postflop_first_seat', 1)
            rh, ro = history_combo_ranges(snapshot, eta=0.)
            # A separate river backend, including its native betting tree and
            # history-conditioned ranges. Keep the V22 root/KL/depth contract.
            _, _, _, at, _ = exact._select(snapshot.board_cards, snapshot.hole_cards)
            values, _, support = exact.solve_river(
                snapshot, hero_range=rh, opponent_range=ro, amounts=river_amounts,
                iterations=w.DCFR_ITERS, context_prior=False, context_range_prior=False,
                root_prior=prior, hero_at=at)
            if support != 1081:
                raise ValueError('exact river must have 1,081 public combinations')
            values = np.where(values > -1e9, values, 0.)
            solved = set(exact.solve_river.last_solution.tree.labels)
            if any(legal[a] and a not in solved for a in range(5)):
                raise ValueError('exact river changed V22 executable domain')
            self.fires += 1
            return kl_fuse(w, agent, snapshot, prior, legal, values)
        if self.arm == 'E5' or (self.arm == 'E1' and street < 3):
            legal, prior = w._legal(snapshot, probs)
            pbs = w.build_pbs(snapshot, context, prior)
            leaf = w.leaf_vector(agent.weights, pbs['feat'])
            if self.arm == 'E1':
                pay, opp0 = continuation_payoffs(w, snapshot, pbs, amounts, leaf, self.nets)
            else:
                pay, opp0 = w._payoffs(snapshot, pbs, amounts, leaf)
                adjusted = hud_response_prior(snapshot, context, opp0)
                self.fires += int(adjusted is not opp0)
                opp0 = adjusted
            values, _ = w._dcfr(pay, pbs['r_hero'], pbs['r_opp'], legal, prior, opp0)
            if self.arm == 'E1':
                self.fires += 1
            return kl_fuse(w, agent, snapshot, prior, legal, values[pbs['hero_bucket']])
        return self.original_fuse(agent, snapshot, context, probs, amounts)

    def respond(self, payload):
        if self.arm == 'E0':
            return self.runtime.respond(self.agent, payload, self.safety)
        if self.arm == 'E4':
            def surgical(*args, **kwargs):
                self.fires += 1
                return apply_surgical_guard(*args, **kwargs)
            with override(self.runtime, 'apply_guard', surgical):
                return self.runtime.respond(self.agent, payload, self.safety)
        if self.arm in PURE and payload['requests'][-1]['public_cards']:
            # Locks remain in respond(). Once a solve starts, empirical guards
            # must not change the warm start or narrow the postflop domain.
            def no_guard(_snapshot, probabilities, *_args, **_kwargs):
                return probabilities, {}
            with override(self.wizard, 'fuse', self.fuse), override(self.runtime, 'apply_guard', no_guard):
                if self.arm == 'E7':
                    with override(self.agent, 'logits', lambda batch: np.zeros_like(batch['legal_mask'])):
                        return self.runtime.respond(self.agent, payload, self.safety)
                return self.runtime.respond(self.agent, payload, self.safety)
        with override(self.wizard, 'fuse', self.fuse):
            return self.runtime.respond(self.agent, payload, self.safety)


def legacy_variant(bundle, arm):
    guard = bundle.modules['deployment_guard']
    if arm == 'L_no_pot_control':
        guard._pot_control = lambda *_args, **_kwargs: None
    elif arm == 'L_no_blind_protection':
        guard._r5_blind_protection = lambda *_args, **_kwargs: None
    elif arm == 'L_no_anti_punt_caps':
        # The cap conditions are inline. Patch only their exact boundaries in
        # this worker's isolated V26 module and fail on source drift.
        with ZipFile(bundle.source) as archive:
            module = archive.read('stage2_runtime/deployment_guard.py').decode()
        node = next(n for n in ast.parse(module).body if isinstance(n, ast.FunctionDef) and n.name == 'apply_guard')
        source = ast.get_source_segment(module, node)
        call_fuse = '''        if info["call_fuse"]:
            audit_before=probs.copy()
            probs[:] = 0.; probs[0] = 1.
            record_guard_rule(info,"legacy.call_fuse",audit_before,probs)
            return probs, info
'''
        active_cap = ('    if 4 in legal and snapshot.me_stack > snapshot.to_call and '
                      'excessive(amount[4]) and probs[4] > 0:')
        if source.count(call_fuse)!=1 or source.count(active_cap)!=1:
            raise ValueError('V26 anti-punt cap boundary changed')
        # Preserve adaptive call protection/forced calls and the cap thresholds
        # that other rules use to select a safe probe.
        source = source.replace(call_fuse, '''        if info["call_fuse"]:
            info["arena_suppressed_call_fuse"] = info["call_fuse"]
            info["call_fuse"] = None
''')
        source = source.replace(active_cap, '    if False:  # arena ablation of active-shove remap')
        exec(compile(source, '<V26 without anti-punt caps>', 'exec'), guard.__dict__)
        # The import on runtime predates the isolated guard function replacement.
        bundle.runtime.apply_guard = guard.apply_guard
