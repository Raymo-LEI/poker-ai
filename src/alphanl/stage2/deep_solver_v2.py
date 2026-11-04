"""Full public combo DCFR with a finite, explicit no-limit betting abstraction.

Root buttons use executable extra-chip amounts. After a bet the responder can
fold, call or raise all-in; after that raise the bettor can fold/call. After a
check the other player can check, half-pot, pot or jam. No phantom responses
follow a call. This is an abstraction, not a full no-limit equilibrium claim.

Conditional CFVs use the compatible root range denominator, with unreachable
hands masked. Regrets retain signs: DCFR(1.5,0,2). Training disables match utility.
"""
import itertools
import math
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
from .cfv import (SCHEMA, BUCKETS, FEATURE_DIM, DECK, CARD_ID, card as _ev,
                  normalize_range, spot_features, predict_residual_cfv)

RIVER_ITERS = 40
TURN_ITERS = 32
RIVER_K = 1081
_PE = False


def _strength(cards):
    from .portable_evaluator import strength
    return strength(cards)


def _pack(cards):
    global _PE
    if _PE is False:
        try:
            from phevaluator import evaluate_cards
            _PE = evaluate_cards
        except ImportError:
            _PE = None
    if _PE is not None:
        return -int(_PE(*cards))
    key = _strength(cards)
    key = tuple(key) + (0,) * (6 - len(key))
    packed = 0
    for part in key:
        packed = packed * 16 + int(part)
    return packed


def pack_many(hands, board):
    """Vectorized exact 5–7-card rank, identical to padded portable strength.

    Numeric IDs are not phevaluator IDs; ordering and ties are identical.
    This avoids thousands of Python Counter/evaluator calls per public board.
    """
    n=len(hands)
    holes=np.array([[CARD_ID[_ev(c)] for c in h] for h in hands],dtype=np.int16)
    public=np.array([CARD_ID[_ev(c)] for c in board],dtype=np.int16)
    ids=np.concatenate((holes,np.broadcast_to(public,(n,len(public)))),axis=1)
    ranks=ids%13+2;suits=ids//13
    counts=np.zeros((n,15),dtype=np.int8)
    sc=np.zeros((n,4),dtype=np.int8)
    rows=np.arange(n)[:,None]
    np.add.at(counts,(rows,ranks),1);np.add.at(sc,(rows,suits),1)
    rank_axis=np.arange(15)[None,:]
    def top(mask,k):
        return np.sort(np.where(mask,rank_axis,0),axis=1)[:,-k:][:,::-1]
    def straight(present):
        out=np.zeros(n,dtype=np.int64)
        out[np.all(present[:,2:6],axis=1)&present[:,14]]=5
        for high in range(6,15):out[np.all(present[:,high-4:high+1],axis=1)]=high
        return out
    present=counts>0
    key=np.zeros((n,6),dtype=np.int64);key[:,1:]=top(present,5)
    pairs=top(counts>=2,2);trips=top(counts>=3,1)[:,0];quads=top(counts==4,1)[:,0]
    mask=pairs[:,0]>0;key[mask,0]=1;key[mask,1]=pairs[mask,0]
    kick=top(present&(rank_axis!=pairs[:,:1]),3);key[mask,2:5]=kick[mask];key[mask,5]=0
    mask=pairs[:,1]>0;key[mask,0]=2;key[mask,1:3]=pairs[mask];key[mask,3]=top(present&(rank_axis!=pairs[:,:1])&(rank_axis!=pairs[:,1:2]),1)[mask,0];key[mask,4:]=0
    mask=trips>0;key[mask,0]=3;key[mask,1]=trips[mask];key[mask,2:4]=top(present&(rank_axis!=trips[:,None]),2)[mask];key[mask,4:]=0
    run=straight(present);mask=run>0;key[mask]=0;key[mask,0]=4;key[mask,1]=run[mask]
    fs=sc.argmax(axis=1);flush=sc.max(axis=1)>=5
    fc=np.zeros((n,15),dtype=np.int8)
    np.add.at(fc,(rows,ranks),(suits==fs[:,None]).astype(np.int8))
    key[flush,0]=5;key[flush,1:]=top(fc>0,5)[flush]
    partner=top((counts>=2)&(rank_axis!=trips[:,None]),1)[:,0]
    mask=(trips>0)&(partner>0);key[mask]=0;key[mask,0]=6;key[mask,1]=trips[mask];key[mask,2]=partner[mask]
    mask=quads>0;key[mask]=0;key[mask,0]=7;key[mask,1]=quads[mask];key[mask,2]=top(present&(rank_axis!=quads[:,None]),1)[mask,0]
    sf=straight(fc>0);mask=flush&(sf>0);key[mask]=0;key[mask,0]=8;key[mask,1]=sf[mask]
    return key @ (16**np.arange(5,-1,-1,dtype=np.int64))


@lru_cache(maxsize=64)
def _public(board):
    pool = [c for c in DECK if c not in board]
    hands = tuple(itertools.combinations(pool, 2))
    rank = pack_many(hands, board)
    # Public static made-hand rank, ties broken by canonical combo identity.
    # Independent of ranges, chips, solver output and the real hero hand.
    order = np.argsort(rank, kind="mergesort")
    bins = np.empty(len(hands), dtype=np.int64)
    bins[order] = np.arange(len(hands)) * 32 // len(hands)
    return hands, rank, bins


def build_public_bucket_map(board):
    hands, _rank, bins = _public(tuple(sorted(_ev(c) for c in board)))
    return {h: int(b) for h, b in zip(hands, bins)}


def _select(board, hero):
    hands, rank, _bins = _public(tuple(sorted(_ev(c) for c in board)))
    hero = frozenset(_ev(c) for c in hero)
    hero_at = next(i for i,h in enumerate(hands) if frozenset(h) == hero)
    return (np.array([CARD_ID[h[0]] for h in hands]),
            np.array([CARD_ID[h[1]] for h in hands]), rank, hero_at, hands)


def expand_range(ranges, bins):
    if np.asarray(ranges).shape == (len(bins),):
        values = np.asarray(ranges,dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values<0) or values.sum()<=0:
            raise ValueError('invalid full-combo public range')
        return values/values.sum()
    ranges = normalize_range(ranges)
    count = np.bincount(bins, minlength=32)
    return ranges[bins] / count[bins]


def compress_range(reach, bins):
    out = np.bincount(bins, weights=reach, minlength=32).astype(np.float64)
    if out.sum() <= 0:
        return np.full(32, 1 / 32.)
    return out / out.sum()


def bucket_values(values, reach, bins):
    mass = np.bincount(bins, weights=reach, minlength=32)
    val = np.bincount(bins, weights=reach * values, minlength=32)
    return np.divide(val, mass, out=np.zeros(32), where=mass > 0)


def _groups(rank):
    order = np.argsort(rank, kind="mergesort")
    ordered = rank[order]
    change = np.flatnonzero(np.diff(ordered)) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(rank)]))
    g0 = np.empty(len(rank), dtype=np.int64)
    g1 = np.empty(len(rank), dtype=np.int64)
    for start, end in zip(starts, ends):
        g0[start:end] = start
        g1[start:end] = end
    return order, g0, g1


def equity_vs(rank, c1, c2, weights, order, g0, g1):
    """Showdown equity of every hand against ``weights``, conflicts removed. O(K)."""
    weights = np.asarray(weights, dtype=np.float64)
    width = len(rank)
    if weights.shape != (width,) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("invalid combo reach")
    ws = weights[order]
    c1s = c1[order]
    c2s = c2[order]
    pref = np.zeros(width + 1, dtype=np.float64)
    np.cumsum(ws, out=pref[1:])
    card = np.zeros((52, width + 1), dtype=np.float64)
    np.add.at(card, (c1s, np.arange(1, width + 1)), ws)
    np.add.at(card, (c2s, np.arange(1, width + 1)), ws)
    np.cumsum(card, axis=1, out=card)
    own = ws
    worse = pref[g0] - card[c1s, g0] - card[c2s, g0]
    group = pref[g1] - pref[g0]
    tie_hit = (card[c1s, g1] - card[c1s, g0]) + (card[c2s, g1] - card[c2s, g0]) - 2.0 * own
    tie_valid = np.maximum(group - own - tie_hit, 0.0)
    total_hit = (card[c1s, width] - own) + (card[c2s, width] - own)
    den = np.maximum(pref[width] - own - total_hit, 0.0)
    den[den < 1e-14] = 0.0
    equity = np.divide(worse + 0.5 * tie_valid, den, out=np.zeros_like(den), where=den > 0)
    back = np.empty(width, dtype=np.float64)
    mass = np.empty(width, dtype=np.float64)
    back[order] = equity
    mass[order] = den
    return back, mass



def _phi_blend(delta_bb, blind, hero_net_bb, hands_played):
    """Terminal-only zero-sum match objective; opponent receives its negative.

    Normal CDF denominator is sigma*sqrt(2). No U(E[chips]) is used.
    Labels always bypass this function and remain linear, zero-sum BB.
    """
    x = np.asarray(delta_bb, dtype=np.float64)
    if hands_played is None or hero_net_bb is None:
        return x
    sigma = 1200.0 * math.sqrt(max(1, 50 - int(hands_played)))
    lead = float(hero_net_bb) * blind
    scale = sigma * math.sqrt(2.0)
    erf = np.frompyfunc(math.erf, 1, 1)
    after = .5 * (1 + np.asarray(erf((lead + x * blind) / scale),dtype=float))
    before = .5 * (1 + math.erf(lead / scale))
    return .5 * x + .5 * (after - before) * sigma / blind


def settle(pot, owed, hero_invest, opp_invest=None):
    """Matched future contributions and refund of an existing unmatched bet.

    pot already contains owed. Return contested pot, H, O; excess new
    contributions are never charged. Short calls refund owed-H from pot.
    """
    h = max(0., float(hero_invest))
    if h <= owed:
        return pot - max(owed - h, 0.), h, 0.
    o = max(0., h - owed) if opp_invest is None else max(0., opp_invest)
    h = min(h, owed + o)
    o = min(o, max(h - owed, 0.))
    return float(pot), h, o


def call_value(eq, pot, owed, me_stack, blind):
    p, h, o = settle(float(pot), float(owed), min(float(owed), float(me_stack)))
    return (np.asarray(eq) * (p + h + o) - h) / float(blind)


class Node:
    def __init__(self, player=None, children=None, leaf=None, labels=None):
        self.player, self.children, self.leaf = player, children, leaf
        self.labels = labels
        self.regret = self.total = self.strategy = None

    def init(self, width):
        if self.player is not None:
            a = len(self.children)
            self.regret = np.zeros((width,a))
            self.total = np.zeros((width,a))
            self.strategy = np.full((width,a), 1./a)
            self.fallback = self.strategy.copy()
            for child in self.children:
                child.init(width)


def betting_tree(pot, owed, me, opp, legal=None, amounts=None, first_player=0, check_closes=False, check_raise=False):
    """One street, capped at one all-in re-raise. Leaves record root investments."""
    if first_player == 1:
        if owed:
            raise ValueError("owed only defined for the first actor")
        tree = betting_tree(pot,0,opp,me,legal,amounts,0,False,check_raise)
        def swap(n):
            if n.player is not None:
                n.player = 1 - n.player
                for child in n.children: swap(child)
            else:
                k,h,o,p = n.leaf
                n.leaf = (k if k == "show" else "fold1" if k == "fold0" else "fold0",o,h,p)
        swap(tree)
        return tree
    leaf = lambda k,h,o,p=pot: Node(leaf=(k,h,o,p))
    def showdown(h,o):
        p,h,o = settle(pot,owed,h,o)
        return leaf("show",h,o,p)
    def response(h):
        o = min(max(h-owed,0.),opp)
        p,hh,oo = settle(pot,owed,h,o)
        children = [leaf("fold1",0,0), leaf("show",hh,oo,p)]
        labels = ["fold","call"]
        # A jam is a legal raise only if it can exceed this bet. Short all-in
        # raises are retained, with only fold/call remaining in the abstraction.
        if opp > o and me > h:
            jam_o = opp
            jam_h = min(me,owed+jam_o)
            children.append(Node(0,[leaf("fold0",h,0),showdown(jam_h,jam_o)],labels=["fold","call"]))
            labels.append("raise_all_in")
        node = Node(1,children,labels=labels)
        node.facing_fraction = h / max(pot, 1.)
        node.risk_pot = pot + h
        return node
    def after_check():
        if opp <= 0 or me <= 0:
            return showdown(0,0)
        children, labels = [showdown(0,0)], ["check"]
        for bet in sorted(set(min(opp,x) for x in (.5*pot,pot,opp))):
            if bet <= 0: continue
            replies = [leaf("fold0",0,0),showdown(min(me,bet),bet)]
            names = ["fold","call"]
            if check_raise and opp > bet and me > bet:
                # Raise-to after calling: call B plus a pot-sized raise P+2B.
                # The second size is a stack jam. Only jam can be a short raise.
                for raise_to in sorted(set((min(me,pot+3*bet),me))):
                    if raise_to <= bet or (raise_to < 2*bet and raise_to != me):continue
                    reply = Node(1,[leaf("fold1",0,bet),showdown(raise_to,min(opp,raise_to))],labels=["fold","call"])
                    reply.facing_fraction = raise_to / max(pot + bet, 1.)
                    reply.risk_pot = pot + bet + raise_to
                    replies.append(reply)
                    names.append("check_raise_to_%g" % raise_to)
            children.append(Node(0,replies,labels=names))
            labels.append(bet)
        return Node(1,children,labels=labels)
    legal = list(legal if legal is not None else ((0,1,2,3,4) if owed else (1,2,3,4)))
    if opp <= 0:
        legal = [a for a in legal if a in (0,1)]
    children, labels = [], []
    for a in legal:
        if a == 0:
            if owed <= 0: continue
            child = leaf("fold0",0,0)
        elif a == 1:
            child = showdown(min(me,owed),0) if owed else showdown(0,0) if check_closes else after_check()
        else:
            h = float(amounts[a]) if amounts is not None and a in amounts else (me if a==4 else min(me,max(owed*2,.5*pot if a==2 else pot)))
            h = min(me,h)
            if h <= owed or opp <= 0: continue
            child = response(h)
        children.append(child)
        labels.append(a)
    if not children:
        raise ValueError("no legal root actions")
    return Node(0,children,labels=labels)


class PublicSolver:
    def __init__(self, snapshot, hero_range=None, opponent_range=None, weights=None,
                 iterations=None, chance_samples=2, seed=0, first_player=0,
                 amounts=None, utility=False, context=None, policy_only=False, root_prior=None, hero_at=None, check_raise=False,
                 context_prior=True):
        self.s = snapshot
        self.policy_only=bool(policy_only)
        self.board = tuple(sorted(_ev(c) for c in snapshot.board_cards))
        if utility and len(self.board)<5:
            raise ValueError("match utility requires terminal outcome distributions; expected chip CFVs cannot provide them")
        if (snapshot.big_blind<=0 or snapshot.pot<=0 or snapshot.to_call<0 or
            snapshot.to_call>snapshot.pot or min(snapshot.me_stack,snapshot.opp_stack)<0):
            raise ValueError("invalid public chip state")
        self.hands,self.rank,self.bins = _public(self.board)
        self.c1 = np.array([CARD_ID[h[0]] for h in self.hands])
        self.c2 = np.array([CARD_ID[h[1]] for h in self.hands])
        self.groups = _groups(self.rank)
        self.width = len(self.hands)
        range_audit = None
        if hero_range is None or opponent_range is None:
            from .search_belief import history_combo_ranges
            (default_h,default_o),range_audit=history_combo_ranges(snapshot,return_audit=True)
            if hero_range is None:hero_range=default_h
            if opponent_range is None:opponent_range=default_o
        self.rh = expand_range(hero_range,self.bins)
        self.ro = expand_range(opponent_range,self.bins)
        from .solver_prior import profile, condition_range
        from .search_belief import _percentiles
        self.match_context = context
        self.match_profile = profile(context, int(snapshot.street), snapshot) if context_prior else profile(None, int(snapshot.street), snapshot)
        self.match_strength = _percentiles(self.rank)
        self.ro = condition_range(snapshot, context, self.ro, self.match_strength, self.match_profile)
        self.weights = weights
        self.iterations = iterations or (RIVER_ITERS if len(self.board)==5 else TURN_ITERS)
        self.chance_samples,self.seed = int(chance_samples),int(seed)
        self.tree = betting_tree(float(snapshot.pot),float(snapshot.to_call),float(snapshot.me_stack),float(snapshot.opp_stack),
                                 getattr(snapshot,'legal_action_ids',None),amounts,first_player,
                                 check_closes=first_player==0 and float(snapshot.to_call)==0 and
                                     getattr(snapshot,'me_seat',0)!=getattr(snapshot,'postflop_first_seat',0),check_raise=check_raise)
        self.tree.init(self.width)
        self._init_match_prior(self.tree)
        if root_prior is not None:
            if first_player!=0 or hero_at is None:raise ValueError('blueprint prior requires the acting hero root information set')
            prior=np.asarray(root_prior,dtype=np.float64)
            if prior.shape!=(5,) or not np.isfinite(prior).all() or np.any(prior<0):raise ValueError('invalid root blueprint prior')
            projected=prior[np.asarray(self.tree.labels,dtype=np.int64)]
            if projected.sum()<=0:raise ValueError('empty executable blueprint prior')
            self.tree.strategy[int(hero_at)]=projected/projected.sum()
            # Only this known private information set gets a network prior.
            # Unknown private hands and continuation infosets remain uniform.
            self.tree.fallback=self.tree.strategy.copy()
        self.rootmass = (self.mass(self.ro),self.mass(self.rh))
        self.utility,self.context = utility,context
        self.trace = {"cfv_calls":0,"chance_branches":0,"leaf_states":[],"support":self.width,
                      "game":"check-bet-fold-call-jam-reraise-call", "schema":SCHEMA,"check_raise":bool(check_raise),"range_audit":range_audit}
        self.trace['match_prior'] = dict(self.match_profile, model='bounded-public-match-v1',
                                       response='restricted_nash_mixture', range='history_context_size_mixture')
        self._chance_cache = {}

    def _init_match_prior(self, node):
        if node.player is None:
            return
        if node.player == 1 and self.match_profile['eta'] > 0:
            from .solver_prior import response_prior, profile
            node.match_prior = response_prior(node.labels, self.match_strength, self.match_profile,
                                              self.match_context, getattr(node, 'facing_fraction', None))
            risk_state = SimpleNamespace(pot=getattr(node, 'risk_pot', self.s.pot),
                                         to_call=0., big_blind=self.s.big_blind)
            eta = min(self.match_profile['eta'], profile(self.match_context, int(self.s.street), risk_state)['eta'])
            node.match_eta = eta
            node.strategy = (1. - eta) * node.strategy + eta * node.match_prior
            node.fallback = node.strategy.copy()
        for child in node.children:
            self._init_match_prior(child)

    def mass(self,w):
        cards = np.bincount(self.c1,weights=w,minlength=52)+np.bincount(self.c2,weights=w,minlength=52)
        z = w.sum()-cards[self.c1]-cards[self.c2]+w
        z = np.maximum(z,0.)
        z[z<1e-14] = 0.
        return z

    def eq(self,w):
        return equity_vs(self.rank,self.c1,self.c2,w,*self.groups)

    def terminal(self,leaf,rh,ro):
        kind,h,o,p = leaf
        p0 = float(self.s.pot)
        blind = float(self.s.big_blind)
        if kind != "show":
            # Future investment by the folder is lost; winner's own unmatched
            # bet is refunded. Use the same centered baseline for both seats.
            value = (-h if kind=="fold0" else p0+o) / blind - p0/(2*blind)
            if self.utility:
                sunk=max(0.,float(getattr(self.s,'starting_stack',self.s.me_stack+p0/2))-self.s.me_stack)
                value += (p0/2-sunk)/blind
                value = float(_phi_blend(value,blind,getattr(self.context,'hero_net_bb',None),getattr(self.context,'hands_played',None)))
            return value*self.mass(ro), -value*self.mass(rh)
        if len(self.board) < 5:
            return self.chance(leaf,rh,ro)
        win = (p+o-p0/2)/blind
        loss = (-h-p0/2)/blind
        tie = .5*(win+loss)
        if self.utility:
            sunk=max(0.,float(getattr(self.s,'starting_stack',self.s.me_stack+p0/2))-self.s.me_stack)
            shift=(p0/2-sunk)/blind
            win,loss,tie = _phi_blend(np.array([win+shift,loss+shift,tie+shift]),blind,getattr(self.context,'hero_net_bb',None),getattr(self.context,'hands_played',None))
        # Obtain W/T/L mass from the preserved grouping kernel. equity alone
        # suffices for linear chips. Nonlinear match utility needs separate ties.
        def payoff(w):
            eq,z=self.eq(w)
            if not self.utility:
                return z*(loss+(win-loss)*eq)
            order,g0,g1=self.groups
            # Tie mass is equity kernel with all other strength groups zeroed.
            ws=w[order]; cs1=self.c1[order];cs2=self.c2[order]
            pref=np.r_[0.,np.cumsum(ws)]
            card=np.zeros((52,self.width+1))
            np.add.at(card,(cs1,np.arange(1,self.width+1)),ws)
            np.add.at(card,(cs2,np.arange(1,self.width+1)),ws)
            np.cumsum(card,axis=1,out=card)
            ties=pref[g1]-pref[g0]-(card[cs1,g1]-card[cs1,g0])-(card[cs2,g1]-card[cs2,g0])+ws
            tm=np.empty(self.width);tm[order]=np.maximum(ties,0.)
            wins=eq*z-.5*tm
            return wins*win+tm*tie+np.maximum(z-wins-tm,0.)*loss
        hv=payoff(ro)
        # Opponent sees the same terminal outcome with the opposite sign.
        # Linear loss/win constants swap; nonlinear utilities swap too.
        win,loss = -loss,-win
        tie = -tie
        ov=payoff(rh)
        return hv,ov

    def allin_chance(self,leaf,rh,ro):
        """All-in has no future betting: sample legal runouts, exact showdown."""
        need=5-len(self.board)
        pool=[c for c in DECK if c not in self.board]
        runs=list(itertools.combinations(pool,need))
        rng=np.random.RandomState(self.seed+123)
        if self.chance_samples>0 and self.chance_samples<len(runs):
            selected=[runs[i] for i in rng.choice(len(runs),self.chance_samples,replace=False)]
        else:selected=runs
        total_pairs=math.comb(52-len(self.board)-4,need) if hasattr(math,'comb') else math.factorial(52-len(self.board)-4)//math.factorial(need)//math.factorial(52-len(self.board)-4-need)
        factor=len(runs)/float(len(selected)*total_pairs)
        hv=np.zeros(self.width);ov=np.zeros(self.width)
        oldboard,oldrank,oldgroups=self.board,self.rank,self.groups
        try:
            for run in selected:
                valid=~(np.isin(self.c1,[CARD_ID[c] for c in run])|np.isin(self.c2,[CARD_ID[c] for c in run]))
                key=('allin',run)
                if key not in self._chance_cache:
                    rank=np.where(valid,pack_many(self.hands,oldboard+run),0)
                    self._chance_cache[key]=(rank,_groups(rank))
                self.rank,self.groups=self._chance_cache[key]
                self.board=oldboard+run
                a,b=self.terminal(leaf,rh*valid,ro*valid)
                hv+=factor*a*valid;ov+=factor*b*valid
        finally:self.board,self.rank,self.groups=oldboard,oldrank,oldgroups
        return hv,ov

    def chance(self,leaf,rh,ro):
        kind,h,o,p = leaf
        if float(self.s.me_stack)-h<=1e-9 or float(self.s.opp_stack)-o<=1e-9:
            return self.allin_chance(leaf,rh,ro)
        if self.weights is None:
            raise ValueError("non-river resolving requires validated CFV weights")
        leafpot = p+h+o
        ms=max(0.,float(self.s.me_stack)-h)
        os_=max(0.,float(self.s.opp_stack)-o)
        pool=[c for c in DECK if c not in self.board]
        # Public sampling excludes no private hand. Per-hand conflicts are
        # masked below; every legal ordered deal has the same inclusion rate.
        rng=np.random.RandomState(self.seed+len(self.board)*997)
        draws=pool if self.chance_samples<=0 or self.chance_samples>=len(pool) else [pool[i] for i in rng.choice(len(pool),self.chance_samples,replace=False)]
        hv=np.zeros(self.width);ov=np.zeros(self.width)
        factor=len(pool)/float(len(draws)*(52-len(self.board)-4))
        first=0 if getattr(self.s,'me_seat',0)==getattr(self.s,'postflop_first_seat',0) else 1
        for draw in draws:
            cachekey=draw
            if cachekey not in self._chance_cache:
                board=tuple(sorted(self.board+(draw,)))
                hands,rank,bins=_public(board)
                lookup=np.full(52*52,-1,dtype=np.int64)
                keys=np.array([CARD_ID[pair[0]]*52+CARD_ID[pair[1]] for pair in hands])
                lookup[keys]=np.arange(len(hands))
                idx=lookup[self.c1*52+self.c2]
                self._chance_cache[cachekey]=(board,bins,idx)
            board,bins,idx=self._chance_cache[cachekey]
            valid=idx>=0
            nh=np.zeros(len(bins));no=np.zeros(len(bins))
            nh[idx[valid]]=rh[valid];no[idx[valid]]=ro[valid]
            if nh.sum()<=0 or no.sum()<=0: continue
            bhr,bor=compress_range(nh,bins),compress_range(no,bins)
            street=len(board)-2
            prefix='cfv_river.' if len(board)==5 else 'cfv_turn.'
            # The net is evaluated for both player perspectives with the same
            # updated branch ranges and real post-action stacks/pot.
            f0=spot_features(board,bhr,bor,leafpot,0,ms,self.s.big_blind,street,os_,first)
            f1=spot_features(board,bor,bhr,leafpot,0,os_,self.s.big_blind,street,ms,1-first)
            pred=predict_residual_cfv(self.weights,prefix,np.stack([f0,f1]))
            self.trace['cfv_calls']+=2
            self.trace['chance_branches']+=1
            if len(self.trace['leaf_states'])<12:
                self.trace['leaf_states'].append(dict(board=board,pot=leafpot,hero_stack=ms,opp_stack=os_,H=h,O=o,hero_range=bhr.tolist(),opp_range=bor.tolist()))
            vh=np.zeros(self.width);vo=np.zeros(self.width)
            vh[valid]=pred[0,bins[idx[valid]]]
            vo[valid]=pred[1,bins[idx[valid]]]
            wh=np.where(valid,rh,0.);wo=np.where(valid,ro,0.)
            zh=self.mass(wo)*valid;zo=self.mass(wh)*valid
            # Enforce zero-sum expectation of the two approximate CFV queries.
            joint=float(np.dot(wh,zh))
            correction=(np.dot(wh,zh*vh)+np.dot(wo,zo*vo))/(2*joint) if joint>0 else 0.
            shift=(leafpot/2-h-float(self.s.pot)/2)/float(self.s.big_blind)
            hv+=factor*zh*(vh-correction+shift)
            ov+=factor*zo*(vo-correction-shift)
        return hv,ov

    def walk(self,node,rh,ro,step,update=True,average=False):
        if node.player is None:
            return self.terminal(node.leaf,rh,ro)
        strategy=node.strategy
        if average:
            den=node.total.sum(axis=1,keepdims=True)
            strategy=np.divide(node.total,den,out=np.full_like(node.total,1./len(node.children)),where=den>0)
        values=[]
        for a,child in enumerate(node.children):
            values.append(self.walk(child,rh*strategy[:,a] if node.player==0 else rh,
                                    ro*strategy[:,a] if node.player==1 else ro,step,update,average))
        own=np.stack([v[node.player] for v in values],axis=1)
        base=np.sum(strategy*own,axis=1)
        other=np.sum([v[1-node.player] for v in values],axis=0)
        if update:
            # Counterfactual child values include opponent reach, never own reach.
            # Average strategy includes the acting player's own realization reach.
            root=self.rh if node.player==0 else self.ro
            reach=rh if node.player==0 else ro
            ownreach=np.divide(reach,root,out=np.zeros_like(reach),where=root>0)
            node.total+=step**2*ownreach[:,None]*strategy
            pos=step**1.5/(step**1.5+1.)
            node.regret=np.where(node.regret>=0,node.regret*pos,node.regret*.5)+(own-base[:,None])
        return (base,other) if node.player==0 else (other,base)

    def update_strategies(self,node):
        if node.player is None:return
        positive=np.maximum(node.regret,0.)
        den=positive.sum(axis=1,keepdims=True)
        node.strategy=np.divide(positive,den,out=node.fallback.copy(),where=den>0)
        if hasattr(node, 'match_prior'):
            # Keep the empirical component throughout resolving. A warm start
            # alone disappears after the first regret-matching iteration.
            eta = node.match_eta
            baseline = np.divide(positive, den, out=np.full_like(positive, 1. / len(node.children)), where=den>0)
            node.strategy = (1. - eta) * baseline + eta * node.match_prior
        for child in node.children:self.update_strategies(child)

    def solve(self):
        for t in range(1,self.iterations+1):
            self.walk(self.tree,self.rh,self.ro,t)
            self.update_strategies(self.tree)
        hv,ov=self.walk(self.tree,self.rh,self.ro,self.iterations,False,True)
        self.state=np.divide(hv,self.rootmass[0],out=np.zeros(self.width),where=self.rootmass[0]>0)
        self.opp_state=np.divide(ov,self.rootmass[1],out=np.zeros(self.width),where=self.rootmass[1]>0)
        den=self.tree.total.sum(axis=1,keepdims=True)
        self.average=np.divide(self.tree.total,den,out=np.full_like(self.tree.total,1./len(self.tree.children)),where=den>0)
        if getattr(self,'policy_only',False):
            self.trace['zero_sum_residual']=float(np.dot(self.rh,hv)+np.dot(self.ro,ov))
            return self
        self.action_values=np.zeros((self.width,len(self.tree.children)))
        for a,child in enumerate(self.tree.children):
            values=self.walk(child,self.rh*self.average[:,a] if self.tree.player==0 else self.rh,self.ro,1,False,True)[0]
            self.action_values[:,a]=np.divide(values,self.rootmass[0],out=np.zeros(self.width),where=self.rootmass[0]>0)
        bet_nodes=[(a,n) for a,n in zip(self.tree.labels,self.tree.children) if a in (2,3,4) and n.player==1]
        self.bet_response_actions=[a for a,n in bet_nodes]
        self.opponent_response=np.zeros((len(bet_nodes),self.width,3))
        self.opponent_response_mask=np.zeros((len(bet_nodes),3))
        for i,(a,n) in enumerate(bet_nodes):
            den=n.total.sum(axis=1,keepdims=True)
            av=np.divide(n.total,den,out=np.full_like(n.total,1./len(n.children)),where=den>0)
            self.opponent_response[i,:,:av.shape[1]]=av
            self.opponent_response_mask[i,:av.shape[1]]=1
        self.trace['zero_sum_residual']=float(np.dot(self.rh,hv)+np.dot(self.ro,ov))
        return self


def solve_public_subgame(snapshot, hero_range=None, opponent_range=None, **kwargs):
    return PublicSolver(snapshot,hero_range,opponent_range,**kwargs).solve()


def solve_river(snapshot, context=None, hero_range=None, opponent_range=None, amounts=None, **kwargs):
    sol=solve_public_subgame(snapshot,hero_range,opponent_range,amounts=amounts,context=context,**kwargs)
    at=next(i for i,h in enumerate(sol.hands) if frozenset(h)==frozenset(_ev(c) for c in snapshot.hole_cards))
    out=np.full(5,-1e9)
    for i,a in enumerate(sol.tree.labels):
        out[a]=sol.action_values[at,i]+snapshot.pot/(2*snapshot.big_blind)
    solve_river.last_state=sol.state
    solve_river.last_hands=sol.hands
    solve_river.last_rank=sol.rank
    solve_river.last_solution=sol
    return out,at,sol.width
