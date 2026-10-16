"""Rebuild one TexasHoldem2p decision from the latest judge request.

Independent of ``src/alphanl/botzone/legal.py``. Chip accounting follows
``tests/fixtures/botzone_judge.py`` (``Holdem.deal_cards_and_blind`` and
``Holdem.player_action``), not the wiki sample's ``round_raise``.

Blinds are not in ``history``. They are prefilled: the non-dealer posts 50,
the dealer posts 100, and ``round_raise`` is 100 (the larger blind). Each
later street resets ``round_raise`` to ``big_blind // 2`` which is 50.
A positive history action is the extra chips just put in, and it updates
``round_raise`` to ``max(round_raise, extra)``. Call and check are stored
as 0; the chips come from ``round_bet - own``. All-in is stored as -2 and
does not change ``round_raise``; the chips are whatever that seat still
held (both seats started at 20000). A fold that ends the hand has no
``round`` key. In two-player play every fold ends the hand.

Street changes are read from a history row's ``round`` when it has one,
and from ``len(public_cards)`` after the rows are applied. The non-dealer
is first to act on every street; this module does not choose the actor.
The request is already addressed to the seat that must act (``my_id``).
"""

STARTING_STACK = 20000
SMALL_BLIND = 50
BIG_BLIND = 100
_BOARD_STREET = {0: 0, 3: 1, 4: 2, 5: 3}


class HandView(object):
    """Judge betting fields plus the rule-bot mask tracker.

    ``status`` is the judge's ``round_player_bet``: a non-negative street
    total, or -1 / -2. ``street_put`` is the chips actually invested on
    this street, including a derived all-in. ``p_commit`` / ``bound_b`` /
    ``p_all_in`` follow ``legal.py`` so the unchanged rule-bot mask can
    be evaluated here without importing it. ``bound_b`` is not the
    judge's ``round_raise``.
    """

    def __init__(self):
        self.chips = [0, 0]
        self.street_put = [0, 0]
        self.status = [0, 0]
        self.round_bet = 0
        self.round_raise = 0
        self.pot = 0
        self.street = 0
        self.dealer = 0
        self.my_id = 0
        self.hole = []
        self.board = []
        self.p_commit = [0, 0]
        self.p_all_in = [False, False]
        self.p_folded = [False, False]
        self.bound_b = 0

    @property
    def opp_id(self):
        return 1 - self.my_id

    @property
    def opp_street(self):
        return self.street_put[self.opp_id]


def min_raise_extra(view):
    """Smallest positive extra ``player_action`` accepts, or None.

    The judge's raise test is ``extra + own >= 2 * round_raise`` and
    ``chips > extra``. All-in (``round_bet < 0``) leaves no positive bet.
    """

    if view.round_bet < 0:
        return None
    own = view.status[view.my_id]
    if own < 0:
        return None
    chips = view.chips[view.my_id]
    if chips <= 1:
        return None
    need = view.round_raise * 2 - own
    if need < 1:
        need = 1
    if need >= chips:
        return None
    return need


def judge_accepts(view, action):
    """True when ``Holdem.player_action`` would accept ``action``."""

    if isinstance(action, bool) or not isinstance(action, int):
        return False
    if action == -1 or action == -2:
        return True
    own = view.status[view.my_id]
    chips = view.chips[view.my_id]
    if view.round_bet < 0 or own < 0:
        return False
    if action == 0:
        inc = view.round_bet - own
        return chips > inc
    if action > 0 and action + own >= view.round_raise * 2 and chips > action:
        return True
    return False


def check_possible(view):
    """True when response 0 is a check (nothing owed), not a call."""

    if view.round_bet < 0:
        return False
    own = view.status[view.my_id]
    if own < 0:
        return False
    if view.round_bet != own:
        return False
    return view.chips[view.my_id] > 0


def reconstruct(request):
    """Return a ``HandView`` for this request. Raises on a broken request."""

    dealer = request["dealer_id"]
    my_id = request["my_id"]
    if dealer not in (0, 1) or my_id not in (0, 1):
        raise ValueError("dealer_id and my_id must be 0 or 1")
    board = list(request["public_cards"])
    hole = list(request["my_cards"])
    if len(board) not in _BOARD_STREET:
        raise ValueError("public_cards length is not 0, 3, 4, or 5")
    view = HandView()
    view.dealer = dealer
    view.my_id = my_id
    view.hole = hole
    view.board = board
    view.chips = [STARTING_STACK, STARTING_STACK]
    sb = (dealer + 1) % 2
    bb = dealer
    view.chips[sb] -= SMALL_BLIND
    view.chips[bb] -= BIG_BLIND
    view.street_put[sb] = SMALL_BLIND
    view.street_put[bb] = BIG_BLIND
    view.status[sb] = SMALL_BLIND
    view.status[bb] = BIG_BLIND
    view.p_commit[sb] = SMALL_BLIND
    view.p_commit[bb] = BIG_BLIND
    view.pot = SMALL_BLIND + BIG_BLIND
    view.round_bet = BIG_BLIND
    view.round_raise = BIG_BLIND
    # legal.new_hand starts bound_b at 2 * big blind. Blinds are not actions.
    view.bound_b = 2 * BIG_BLIND
    view.street = 0
    for entry in request["history"]:
        _apply_history(view, entry)
    target = _BOARD_STREET[len(board)]
    while view.street < target:
        _advance_street(view)
    if view.chips[0] + view.chips[1] + view.pot != STARTING_STACK * 2:
        raise ValueError("chip invariant broken")
    return view


def _advance_street(view):
    if view.street >= 3:
        raise ValueError("street past river")
    view.street += 1
    view.round_bet = 0
    view.round_raise = BIG_BLIND // 2
    view.bound_b = BIG_BLIND
    view.p_commit = [0, 0]
    view.street_put = [0, 0]
    for seat in (0, 1):
        if view.status[seat] >= 0:
            view.status[seat] = 0


def _apply_history(view, entry):
    action = _coerce_action(entry["action"])
    pid = entry["player_id"]
    if pid not in (0, 1):
        raise ValueError("player_id must be 0 or 1")
    kind = entry.get("action_type")
    is_fold = action == -1 or kind == "fold"
    if "round" in entry:
        rnd = entry["round"]
        if rnd < view.street:
            raise ValueError("history round went backwards")
        while view.street < rnd:
            _advance_street(view)
    elif not is_fold:
        raise ValueError("non-fold history row has no round")
    if is_fold:
        view.status[pid] = -1
        view.p_folded[pid] = True
        return
    if action == -2 or kind == "allin":
        _put(view, pid, view.chips[pid], kind="allin")
        return
    if action == 0 or kind in ("call", "check"):
        if view.round_bet < 0 or view.status[pid] < 0:
            raise ValueError("call is not legal in this history")
        _put(view, pid, view.round_bet - view.status[pid], kind="call")
        return
    if action <= 0:
        raise ValueError("unrecognized action")
    _put(view, pid, action, kind="raise")


def _put(view, pid, amount, kind):
    if amount < 0 or amount > view.chips[pid]:
        raise ValueError("history spends chips the seat does not have")
    prev_max = view.p_commit[0]
    if view.p_commit[1] > prev_max:
        prev_max = view.p_commit[1]
    added = view.p_commit[pid] + amount
    if added > prev_max:
        doubled = 2 * amount
        if doubled > view.bound_b:
            view.bound_b = doubled
    view.p_commit[pid] = added
    view.chips[pid] -= amount
    view.pot += amount
    view.street_put[pid] += amount
    if view.chips[pid] == 0:
        view.p_all_in[pid] = True
    if kind == "allin":
        view.status[pid] = -2
        view.round_bet = -2
        view.p_all_in[pid] = True
        return
    if kind == "call":
        view.status[pid] = view.round_bet
        return
    view.round_raise = view.round_raise if view.round_raise > amount else amount
    view.status[pid] = view.status[pid] + amount
    if view.status[pid] > view.round_bet:
        view.round_bet = view.status[pid]


def _coerce_action(action):
    if isinstance(action, bool) or action is None:
        raise ValueError("action is not an integer")
    if isinstance(action, int):
        return action
    if isinstance(action, float) and action.is_integer():
        return int(action)
    raise ValueError("action is not an integer")
