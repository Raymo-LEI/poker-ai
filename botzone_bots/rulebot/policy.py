"""Rule-bot policy, copied so the zip does not import training code.

The numbers and the priority order are ``src/alphanl/botzone/rule_bot.py``.
The five responses are packed the way ``actions.py`` packs them: fold -1,
check/call 0, min raise = ``safe``, large raise = ``max(safe * 2, pot)``,
all-in -2. ``safe`` is ``legal.py``'s judge minimum
``max(1, bound_b - own)``, where ``bound_b`` stores ``2 * round_raise``.
The priority order is unchanged, so the chosen id matches the source
rule bot and the packed integer matches ``response_for_action``.

The evaluator is the local five-card ranker ``table._local_rank`` uses.
No numpy and no torch.
"""

import itertools

FOLD_ACTION = 0
CHECK_CALL_ACTION = 1
RAISE_MIN_ACTION = 2
RAISE_LARGE_ACTION = 3
ALL_IN_ACTION = 4

HIGH_AT = 7
MID_AT = 4
LOW_AT = 2

HIGH_PRIORITY = (
    ALL_IN_ACTION,
    RAISE_LARGE_ACTION,
    RAISE_MIN_ACTION,
    CHECK_CALL_ACTION,
    FOLD_ACTION,
)
MID_PRIORITY = (
    RAISE_MIN_ACTION,
    RAISE_LARGE_ACTION,
    CHECK_CALL_ACTION,
    ALL_IN_ACTION,
    FOLD_ACTION,
)
LOW_PRIORITY = (
    CHECK_CALL_ACTION,
    RAISE_MIN_ACTION,
    FOLD_ACTION,
    RAISE_LARGE_ACTION,
    ALL_IN_ACTION,
)
REST_CHECK_PRIORITY = (
    CHECK_CALL_ACTION,
    FOLD_ACTION,
    RAISE_MIN_ACTION,
    RAISE_LARGE_ACTION,
    ALL_IN_ACTION,
)
REST_FACING_PRIORITY = (
    FOLD_ACTION,
    CHECK_CALL_ACTION,
    RAISE_MIN_ACTION,
    RAISE_LARGE_ACTION,
    ALL_IN_ACTION,
)


def preflop_strength(hole):
    """``rule_bot.preflop_strength`` on wiki card ids."""

    if len(hole) != 2:
        raise ValueError("hole must be two cards")
    r1 = hole[0] // 4
    r2 = hole[1] // 4
    high = r1 if r1 > r2 else r2
    low = r2 if r1 > r2 else r1
    suited = hole[0] % 4 == hole[1] % 4
    if r1 == r2:
        return 4 + (high // 3)
    if high >= 12 and low >= 9:
        return 6
    if high >= 12:
        return 5
    if suited and high - low <= 2 and low >= 4:
        return 4
    if high >= 10:
        return 3
    return 1


def postflop_strength(hole, board):
    """Made-hand category on the 1..9 scale. Best five cards only."""

    if len(hole) != 2 or len(board) < 3:
        raise ValueError("postflop needs two hole cards and a board")
    cards = tuple(hole) + tuple(board)
    ranked = _local_rank(cards)
    return int(ranked[0]) + 1


def hand_strength(hole, board):
    if len(board) == 0:
        return preflop_strength(hole)
    return postflop_strength(hole, board)


def action_priority(strength, to_call_chips):
    if strength >= HIGH_AT:
        return HIGH_PRIORITY
    if strength >= MID_AT:
        return MID_PRIORITY
    if strength >= LOW_AT:
        return LOW_PRIORITY
    if to_call_chips <= 0:
        return REST_CHECK_PRIORITY
    return REST_FACING_PRIORITY


def choose_action(legal, strength, to_call_chips):
    allowed = set()
    for action in legal:
        allowed.add(int(action))
    if not allowed:
        raise ValueError("no legal action")
    for action in action_priority(strength, to_call_chips):
        if action in allowed:
            return action
    raise ValueError("priority missed every legal id")


def _owed(view):
    return max(view.p_commit) - view.p_commit[view.my_id]


def safe_extra(view):
    """``legal.safe_extra``: ``max(1, bound_b - own)``.

    ``bound_b`` is ``2 * round_raise``. The judge accepts a positive extra
    when ``extra + own >= 2 * round_raise`` and ``extra`` is strictly less
    than the stack (``botzone_judge.py`` lines 299-303). Preflop, the small
    blind facing the big blind therefore needs 150 (``2 * 100 - 50``). The
    old ``max(bound A, bound B)`` value of 200 is still accepted, and is
    stricter than this minimum. ``decide.map_target`` is unchanged and
    still drops anything the judge would reject.
    """

    if any(view.p_all_in):
        return None
    extra = view.bound_b - view.p_commit[view.my_id]
    if extra < 1:
        extra = 1
    remaining = view.chips[view.my_id]
    if extra < remaining:
        return extra
    return None


def _allows(view, response):
    remaining = view.chips[view.my_id]
    if remaining <= 0:
        return False
    if response == -1 or response == -2:
        return True
    if any(view.p_all_in):
        return False
    if response == 0:
        return remaining > _owed(view)
    if isinstance(response, bool) or not isinstance(response, int):
        return False
    if response <= 0:
        return False
    safe = safe_extra(view)
    if safe is None:
        return False
    return safe <= response < remaining


def action_responses(view):
    safe = safe_extra(view)
    if safe is None:
        large = None
    else:
        large = safe * 2
        if view.pot > large:
            large = view.pot
    candidates = (-1, 0, safe, large, -2)
    packed = []
    for response in candidates:
        if response is not None and _allows(view, response):
            packed.append(response)
        else:
            packed.append(None)
    return tuple(packed)


def legal_action_ids(view):
    ids = []
    for action_id, response in enumerate(action_responses(view)):
        if response is not None:
            ids.append(action_id)
    return tuple(ids)


def policy_response(view, request=None):
    """Integer the rule bot would send for this view.

    ``request`` is accepted so callers can pass the raw request; the
    cards and the mask already live on ``view``.
    """

    del request
    strength = hand_strength(view.hole, view.board)
    action_id = choose_action(legal_action_ids(view), strength, _owed(view))
    return action_responses(view)[action_id]


def _local_rank(cards):
    best = None
    for combo in itertools.combinations(cards, 5):
        ranked = _score_five(combo)
        if best is None or ranked > best:
            best = ranked
    if best is None:
        raise ValueError("need at least five cards")
    return best


def _score_five(cards):
    ranks = [card // 4 + 2 for card in cards]
    flush = len(set(card % 4 for card in cards)) == 1
    straight, top = _straight_high(ranks)
    counts = sorted(
        ((ranks.count(rank), rank) for rank in set(ranks)),
        reverse=True,
    )
    if straight and flush:
        return (8, top)
    if counts[0][0] == 4:
        return (7, counts[0][1], counts[1][1])
    if counts[0][0] == 3 and counts[1][0] == 2:
        return (6, counts[0][1], counts[1][1])
    ordered = tuple(sorted(ranks, reverse=True))
    if flush:
        return (5, ordered)
    if straight:
        return (4, top)
    if counts[0][0] == 3:
        kickers = tuple(rank for rank in ordered if rank != counts[0][1])
        return (3, counts[0][1], kickers)
    if counts[0][0] == 2 and len(counts) > 1 and counts[1][0] == 2:
        high, low = counts[0][1], counts[1][1]
        kicker = next(rank for rank in ordered if rank != high and rank != low)
        return (2, high, low, kicker)
    if counts[0][0] == 2:
        kickers = tuple(rank for rank in ordered if rank != counts[0][1])
        return (1, counts[0][1], kickers)
    return (0, ordered)


def _straight_high(ranks):
    unique = sorted(set(ranks))
    if len(unique) != 5:
        return False, 0
    if unique[-1] - unique[0] == 4:
        return True, unique[-1]
    if unique == [2, 3, 4, 5, 14]:
        return True, 5
    return False, 0
