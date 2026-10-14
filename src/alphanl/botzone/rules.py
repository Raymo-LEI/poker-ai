"""Botzone TexasHoldem2p betting rules: the single source of truth.

Used by ``alphanl.transformer.botzone_table.BotzoneTable`` (training env,
``env.rules = "botzone"``) and by ``alphanl.botzone.adapter`` (the bot that
answers Botzone requests). Verified against 41 public Botzone matches
(2050 hands, 7785 decisions): see tests/test_botzone_rules.py and
tests/fixtures/botzone_real.jsonl.gz.

Response ints: -1 fold, -2 all-in, 0 check/call, >0 raise = EXTRA chips put in
by this action (new street bet = old street bet + amount).
"""

from __future__ import annotations

FOLD = -1
ALLIN = -2
CALL = 0


def min_raise_amount(round_bet: int, my_bet: int, last_raise: int, big_blind: int) -> int:
    """Smallest raise (EXTRA chips) that is legal under every rule consistent with real logs.

    All 1246 accepted raises in the logs satisfy BOTH
      (A) new street bet >= 2 * street max bet            (wiki wording)
      (B) raise over max bet >= last raise action amount   (judge ``round_raise``)
    and every postflop opening bet was >= BB. The sample bot's rule (>= 2x the
    last raise action, 200 preflop) is refuted: SB raises of 150 (to 200) were
    accepted. No rejected raise was observed, so this is a conservative lower
    bound: max(A, B, BB).

    ``last_raise`` is the previous raise action amount on this street, or BB
    preflop / SB postflop before any raise (judge display values).
    """
    to_call = max(0, int(round_bet) - int(my_bet))
    return max(2 * int(round_bet) - int(my_bet), to_call + int(last_raise), int(big_blind))


def classify_response(
    response,
    *,
    round_bet: int,
    my_bet: int,
    chips_behind: int,
    last_raise: int,
    big_blind: int,
    allin_locked: bool,
) -> str:
    """'fold' | 'allin' | 'check' | 'call' | 'raise' | 'illegal' (Botzone: illegal == fold)."""
    if isinstance(response, bool) or not isinstance(response, int):
        return "illegal"
    r = int(response)
    if r == FOLD:
        return "fold"
    if r == ALLIN:
        return "allin"
    if r < -2 or allin_locked:
        return "illegal"  # after an all-in only fold / all-in
    to_call = max(0, int(round_bet) - int(my_bet))
    if to_call >= chips_behind:
        return "illegal"  # must have MORE chips than the call to call or raise
    if r == CALL:
        return "check" if to_call == 0 else "call"
    if r >= chips_behind:
        return "illegal"  # raise must be strictly below remaining chips (else send -2)
    if r < min_raise_amount(round_bet, my_bet, last_raise, big_blind):
        return "illegal"
    return "raise"
