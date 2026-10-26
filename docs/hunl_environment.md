# Standard HUNL 200BB environment

The CFR path uses `alphanl.transformer.hunl.HunlTable`, a PokerKit-backed
table with fixed 50/100 blinds and 20,000 starting chips per seat per hand.
Seat 0 posts BB; seat 1 is SB/button. SB acts first preflop. BB opens flop,
turn, and river. `payoffs()` returns chip deltas, so divide by 100 for BB.

The existing five action ids, sequence, 275-token vocabulary, and four
numeric channels remain. Half-pot and pot actions retain the existing
PokerKit abstraction: add half/current pot, subject to legal minimum raises.
They are not a new raise-size formula. Illegal actions remain masked.

`step_raise_to(amount)` accepts an arbitrary legal **street raise-to** and
records the actual chip delta, pot, stack, and call amount in `ActionEvent`.
The event has no five-action id; its semantic type is inferred from the
actual action. HUNL history tokens use the nearest existing positive pot
ratio representative (0.25, 0.5, 0.75, 1, 1.5, 2, 4, 8), with ties assigned
to the smaller ratio. Bucket 0 remains for zero-chip actions.

`HunlTokenizer` uses the same three logarithmic chip channels as the existing
tokenizer; the fraction channel is **actual fraction / 10 without clipping**.
This preserves arbitrary off-tree ratios, including all-ins above 10x pot.
Its feature version is `hunl_exact_fraction_v1`, which CFR checkpoints must
record. The original `BettingHistoryTokenizer` still clips this channel and
all existing PPO encodings remain unchanged. Although tensor shapes match,
the two feature conventions are explicit and should not be mixed silently.

`clone()` independently copies state, deck, history, and RNG. A complete
shuffled deck is sampled at `reset`; all action branches of that traversal
share this chance sample. Opponent and traverser observations still expose
only their own two hole cards and the currently dealt public board.

Install `requirements-cfr.txt` for the `phevaluator==0.6.0` perfect-hash
lookup evaluator. The standard CFR table requires it by default. PokerKit
settlement receives a local comparable hand adapter that calls the 5/6/7
card lookup directly, avoiding its combinatorial five-card Python loop.
`fast_evaluator=False` is available only as a diagnostic reference. The
adapter is isolated to these tables, so legacy PPO evaluator behavior is
unchanged.

## Existing rules audit

`RlcardTable` wraps rlcard 1.1.0, whose heads-up implementation acts SB first
on every street. It also has legacy pot bookkeeping and 1/2 blind defaults.
It is kept for old checkpoints and PPO regression tests, and the CFR path
does not permit selecting it.

The repository's `BotzoneEnvTable` mirrors the checked-in **simple** judge
`tests/fixtures/botzone_judge.py`. That judge starts each postflop street at
`dealer_idx` and then calls `_next_player()`, putting the SB first again.
Its `dealer` label denotes the BB, and its minimum positive-extra rule also
differs from standard no-limit rules. This historical adapter and judge
fixtures remain intact. Therefore the new standard HUNL environment follows
the requested Phase 1 rules but **does not claim equivalence to that simple
judge**. Deployment must select/verify the intended competition rules.

Regression coverage in `tests/test_hunl_environment.py` checks every street's
order, exact off-tree observations, independent clones, checkpoint feature
and rule tags, lookup ranking/settlement parity, all-ins, ties, minimum raises,
and 40,000-chip conservation.
