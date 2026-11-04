# Terminal MatchCtx reconstruction and online RNR

Botzone can omit the opponent's final Fold or Call from the next request. The
runtime now persists a public contribution ledger after each executable hero
action and reconciles it against completed-hand `total_win_chips` at the next
observed hand. The frozen Transformer, `max_seq_len=64`, and the existing 14-column
`PRO_CONTEXT_FIELDS` interface are unchanged. Legacy 12-column models remain
supported as well.

## Exact public evidence

The checked-in official judge (`tests/fixtures/botzone_judge.py`) updates
`total_win_chips` after settlement, then deals and posts new blinds. Therefore
the cumulative score excludes current-hand blinds; `my_chips` includes their
deduction and must not be substituted for the score. An executable referee test
verifies this ordering and its amounts.

The persisted ledger includes `last_hand_idx`, contributions before/after the
hero action, the opposing contribution at that action, aggression, street,
remaining stacks, action kind, and whether a river check closes the street.

- A terminal gain equal to the opponent's pre-response contribution proves a
  Fold to a positive unmatched bet. Its synthetic event goes through the same
  HUD accounting as an observed Fold, including preflop/3-bet opportunities,
  postflop responses, aggression and trap denominators.
- A matching terminal Call is reconstructed only when the river or an all-in
  closes the hand and the payoff agrees with effective matched contributions.
  A short Call refunds uncalled excess. A matching showdown payoff within one
  chip of zero is recorded as a split with opponent win score `0.5`.
- A hero river Call or a closing in-position Check records showdown without
  inventing another opposing Call. An out-of-position Check alone proves no
  showdown. Observed terminal events are never reconstructed again.
- Every skipped hand is checked against alternating seats. An entirely
  unqueried hero BB hand proves an opposing SB first-action Fold and `+50`.
  That gain is removed before settling the last observed hand. Any skipped
  hero SB turn makes the multi-hand attribution unknown. With two alternating
  seats, at most one intervening hand can satisfy this proof.
- An inconsistent payoff is `unknown`. All staged profile counters from that
  hand, including sizing and deployment statistics, are restored to their
  beginning-of-hand values. Public score and hand progress still advance.
- At either seat's cumulative score reaching 90% of its remaining all-fold
  blind budget, regular frequencies freeze for the rest of the match. The
  authoritative score and terminal diagnostics continue. New matches reset
  the freeze.

The final hand requires a subsequent settlement message to update the HUD;
the decision protocol supplies no next-hand request after hand 49. This has no
effect on another in-match decision.

## Size fingerprints and solver conditioning

Additional serialized fields carry SB open extra-chip mean and variance in BB,
postflop small (`<=0.4P`) and overbet (`>1P`) counts, and Fold/Call/Raise response
counts by street and size bucket. No extra columns enter `context_encoder`.

Online `wizard -> evening_strategy -> deep_solver_v2` uses the repaired public
frequencies in every opposing-node strategy update:

```
sigma_opp = (1 - eta_eff) * regret_matching_strategy + eta_eff * empirical_prior
eta_sample = min(0.35, hands_played / 40)
eta_eff = eta_sample * exp(-pot_risk_bb / risk_scale_bb)
```

The empirical prior uses Beta-smoothed Fold/aggression rates, conditional street
and size counts, and public hand-strength percentiles. It is projected onto the
node's available actions. The mixture persists after every regret update, rather
than disappearing after initialization. No observed opportunities means no
empirical mixture.

Risk scale is 20BB, reduced to `max(2, min(20, abs(hero_net_bb)/2))` when there is
a nonzero lead. Risk is the larger of current pot and call amount. Opponent
response nodes also include the hypothetical hero investment in their pot risk.
At 40BB or more, the empirical weight is exactly zero. This prevents a small
root pot from retaining its exploit weight after a large hypothetical bet.

The range prior uses a positive, normalized context-and-size likelihood on full
board-legal combo support. It blends with the existing public-history posterior;
with a uniform baseline the formula is exactly `(1-eta)*uniform + eta*context_size`.
Existing current-hand evidence is retained. Neither actual hero hole cards nor
revealed opponent cards determine the public prior.

Offline teacher calls explicitly disable match priors. Large online river pots
use the existing terminal W/T/L match-utility surrogate; earlier streets retain
chip CFVs and the existing guaranteed-win blind lock. Neither finite-tree DCFR
nor that surrogate is a claim of full-game exact GTO or calibrated match equity.

## Validation

Run with the project's Python environment:

```sh
python -m pytest tests/test_match_context_terminal.py tests/test_botzone_stage2.py \
  tests/test_stage2_botzone_native.py tests/test_stage2_pro_hunter.py \
  tests/test_evening_v2.py tests/test_evening_production.py \
  tests/test_overnight_production.py tests/test_search_entry.py tests/test_wizard_solver.py -q -ra
```

Result: **109 passed, 2 skipped**. The skips require Linux memory-limit/taskset
acceptance and a Python 3.6 interpreter. The new file has 29 cases covering ten
hidden folds, cumulative/data parity, repeated requests, preflop opportunities,
all-in Calls, wins/losses/splits, odd-chip tolerance, Check-behind, observed-event
deduplication, unknown rollback, skipped hands, either-seat lock freeze, legacy
state, size statistics, every-iteration RNR, risk decay, public range support,
offline bypass, and the actual exported 64-token/14-context online path.

In the deterministic 2BB river test, ten hidden folds raise the weakest 30% of
holdings' average betting frequency from 33.57% to 44.99% after 24 iterations.
This verifies a persistent steal adjustment, not a demonstrated online win rate.

Local warm runtime timing used the unchanged 192-wide, four-layer
`u250_ladder_slayer.pt` backbone, 64 tokens and 14 context inputs, with identical
temporary random CFV heads for both source revisions. This is a latency fixture,
not a trained-head strength evaluation. Each revision's temporary fixture was
bound to its own solver source hash; no production artifact was rebound.

On macOS, Python 3.11.17 / NumPy 2.4.6, single-thread numerical libraries, 64
randomized requests across all streets:

- Main-source baseline: mean 20.59ms, p99 36.77ms.
- New source: mean 22.07ms, p99 34.24ms.
- New source with ten reconstructed folds: mean 20.64ms, p99 32.87ms, max 33.14ms.

These local warm samples remain below 65ms. Cold process startup and the actual
Botzone Python 3.6 / NumPy 1.19 runtime were not certified here.

The checked-in `stage2.zip` is the reproducible development source artifact,
not an allocated upload release. Source-hash checks still reject older CFV
artifacts against a changed solver. A production release requires compatible
validated CFVs, target-runtime checks, and a new increasing Vn delivery under
AGENTS.md. No model/checkpoint, training state, or deployed Botzone package was
changed by this PR.
