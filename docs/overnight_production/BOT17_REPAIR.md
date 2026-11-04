# V22 bot17 audit repairs

The supplied audit identifies the actually deployed bot as V22,
`release=V22_SEARCH_LOCK_20261104`, with V14-equivalent weights and legacy
search/guard ordering. The uniform-range Evening result was an offline
counterfactual in that audit, not evidence that this new overnight branch had
already played those matches. These changes repair the overnight branch.
The audited matches and old artifacts are unchanged.

## Public history conditioning

`search_belief.history_combo_ranges()` owns the declared
`public-combo-size-bayes-v1` observation model. Both `search_policy()` and a
direct default `PublicSolver`/`solve_river()` call now use complete board-legal
combo weights: 1,081 on the river, 1,128 on the turn and 1,176 on the flop.
Explicit offline ranges remain supported, including real neural PBS ranges
and Dirichlet exploration inputs; they are not double-conditioned.

Each player's action likelihood updates that player's public range. Preflop
sizes condition on the embedded preflop equity table, with stronger weighting
for >=3.3BB opens. Postflop likelihoods use tied made-hand percentiles on the
board prefix actually visible at the historical action. Future board cards
are used only as current public blockers, never to grade an earlier action.
Small free probes retain broad air mass. Overbets concentrate value smoothly;
a >50BB late-street aggression following a previous raise is more polar.
All-in calls are calls, never raises. Bluff likelihoods remain positive.

The actual Hero hole cards never enter this public-range construction. The
32-bucket CFV input remains unchanged at continuation leaves, but the root
posterior is no longer compressed/re-expanded through buckets. Range audit
fields record model, support size, action counts, entropy and peak mass.
This is an explicit heuristic observation model, not a calibrated inference
of bot17's private strategy or a claim that large bets cannot be bluffs.

## Search output and blind protection

`apply_guard(..., search_active=True)` bypasses every empirical postflop rule.
It validates probabilities and the executable domain. On a giant river
all-in with a previous postflop raise, it may move Call mass to Fold only when
exact showdown equity against the contracted public posterior is below the
actual call price, refunding unmatched chips before pricing a short call.
It never moves Fold to Call. Real `Agent.policy()` uses this
route for both full/slim labels, including an own guaranteed-win search lock.
Optional legacy safety and opponent-lock transfers cannot run afterward.

For first BB responses to large SB opens, `good_hand_blind_side()` rejects
>=3.2BB total opens / >=2.2BB calls. For enabled deployment guards,
`apply_guard()` preserves the complete legal input distribution before the
weak-reraise/pot-control/pruning chain. Small-open behavior remains available.
All-in opens retain their separate existing preflop shove safety path.
The independent frozen preflop weight route is unchanged.

## Audited public-request regressions

`tests/fixtures/overnight_bot17_audit.json` contains decision-time Hero requests
extracted from the cached complete match logs, with no opponent hole cards.
Tests explicitly supply the audit's original five-action probabilities when
checking their transformation; they do not imply that the overnight champion
has the deployed V22 model's exact logits.

- Match 2, Hand 7 flop: input Fold 85.3% remains 85.3%; no forced Call.
- Match 2, Hand 22 flop: the approximately 84.8% Fold input remains unchanged
  after normalization; no forced Call.
- Match 2, Hand 36 preflop: the approximately 48.5% Fold input remains intact
  against +291 added / 3.41BB total. The 219/220-chip boundary is tested.
- Match 2, Hand 9 river: with the audited blueprint prior and 16 online river
  iterations, Call is approximately 58.7% and Fold 18.9%. The final guard
  preserves the exact computed blueprint/DCFR mixture. At 40 offline
  iterations the pure solver Call is approximately 75.4%. The remaining mass
  includes legal raises; the repair does not force a hindsight-based Call.
- Match 3, Hand 17 river all-in: the explicit uniform-range control still
  calls >95%. Default full-combo history conditioning yields <1% pure-DCFR
  Call (approximately 0.0023% at 40 iterations). The safety fuse moves remaining
  mixed Call mass to Fold because posterior model equity is below 40.28% price.
  Weighted zero-sum error, positive bluff support and Hero-hole independence
  are also checked. The posterior equity is a model estimate, not hindsight.

No opponent's revealed cards or match/hand identifiers are used by inference.
The cases are development regressions; final match holdout seeds remain
separate from these known examples.

## Audited size replica and teacher pool

The existing five-opponent tournament keeps the name `Bot17_Probe3Bet` and
uses its new `bot17_public_size_v2` behavior in both judge evaluation and native
teacher collection. SB thresholds are Fold below .38, limp below .43, graded
+210..233 / +235..285 raises, and +291 cap at >=.586. BB versus a limp checks
below .50 and grades +220..292 above it. Chip targets are **additional chips**,
correctly converted to `PotFractionBet`'s call-plus-fraction convention.

Weak probes are exactly 100 into a 200 pot and .25..34 pot in larger pots.
Late strong made hands produce >1.5-pot pressure; full houses can river jam.
Rare off-style bluffs remain possible because the audit itself contains a
large flop bluff. Tests exercise threshold sizes, real native-engine +291
execution, the actual judge adapter, probe sizes and teacher-pool membership.
This approximates the three-game observations; it is not recovered bot17
source and does not establish long-run strength against the online opponent.

## Local acceptance

Full inherited/new regression: 175 passed in 167.62 seconds. Final focused
refund/guard regressions: 39 passed in 2.91 seconds. The actual source-only
import test covers spawn workers that have `src` but no repository root on
`sys.path`; the first development smoke exposed that boundary, and moving the
old engine import into its sole caller repaired it.

A full two-worker development smoke with the real 192/4/6 champion reached
COMPLETE in 110.55 seconds; all three generations had preflop TV=0. The wider
residual / 16-iteration river ZIP pilot measured mean 25.93ms, p99 54.60ms and
cold maximum 0.203s while a separate regression process was running. Both
partial-match smoke evaluation suites also passed their hard mean/p99 gates.
Smoke uses tiny one-epoch networks and partial two-hand matches; production
CFV quality, full matches, 5090 kernel acceptance and target platform limits
remain server checks. The final handoff also runs a clean committed-source
smoke. No server job, existing model, checkpoint or release was modified.
