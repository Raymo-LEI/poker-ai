# Stage 2 v3 Pro Hunter

Date: 2026-10-29 (Japan time).

## Experiment decision

Start a new `stage2_v3_pro_hunter` experiment from C200 policy weights, for
**300 new PPO updates**. Optimizer, PFSP feedback, RNG, match counters and
self-play population are new. The frozen high-fidelity student is unchanged.
This is not a resume of the completed experiment, nor updates 201..500 under
the old identity. The original code/output/checkpoints are preserved.

The two residual-output blends were tested on the original selection deals.
Neither establishes an advantage over C200 in eight-script mean, or combines
both constituents' best individual matchups. The 50/50 blend ranks first under
the seven-candidate weakest-three-rank rule; that ranking is descriptive and
cohort-dependent. C200 was fixed as the online-test policy before results from
the new online campaign were inspected. No soup is automatically deployed.

## Residual-output blend

`ResidualOutputBlend` checks identical poker contracts, residual settings and
every frozen-reference tensor. It evaluates the shared reference trunk once,
then forms:

```text
logits = reference_logits
       + w100 * alpha * tanh(delta100(features, context))
       + w200 * alpha * tanh(delta200(features, context))
```

The legal mask and softmax are applied after blending. This is a convex blend
of bounded residual outputs, not an average of hidden-layer weights or action
probabilities. Models remain frozen. Descriptors pin both member checkpoint
hashes and update identities.

Use `scripts/eval_residual_blends.py` with the original complete local
`report.json` and two blend descriptors. It checks the original deck-schedule
hash and reuses old cells rather than repeating them. It only plays the new
eight-script and direct-match cells. A missing local teacher is marked as a
pending matchup and prevents a final recommendation until that cell is added.

## Same-deal results

Seed `20261030`, 1,000 duplicate deals per matchup, 64 equity samples and the
same 50-hand match-pair clustering as the original Pro Gauntlet. Two blends
added 27 played matchups / 54,000 hands to the original 50 / 100,000. The
expanded seven-candidate matrix therefore contains **77 played matchups,
154,000 hands and 98 displayed cells**. Mirrored cells are zero-sum derivations.

- **40/60**: eight-script mean **+1,362.72 mbb/hand**; Equity −927.75,
  TAG −900.25, Trapper −1,006.0. The paired equal-script gain over C200 is
  **−175.75**, nominal 95% CI **[−611.01, +259.51]**.
- **50/50**: eight-script mean **+1,311.97 mbb/hand**; Equity −1,059.5,
  TAG −719.25, Trapper −945.5. The paired gain over C200 is **−226.5**,
  nominal 95% CI **[−848.97, +395.97]**.
- C200 remains higher in eight-script point mean (**+1,538.47**), and C100
  remains better against Trapper by point estimate (**−428.25**). This does
  not prove a statistically significant ordering of all these policies.
- The seven-candidate rank rule provisionally selects **50/50**, but no
  candidate passes the adjusted simultaneous-bound champion criterion.

The aggregate gain intervals average each common-deal match-pair difference
across the eight scripts *before* estimating uncertainty, preserving the
shared-deck correlation across opponents. These reused deals are selection
data; independent confirmation remains necessary.

## Eleven-family training pool

The opt-in [configuration](../configs/stage2/pro_hunter.json) gives TAG and
Trapper **15% each at initialization**. Reference/history/self retain their
fixed 10% / 15% / 10% quotas, implemented as 2 / 3 / 2 slots in every 20-match
block. The remaining 35% is allocated proportionally to the original six
exploit families. The original nine-family default is still available.

PFSP subsequently changes exploit-family probabilities using the existing
rolling weakness rule. Fifteen percent is an initial probability, not a fixed
per-update count or a permanent sampling floor.

`JitteredProScriptActor` draws parameters once per 50-hand match and retains
them for all hands. Jitter covers opening/continuation cutoffs, BB defense,
reraises, dry/wet c-bets, strong-hand/draw trap rates, semi-bluffs, river
bluff-catcher folds and equity margin. Parameters and style identity never
enter observations.

Fixed `TAG_Pro` and `Trapper_CR` remain evaluation-only parameter settings.
They were verified against the previous implementation on 364 decisions in
200 hands, with identical actions. Periodic evaluation uses a distinct seed
and fixed parameters against all eight scripts plus the frozen reference.
This is a parameter/deal holdout, not an independent implementation or proof
of professional strength.

## Trap HUD and checkpoint compatibility

The legacy 12-field context remains the default for old models. The explicit
14-field context appends:

- `check_raise_freq`: raises among legal raise opportunities after that
  opponent checked and then faced a bet, counted at most once per street.
  Calls/folds at those opportunities count as failures. Checks by themselves
  and forced all-in calls do not create opportunities.
- `cbet_freq`: flop bets by the last preflop aggressor at its first flop
  action, when no flop aggression precedes it and no bet is faced. Checking
  counts as a failure; responding to a donk bet is not a c-bet opportunity.

Both are public-event Bayesian rates with prior strength two, using means
0.15 / 0.55. They can react after a few *opportunities*, without promising
identification within a fixed number of dealt hands. No future board,
opponent cards or opponent-family labels are used.

`--warm-start` validates the frozen reference and copies C200 policy weights.
The old 12 context columns are copied exactly and the new two columns are
zero-initialized. They can subsequently learn through PPO. Strict loading
checks match serialized context fields to policy settings. Optimizer and
population state are not imported. `--resume` remains separate and cannot
be combined with a warm start.

The initial checkpoint is saved at update zero. Updates 100/200/300 also get
dedicated policy checkpoints. Both actor paths, the match environment,
periodic evaluator and NumPy runtime use the model's explicit field list.
Legacy 12-field inference and new 14-field portable parity are tested.

## Launch and verification

```bash
PYTHONPATH=src python scripts/train_stage2.py \
  --config configs/stage2/pro_hunter.json \
  --warm-start runs/frozen_stage2/update_0000200.pt \
  --out-dir runs/stage2_v3_pro_hunter \
  --device cuda --updates 300 \
  --experiment-kind formal --experiment-name stage2_v3_pro_hunter
```

Replace example reference/teacher paths with verified artifacts. Use a new
output directory. The reference retention stop remains enabled; do not
silently bypass it to reach the budget.

The authorized server launch used isolated code/config/output directories.
Initial checkpoint inspection verified every copied C200 tensor, zero new
context columns, update zero, all eleven families and an unchanged frozen
reference after real GPU PPO updates. Local affected regression: **77 tests
passed**; server v3-specific regression: **7 passed**; frontend: **10 passed**.

The private local handoff records the exact PID, paths and latest progress.
Training is launched, not certified complete or stronger. Inspect independent
fixed-parameter evaluations at updates 100/200/300 and keep the original
checkpoints available for comparison.

## Online Slumbot protocol

The frozen C200 is evaluated for ten complete 50-hand HUD blocks, with no
online learning or parameter changes. Server-chosen online deals are separate
from the local duplicate tournament. Confidence intervals cluster the ten
HUD blocks and remain approximate at this small sample size.

An initial attempt stopped after 83 completed hands on a DNS failure; its
partial evidence is retained separately. A new full 500-hand attempt was
started from scratch. Neither the unfinished hand nor that partial campaign
is silently pooled into the declared 500-hand result.

The adapter now retries only failures known to occur before HTTP request
transmission: DNS resolution and connect timeout. Read timeouts, generic
connection errors and HTTP failures remain ambiguous and stop with partial
evidence rather than blindly resending a poker action. No session tokens
are persisted. See the accompanying result bundle for final online status.

The confirmation attempt completed **500 hands / ten complete blocks** in
about 15.6 minutes. It lost **455.5 BB**, or **−911.0 mbb/hand**, with a nominal
block-normal 95% interval **[−2,499.28, +677.28]**. There were 826 decisions,
zero illegal selections, zero mask leaks and zero nonfinite outputs. Fourteen
of 334 observed opponent bets were off-tree. SB VPIP was 39.6%, SB PFR 9.2%,
and BB fold-to-open 69.6%; these frequencies are diagnostics, not causal EV
attributions. Ten blocks provide limited precision, and crossing zero does
not demonstrate parity or profitability.

The [result bundle](benchmarks/stage2_pro_hunter_20261029/) contains the
expanded matrix, added-cell pair totals, online summary and token-free complete
500-hand ledger. The first 83-hand failed attempt is documented separately and
excluded from the confirmation statistics.
