# Clean search architecture (supersedes prior guard/size patches)

The live release audited earlier was V22_SEARCH_LOCK_20261104. Its old guard
behavior and the previous audit-patch measurements remain historical evidence.
The four principles below define the current overnight deployment contract.

## 1. Slim safety owns no poker strategy

Every Evening street, including preflop, uses only executable action masking,
finite/nonnegative normalization and free-Fold mass moved to Check. The native
voluntary all-in heuristic and Stage-1 abstraction masks are bypassed by the
explicit `legal_only` runtime profile; authoritative judge buttons still apply.
There are no call-price fuses, 30BB shove caps, tail pruning, blind-protection
equity thresholds, or post-search Fold/Call/Raise redistribution.

Proven own-score locks run before search, including judge-legal free folds.
An unopened SB with a confirmed opponent lock may call exactly the extra 50
chips. An unconfirmed opponent-lock probe cannot force a Call in this pipeline.
These are public score/schedule constraints, not hand-strength overrides.

The old R7 full guard exists solely as an explicitly tagged internal
`legacy_full` comparator. Formal overnight packaging rejects it; it cannot win
an ablation and become the deployment mode. The old comparator and new Slim
mode share the same checkpoint, legal-only observation, RNR ranges, search
budget and terminal utility, isolating the final guard intervention.
Legacy timing is measured and reported in observe-only mode; an old comparator
missing the deployment timing limit cannot suppress the required comparison.
All Slim pilots and formal/final-evaluation gates remain strictly enforced.

Each role is evaluated in **200 complete 50-hand matches per mode**: five
opponents, 20 common deck pairs each, two swapped seats. There are 100
independent duplicate pairs per mode, 400 matches / 20,000 hands per role across
both modes. Reports retain the actual paired scores, EV and net-positive-match
win rate, and 95% paired confidence intervals. Production promotion requires
positive lower confidence bounds for both EV and win-rate gains for at least
one Slim role. Otherwise QUALITY_REPORT.json preserves all evaluations,
FAILED.json records the failed improvement gate, and no COMPLETE delivery
report is published. The old guard is never re-enabled as a fallback.

## 2. Every unlocked river runs full-combo DCFR

`respond -> Agent.policy -> evening_strategy.fuse -> PublicSolver` resolves
all unlocked river nodes, both checks and facing bets, with all 1,081
board-legal combinations and exact terminal showdown ranks. River resolution
does not depend on a CFV network being present. Production does not silently
fallback to the blueprint after a river numeric/range error. Debug/audit fields
identify `river_full_combo`, support size, range model and terminal utility.

Full-combo exact showdown is not an infinite-iteration equilibrium claim: the
online solver has a measured finite DCFR budget and blueprint mixture. Public
priors and opponent observation likelihoods remain model approximations.

## 3. Nonlinear terminal match outcomes, never a shove rule

Online river/fold/all-in-runout terminals evaluate each win/tie/loss chip
outcome separately. Current-hand sunk investment and unmatched-bet refunds
are accounted for before computing final cumulative score. The solver uses
`P(match win)+0.5*P(match tie)-0.5`; the opponent receives the exact negative.
There is no residual 50/50 chip/CDF utility blend.

Remaining blind schedules are taken from the public match metadata, including
seat alternation. On the final hand, final score sign gives exact 0/0.5/1.
Beyond the future own/opponent blind-lock boundaries, the corresponding side
can guarantee the result by folding. Between those boundaries, a declared
zero-mean normal residual-score model uses sigma=12BB*sqrt(remaining hands).
That uncertainty scale is a model assumption, not fitted evidence or an exact
solution of the remaining match. For this declared model, terminal W/T/L
expectations are evaluated exactly; U(E[chips]) is never substituted.

Flop/turn all-in terminal runouts apply the same independent outcome utility.
Public runouts are sampled using the online chance budget; this is not full
enumeration of every possible flop/turn runout. Exact rank and utility mapping
apply separately at every sampled terminal. Non-all-in CFV networks contain
only chip first moments: they use an explicitly tagged common root tangent
in probability units, preserving zero-sum expectation. This is a continuation
approximation, not a nonlinear utility claim from a chip mean. On the final
hand, continuation uses a sampled checkdown outcome distribution instead.
Offline CFV labels/value-head supervision remain centered chip BB.

For the known Hand #17 jam, the current 16,112 stack means 3,888 chips already
invested. Starting match lead +911 becomes -2,977 after folding, exceeding the
opponent's 2,450 future blind budget. At that decision Fold already loses to
an opponent lock; a mathematical utility solver can rationally Call. Removing
the earlier forced-fold fuse is intentional. Risk can instead be avoided at
earlier decisions; the audit's known result is never an action rule.

## 4. Size information enters only public range inputs

The public size likelihood model still grades each action on the board visible
at that action and uses only public history. No opponent hole cards or actual
Hero hole cards define this public prior. Small probes retain broad air mass;
late overbet raises tilt the size component toward strong made hands.

The actual opponent range entering DCFR is now
`(1-eta)*uniform_combo_prior + eta*size_conditioned_prior`, default eta=0.5.
The runtime accepts eta in [0,0.5], with production using [0.25,0.5]. At eta=.5
every board-legal combo retains at least .5/N unconditional mass. Explicit
offline PBS/exploration ranges are not double-conditioned. Audit fields record
eta, the uniform floor and size-versus-mixed entropy.

This implements the requested conservative range mixture. It is not the full
restricted-Nash-response augmented information-set game, and no equilibrium
exploitability guarantee is claimed for this range mixture.

## Actual local 200-match comparison and limits

The complete paired local study used the real 192/4/6 champion with **tiny
one-epoch smoke CFV heads**, four CPU workers, seed base 1985000000. Each mode
played 200 complete matches, 50 hands each; all 400 matches completed in 203.5s.

- Slim net-positive-match win rate: 62.0%; old full guard: 71.5%.
- Paired win-rate difference: -9.5 percentage points, 95% CI [-18.71,-0.29].
- Slim EV: -2063.609 mbb/hand; old full: +394.260 mbb/hand.
- Paired EV difference: -2457.869 mbb/hand, 95% CI [-3482.841,-1432.897].

The local study **does not demonstrate improvement**. The unqualified smoke
CFV fixture cannot certify production playing strength. This negative result
is retained rather than inverted, omitted, or used to restore old rules.
The actual 50-epoch production models repeat the independent full comparison
and must pass the promotion gate; a source/pipeline smoke is insufficient.

All historical and independent final matrices remain: 160,000 final-evaluation
hands across baseline/G3/G4/G5, plus 80,000 guard-comparison hands. Machine
calibration reserves all 4,800 complete match jobs before sizing production
data. The 8.5h target, 9h group watchdog, champion/source locks, 24 one-thread
workers, and strict mean<40ms/p99<120ms/cold<0.5s gates remain unchanged.

The active server's earlier evening branch has its own mean<65ms patch; this
overnight source does **not** inherit that relaxed gate or mutate that run.

The production handoff has **no absolute 10:00 cutoff**. Launch with `--hours 8.5`
and omit `--deadline`: the complete 8.5-hour target and 9-hour process-group
watchdog start from the actual launch time. An optional absolute deadline remains
available for callers who explicitly need one.

The merged public terminal-reconciliation ledger is retained. For this overnight
contract, `context_prior=False` prevents the separate HUD response/range prior
from conditioning the public range a second time: the sole empirical range is
exactly `(1-eta)*uniform + eta*size_posterior`. The general solver retains its
optional HUD prior for existing callers; offline labels disable it as well.
