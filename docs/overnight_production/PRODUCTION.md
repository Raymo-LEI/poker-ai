# Overnight phase-two production

This controller builds on PR #82's exact
`531281544d96f8d49ce910444e5d383fe4823c5d` source. It runs in a separate clean
checkout, leaving the running evening checkout and all source checkpoints
untouched. Production defaults to an 8.5-hour target and a fixed 9-hour
process-group deadline. CPU operation and `--allow-dirty` are smoke-only.

Run `scripts/sprint24/overnight_production.py --expected-commit COMMIT_SHA`
using the server's working Torch/NumPy/PokerKit/phevaluator interpreter. The
handoff command accompanying this change fetches that exact revision into a
detached worktree. Do not switch the checkout used by the running evening job.

## Identity and handoff

Every stage checks both HEAD and the complete `git status --porcelain`,
including untracked files. Keep outputs outside the checkout. The original
champion must have SHA256
`f298ab7a44956f92f2fa02c914ec4a68749a1381be2e2952159f989cc53c48fe`,
192 dimensions, four Transformer layers and six heads. Production probes an
actual RTX 5090 matrix multiplication and backward kernel. It never installs
or replaces server dependencies.

The latest completed, non-smoke `FINAL_REPORT.json` under
`/root/autodl-tmp/sprint24/evening_production/` supplies baseline/G1/G2 scores.
Ranking equally weights the benchmark and tournament phases; opponents are
equally weighted within each phase using measured mbb/hand. Every benchmark
cell must contain at least 5,000 hands and every tournament cell 1,000 hands.
The selected source/model hashes, release binding, checkpoint configuration,
and every exported checkpoint tensor are verified. A corrupt completed report
or selected artifact is an error. An absent completed report, or `--fresh`,
uses the original champion. No earlier CFV labels or networks are reused.

Warm-start postflop weights are retained. The preflop residual is always
rebound directly to the original champion, including its original 64-wide
hidden branch. The reference Transformer and alpha must match that champion.
Expanded checkpoints validate the independent anchor's original shape. The
final runtime probe compares each generation directly with a separate export
of the original champion, requiring preflop TV <=1e-12.

## Expanded bilateral tree and online budgets

`check_raise=True` adds the following continuation, for both player positions:
check, opposing half-pot/pot/jam, then fold/call/pot-sized check-raise/jam,
then opposing fold/call. If the opponent bets B into P, a pot-sized raise after
calling is a total future investment P+3B. Duplicate sizes are removed. A
short raise is retained only as a stack jam; no raise follows an opposing jam.
IP check-after-check still ends the street. Root executable button amounts
remain unchanged. All showdown branches use `settle()`; a folding bettor's B
is lost, while the check-raiser's unmatched chips return to that player.
Signed DCFR(1.5,0,2), simultaneous strategy snapshots, own-reach averaging,
compatible-combo conditioning and centered zero-sum root-BB values are retained.
The legacy evening tree remains the default when this option is absent.

The pilot measures the exact ZIP `runtime.respond()` with the maximum
256-wide residual, full guard and expanded tree. It selects the first passing
online budget from 4/4/8, 3/3/6 and 2/2/4 flop/turn/river iterations, with one
chance sample and the existing 0.5/0.5/0.3 blueprint mixture. Offline precision
is never lowered to meet an online timing limit. Every guard variant and
formal release repeats mean<40ms, p99<120ms and maximum cold<0.5s checks.
Final full-match runtime samples must also pass mean/p99 limits.

## On-policy PBS / exploration CFV mixture

Exactly half of each train/holdout dataset consists of actual current-blueprint
self-play states from the checked-in HUNL engine; half consists of independent
Dirichlet exploration states. These are generated engine trajectories, not
recorded online/human games or a measurement of the deployed search mixture's
stationary distribution. Each street has 2,048 independent holdout rows.

For each self-play hand, 64 public private-hand particles are sampled without
access to either player's dealt cards. Both players start with public uniform
mass on that support. Each observed action updates only its actor's mass using
the current model's conditional action probability, including the native
action mask and actor's public HUD. Board reveals remove blocked particles.
There is no strength-heuristic likelihood, hero-hole conditioning, epsilon
floor or street resampling. An exhausted particle support ends collection of
that hand; rejection counts are recorded. The resulting distributions are
projected into the established 32 public buckets, with uniform within-bucket
support in the solver. This is a finite-particle/bucket PBS approximation,
not exact full-combo posterior inference.

River labels use 64 signed-DCFR iterations; turn labels use 32 iterations and
four public chance samples with the newly trained river continuation network.
The public solver retains full combo support. Both CFV heads train for 50
epochs with cosine learning-rate annealing, from 1e-3 to 1e-5. The holdout gate
requires at least 10% MAE improvement over the training-mean bucket constant,
normalized pot-MAE <=12%, and a positive paired 95% lower bound on improvement.
Torch/NumPy parity must also pass. Counts cap at 500,000 river and 350,000 turn.
These gates measure solver-value approximation, not poker playing strength.

## G3, G4, G5 and Astra

Each generation collects new current-blueprint self-play states and public
posteriors. Pure DCFR average policies and centered conditional Hero root
values supervise the policy and value heads; labels contain no online prior
or policy mixture. Teacher budgets are 32 flop/turn and 64 river iterations,
with two chance samples. Collection publishes observations, policy/value
targets, priors, exact identities and hashes for every split.

The residual expands from 64 to 256 dimensions without changing initial
logits. Its extra capacity, context, fusion, policy and value parameters train;
the Transformer remains frozen. Twelve joint epochs minimize policy cross
entropy + 0.01 KL to the source blueprint + 0.1 SmoothL1 root-BB value loss,
with cosine learning-rate annealing. Frozen backbone features are cached once.
The independent preflop branch is outside the optimizer.

Distillation validation requires non-regressing policy CE and value MAE and
zero preflop drift. After validation, every generation actually invokes
`surgery_fn`. Astra uses bad/good solver-CE activation contrasts from training
data, the existing orthogonalized direction/column ranking/outer-product
operator, three sites, three alphas and two column counts: 18 rank-one
candidates. An independent tuning split chooses a candidate; a separate
surgery-gate split requires positive paired 95% CE gain above 1e-5,
non-regressing value MAE, and preflop TV <=1e-12. Final acceptance uses another
untouched holdout. Surgery significance refers to solver-policy fidelity,
not a statistically established match-score gain.

Rejected edits roll back. A rejected generation retains its predecessor and
is explicitly marked `accepted=false`; later generations still execute.
Nothing forces a failed generation to look like an improvement. The final
matrix includes baseline and all three generation roles, including rollbacks.
Policy-evolution probes hold guard mode fixed and additionally report raw
blueprint TV, so guard ablations cannot inflate the claimed learning change.

## Empirical guard ablation and final evaluation

The previous export API defaults to `deployment_guard=None`, which disables
the empirical guard, despite its runtime always calling `apply_guard`. Phase
two makes both compared modes explicit. Full uses canonical `V10_GUARD` with
`enabled=true` and `victory_blind_lock_enabled=true`; its existing per-rule
defaults are preserved. Slim retains the same preflop behavior, blind-lock
path, native all-in mask and final executable-domain validation, while skipping
postflop empirical distribution edits. Search continues in both modes.

All four roles run both modes against all five tournament opponents, four
duplicate 50-hand pairs each, on separate tuning and confirmation seed sets
1951000000/1961000000. Slim is selected only when its mean mbb/hand beats full
on both independent sets; ties retain full. No final evaluation seed tunes
the mode. Every variant must pass the packaged latency gate.

The selected formal ZIPs are allocated and latency-checked before full
evaluation; both final suites execute those exact formal bytes. The historical
seed base 1800000000 reproduces PR #82's evaluation namespace, and independent
holdout base 1971000000 is unused by training, surgery or guard selection.
For each role and each seed suite:

- equity, TAG_Pro and Trapper_CR: 50 duplicate pairs of complete 50-hand
  matches, 5,000 played hands per opponent.
- Bot17_Probe3Bet, TAG_Pro, Trapper_CR, equity and NearShove_19949: ten duplicate
  pairs of complete 50-hand matches, 1,000 hands per opponent.

This is 160,000 final-evaluation hands plus 32,000 guard-ablation hands across
four roles. Scores, paired confidence intervals, per-match files, runtime
counts and all matrices are retained. No match terminates on a projected lead.
The final holdout ranks the four frozen candidates; selecting its winner is
model selection, not an unbiased estimate of a winner's future advantage.

## Measured allocation and deadlines

Workers default to `min(24, cpu_count)`. Spawn workers set all BLAS/OpenMP
thread variables and Torch intra/inter-op threads to one and hide CUDA.
The parent keeps its CPU affinity and owns the 5090. Calibration measures full
worker-pool river/turn mixture generation, real teacher and anchor collection,
two CFV and policy/value GPU epochs, and a complete packaged 50-hand match.
Worker startup is separated from active compute time.

The plan reserves all 3,840 evaluation/ablation matches at measured match cost,
with a 1.5 safety factor, 1,800 seconds for startup/surgery/IO and 12% target
slack. Independent holdout collection is priced separately. Dataset costs
include all 50 CFV epochs and all three 12-epoch policy/value generations.
Remaining budget is split 25% river, 30% turn and 45% three-generation training;
row counts round down to multiples of 256. Floors are 8,192 river, 4,096 turn
and 4,096 teacher rows per generation. An infeasible machine-measured floor
fails preflight instead of changing precision or silently overrunning.

The watchdog owns a new session/process group. Failure, SIGTERM or Ctrl-C
cleans the entire group even if its leader exits before its workers. At the
9-hour deadline there is no additional five-second grace beyond the deadline.
Watchdog stops write `WATCHDOG_FAILED.json`; stage errors write `FAILED.json`.
No complete deliver report is published after a failed gate.

## Delivery

The shared persistent counter under `/root/autodl-tmp/sprint24/deliver` reserves
strictly increasing versions; failed reservations remain consumed. If V22–V24
have been allocated by evening and no later versions exist, the four new roles
receive V25–V28. Never use a new empty delivery root to reuse visible numbers.

Each `UPLOAD_Vn_OVERNIGHT_<ROLE>_<YYYYMMDD>/` contains only `Vn_SOURCE.zip`,
`Vn_MODEL.npz` and a one-page upload note. The same Vn/release binds source,
weights metadata, WEIGHTS_NAME, debug, suggested bot name and report. The note
contains both SHA256 values, the full expected release and the replay verifier
command. No package is automatically uploaded or deployed.

The run publishes `OVERNIGHT_FINAL_REPORT.json` and `DELIVERY.md`, and atomically
copies the final report plus a timestamped copy into the shared delivery root.
Formal report availability, gate status and hashes determine readiness; package
directory existence alone never establishes a successful run. Smoke uses only
internal V999 fixtures, partial two-hand protocol matches, and no formal counter.

See [VALIDATION.md](VALIDATION.md) for local evidence and its limits.
