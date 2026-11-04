# Production controller: fixed source, measured budgets, real champion

Both #80 and #81 are merged ancestors of the production branch. This controller
keeps their canonical `respond → Agent.policy → wizard` entry and math tests.
Server operators fetch the locked commit and run one command; the controller
never edits repository source, the champion checkpoint or previous releases.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u scripts/sprint24/evening_production.py --hours 3
```

Use the exact commit supplied with the handoff, not a moving branch tip.
The controller checks the commit and tracked source at every stage. Production
rejects `--allow-dirty`; that flag is only available with local `--smoke`.

The default champion is located at
`/root/autodl-tmp/sprint24/ladder/u250_ladder_slayer.pt`, with bounded known-path
fallbacks and filename discovery. Its SHA256 must match the verified champion
constant in `evening_production.py`, and its config must be 192 dimensions,
four Transformer layers and six attention heads. The residual head remains
64 wide. This prevents accidentally using the earlier random 32-d smoke policy.
An intentional different champion requires an explicit `--champion` and
`--expected-champion-sha256`; operators must not change source to bypass checks.

Production requires an actual RTX 5090 CUDA kernel/backward probe, not merely
`torch.cuda.is_available()`. It uses the server's installed compatible Torch,
NumPy, PokerKit and phevaluator dependencies. CPU execution is restricted to
`--smoke`. No CUDA/5090 production run was performed on the local Mac.

## Online prior and mixture contract

Offline CFV labels and distillation teachers remain pure signed DCFR averages.
They never warm-start from a blueprint and never train against the online mix.

For online decisions, current-model blueprint probabilities are masked and
normalized over executable buttons. The current Hero root information set gets
that policy as sigma_1 and its zero-positive-regret fallback. Other private
hands and future information sets stay uniform: the actual private hand's
network output is not broadcast as if it were a public all-hand strategy.
Strategy sums start at zero and use the ordinary t^2 average; no hidden prior
pseudo-count is introduced.

- Flop/turn: 4 iterations, 1 public chance sample; final policy is
  `0.5 * blueprint + 0.5 * DCFR_average`.
- River: 8 iterations, exact terminal showdown; final policy is
  `0.3 * blueprint + 0.7 * DCFR_average`.
- Preflop: frozen source residual routing; no training-induced policy drift.
- The named final empirical guard still executes once after search. Debug
  identifies `blueprint_dcfr_mixture`, not an equilibrium-policy claim.

The three roles are baseline, G1 and G2. A persistent locked release counter
allocates fresh increasing versions. V21 already identifies an earlier smoke
release, so the first new formal trio is V22/V23/V24 if those numbers are free.
The shared default delivery root is `/root/autodl-tmp/sprint24/deliver`; it is
not reset per run. Failed allocations also consume their version. Do not point
`--deliver` at a new empty root to reuse old visible versions.

## Fixed offline precision; calibrated production counts

- River CFV labels: 64 DCFR iterations on complete public combo support.
- Turn CFV labels: 32 iterations, 4 public chance samples.
- Distillation teacher: 32 flop/turn iterations, 64 river iterations,
  2 public chance samples where required.
- CFV training: 30 epochs. Residual distillation: 12 epochs for each generation.
- Separate 1,024-row holdouts/anchors for each production split.

Default worker count is exactly `min(24, os.cpu_count() or 24)`. Every spawn
worker forces OPENBLAS/OMP/MKL/NUMEXPR/VECLIB threads to one, hides CUDA and
sets Torch intra/inter-op threads to one. The parent retains the full affinity
set and schedules the 5090. Frozen backbone features are prepared once on GPU
before residual epochs; no backbone parameters are trained.

Before allocating production counts, the controller measures on the actual
server: parallel river/turn solving, champion search-driven collection and a
complete 50-hand target-judge match. Pools wait until every worker initializes;
active compute span is measured separately from process startup, preventing
startup overhead from masquerading as per-row cost.

`plan.json` fixes sample counts before the main run. The target is 3 hours,
with measured evaluation cost, 900 seconds for training/startup/IO and 15%
slack reserved. Per-row rates get a 1.25 safety factor. Remaining CPU budget is
split 25% river / 35% turn / 40% both distillation generations. Count bounds:
river 8,192–300,000; turn 4,096–200,000; each distillation generation
4,096–200,000. An infeasible floor is a preflight failure, not a silent
precision reduction. Thus final counts are intentionally machine-calibrated,
not a fabricated fixed promise based on an unmeasured server.

Local single-process measurements were 0.19584s per 64-iteration river label
(16 boards) and 0.35149s per 32-iteration/4-chance turn label (8 boards).
Ideal division by 24 gives 0.00816s / 0.01465s per row; these are projections,
not measured RTX-server throughput. The actual server pool calibration replaces
these projections. A process-group watchdog stops only this run and its workers
at 3.5 hours. This is a deadline, not a guarantee that training or quality gates
will pass on unmeasured hardware.

## Ordered stages and gates

1. Verify source, champion identity/topology and RTX 5090 kernel execution.
2. Build independent pilot CFVs and measure the real packaged runtime.
3. Calibrate CPU/evaluation throughput; atomically publish the plan.
4. Parallel CFV generation and GPU training. Holdout MAE must beat a constant
   bucket predictor by at least 5%, normalized pot-MAE must be <=15%, and
   Torch/NumPy residual parity must pass. These are gates, not strength proofs.
5. Package baseline and run its full latency gate before expensive distillation.
6. Parallel search-driven G1/G2 collection, frozen-route training, independent
   validation and rollback. The online mix does not contaminate teacher labels.
7. Package and latency-check G1/G2. Independently probe actual Agent.policy
   distributions: each generation must change postflop policy and keep preflop
   TV <=1e-12. A rejected/identical model cannot be advertised as an evolved one.
8. Run all target-judge evaluations through the final ZIP's runtime.respond,
   in parallel. Each model gets 5,000 hands against each of equity/TAG_Pro/
   Trapper_CR (50 duplicate pairs each), plus ten duplicate 50-hand match pairs
   against each of the five tournament scripts: Bot17_Probe3Bet, TAG_Pro,
   Trapper_CR, equity, NearShove_19949. Total: 60,000 actually played hands for
   all three models. Matches are never terminated early on a lead estimate.
9. Check evaluation latency, champion immutability and source lock; publish
   FINAL_REPORT.json, DELIVERY.md and per-version upload directories.

Preflight identity/GPU errors exit immediately. Once the run directory exists,
a failed quality, evolution, latency or worker check produces FAILED.json
and a nonzero exit; it never silently downgrades precision or substitutes a
random model. Pool tasks have bounded stalls, epochs print progress, and the
outer watchdog prevents an unnoticed many-hour run. A failed distillation gate
still rolls back; production then rejects an unchanged generation rather than
forcing an unsafe update. No training method guarantees playing-strength gains.

## Local evidence and remaining target checks

[LOCAL_VALIDATION.json](LOCAL_VALIDATION.json) contains the measured data.
With the actual 192-d champion, the full online prior/mix and production
4/4/8 budgets, 512 changing requests gave mean 19.25ms, p99 35.05ms;
32 fresh ZIP processes had maximum wall time 0.1785s. Each board-size group
also passed. These are actual packaged runtime calls, not solver-only timings.
The CFV heads were small trained fixtures; production rechecks each final model.

A real two-generation CPU probe used 128 teacher rows, three epochs and fresh
64-row validation/holdout splits per generation. Both were accepted. Maximum
postflop TV was 0.002139 (G1 vs baseline) and 0.002690 (G2 vs G1); both preflop
TVs were exactly zero. This demonstrates actual train-to-runtime influence,
not a claim of improved online strength. The probe used 32 teacher iterations;
the production river teacher budget is increased to 64.

The local complete controller smoke uses the actual champion and two spawn
workers, builds internal-only packages, runs both generations and all five
opponent adapters, and reaches COMPLETE. It uses partial two-hand matches;
only production uses complete 50-hand matches and the full sample counts.
Local smoke never consumes formal visible release numbers.

The local host is macOS/arm64, Python 3.11.17, NumPy 2.4.6, with all BLAS thread
variables set to one. Affinity pinning is unavailable on this host; the Linux
latency subprocess pins one permitted CPU without restricting the production
parent or its pool. Local timings do not certify Python 3.6/NumPy 1.19,
Botzone memory limits, a busy future machine or unmeasured server throughput.
Production repeats the hard mean<40ms, p99<120ms, cold<0.5s gates and aborts if
any final package fails. It does not upload or deploy a bot automatically.
