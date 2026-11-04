# Xeon + RTX 5090 server profile

Run `scripts/sprint24/overnight_production.py --server-profile` from a clean,
locked checkout. The earlier controller and its tests retain their original
40ms gate; this explicit server profile implements the requested new contract.
No existing test is removed, edited or newly skipped.

## Measurement and CPU allocation

The first measured runtime is frozen PR85 commit
`257a7623dc399a5e63b272f4d7143b1f19160eea`, exported from a Git archive with its
own compatible, unqualified calibration CFVs. It measures both the unchanged
PR85 profile and the same profile with its PR84 HUD prior enabled. The latter
is required because PR85's overnight default had disabled that prior.
Calibration fixtures are never final-arena or upload inputs.

A corpus is collected from complete, genuine 50-hand target-judge matches,
retaining the complete cumulative requests/responses and previous cross-process
MatchCtx data. This includes the real late-match history overhead.
States are sampled evenly across preflop/flop/turn/river with replacement;
unique-state counts and IDs are reported. No cards or HUD values are invented.
Each benchmark has exactly 32 warmup requests, 512 timed requests and 12 fresh
process launches. It reports mean/p50/p99/max, per-street statistics, all timed
samples, cold samples, full-combo river calls and nonzero-HUD calls.

The server hard policy is **p99 <120ms, max <300ms, cold max <0.5s**. Mean
**<65ms** is a warning only. Equality fails the corresponding strict limit.
The preliminary frozen server measurements in `pr85_benchmark/` used a
compact current-request/real-data corpus and showed approximately
59.5/60.4ms mean, but 178.5/180.6ms p99 without/with HUD, so 16 river iterations
are not assumed deployable. The mandatory startup stage repeats the frozen
measurement with the complete cumulative payloads before calibration.
The controller benchmarks iteration budgets
16/12/8/6/4, retaining full 1,081-combo river solving and the expanded tree.
It does not change thresholds to accommodate an arm.

Inference uses one allowed logical CPU other than CPU0. All its SMT siblings
are excluded from the controller, label/training workers and judge/opponent
workers. BLAS and Torch threads are one. The arena uses one persistent inference
process on this core and up to 24 judge/opponent workers on other CPUs; it does
not run 24 inference processes on the same core. Engine service time and RPC
queue/roundtrip time are reported separately. The serial inference cost is
reserved without dividing it by worker count. CPU affinity, sibling topology,
cgroup quota and all exposed governors are recorded in `CPU_RESERVATION.json`
and the run log. Isolation covers run-owned processes; the container cannot
promise host-wide exclusive ownership or change the host governor.

## Surgical guard and arm definitions

Surgical retains Legacy's Fold-only dominant-tail and thin-call rules and its
anti-punt principle. It never invokes pot control, blind protection, strong-hand
Fold relief, forced-call rules, or probability purification. Every priced
ordinary decision preserves/increases the legally normalized input Fold mass.
Free Fold becomes Check; legal masking and proven match locks remain explicit
exemptions. River outputs get only legal normalization/free-check safety.

The common chip cap is `max(30BB, min(0.25 * starting_stack, 2 * current_pot))`,
with the existing 6BB preflop cap. The same threshold is applied to Call and
all raise/jam investments. Legacy strong-value qualifications are retained;
otherwise weak mass goes to Fold/Check and medium mass to an uncapped
Call/Check. A medium redirect cannot manufacture pure Call: redirected mass
above 95% total Call goes to Fold. Existing pure Call is not invented by this
rule. Each rule records fires and boundary total-variation mass; the pipeline
also records its net TV. Legacy helper boundary totals can overlap at nested
calls and are not summed to claim a net pipeline TV.

All arms use the same final G5 checkpoint (including independently validated
rollbacks) and the same production CFVs:

- A: Legacy, solver HUD and size priors off.
- B: Slim, solver HUD and size priors off.
- C: Surgical, solver HUD and size priors off.
- D: Surgical, PR84 HUD response prior plus size-conditioned RNR range.

Neural MatchCtx features/reconciliation remain identical across arms. D caps
HUD/RNR eta at 0.25, retains exponential pot/lead-risk attenuation and sets
eta to zero at 40BB risk. Its size range is conditioned once: the separate
HUD range reweighting is disabled. Offline teacher targets remain HUD-free.
The archived PR85 measurement intentionally retains its original HUD formula;
D alone uses the new 0.25 bound.

## Production, arena and delivery

CUDA identity/backward probes, source/champion hashes, calibrated label budgets,
50/50 PBS/Dirichlet data, 2,048-row holdouts, 50 cosine CFV epochs and the G3/G4/G5
Policy+Value/Astra checks remain. The final arena asserts production manifests,
both recorded 50-epoch cosine runs, source/array hashes, production sample
floors and both CFV holdout gates. It has no fixture/smoke bypass.

A, C, D, then optional B each play exactly 200 complete 50-hand matches:
5 opponents x 20 independent deck pairs x 2 seats. All arms share the same
seeds. If the measured budget cannot fit four arms, B is dropped first; epochs,
holdouts and the other three match counts are not reduced. If even three do
not fit, the run reports failure rather than claiming a completed comparison.

The summary reports match wins/ties, mbb/hand, paired 95% EV/win intervals vs A,
per-opponent paired results, rule fire/mass statistics and the 20 largest losing
hands with all Hero decisions' before/after distributions and public state.
An arm's hard latency failure excludes packaging while other arms continue;
its research arena may still provide the A comparison. No arm is promoted
merely because training/smoke finished.

Arms rank by paired EV CI lower bound vs A, then match win rate. Three safe
arms produce V26/V27/V28; status is `promoted` only for a strictly positive
lower bound. With no promoted safe arm, eligible Legacy is V26. Legacy is
explicitly authorized for this profile's fallback packages. Reservations and
atomic staging prevent a crash from reallocating/reusing the version numbers.
Exact final ZIP/NPZ bytes are benchmarked before publication. A package failing
this final check is excluded; unsafe bytes are never published. If fewer than
three safe arms remain (or training never qualifies), the report explains the
shortfall instead of manufacturing three packages or overwriting versions.

Each upload directory contains only `Vn_SOURCE.zip`, `Vn_MODEL.npz` and one
Chinese upload page with hashes, release/debug binding, bot name and replay
verification command. V26-28 must still be unused in the destination. No
Botzone account upload is performed.

## Resume and operator handoff

Use the exact same command/out/commit with `--resume` after a crash. Stage
receipts hash trained artifacts; completed stages are not retrained. Each arena
match is atomically committed, then each completed arm is checkpointed. A
partially completed arm schedules only missing matches. Resume rejects changed
inputs/artifacts/core reservation. Version reservations and published-byte
receipts cover packaging crash windows as well.

The default is an 8.5-hour target and cumulative 9-hour process-group watchdog
from first launch, including resumes. No absolute morning cutoff is set.
`--benchmark-only` runs the genuine benchmarks/calibration without formal
training; continue that run with `--resume` and without `--benchmark-only`.
`--smoke` is development only and explicitly cannot enter the final arena.

Canonical report: `<out>/OVERNIGHT_FINAL_REPORT.json`. All stages continuously
update that single summary, including failures/exclusions. Supporting artifacts
are in `<out>/stages/`, restart state in `<out>/CHECKPOINTS.json`, and packages
in `/root/autodl-tmp/sprint24/deliver/UPLOAD_V26_*` through `UPLOAD_V28_*`.
