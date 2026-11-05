# CFV data recovery after the 408a2ff holdout failure

The 0.12 pot MAE ratio gate is unchanged. This change restores data volume and
transfer initialization; it does not claim a passing holdout or stronger model.
Online production profiles remain **8/8/16**, full combination domain, two chance
samples, and the existing server latency/memory gates remain unchanged.

## Explicit historical compatibility

The completed V23–V25 run is
`/root/autodl-tmp/sprint24/evening_production/20261104T142449Z_5cbf1db6`.
Its `b/river_ds.npz` has 100,608 rows; `b/turn_ds.npz` has 68,096 rows.
`--reuse-source` must point to the unmodified source checkout at
`5cbf1db67d6b5f34c855c56b7aeddfbc7b8bb001`.

Import verifies NPZ SHA256, array digest, all manifest source hashes, completeness,
kind, dimensions, schema, units, and the actual historical git source bytes.
The entire feature/normalization implementation must match current `cfv.py`.
The repaired evaluator (`pack_many`), public bucket map, range expansion, and
bucket-value reduction must have identical AST hashes; the portable evaluator
must match byte for byte. `COMPATIBILITY.json` records a schema fingerprint and
those evaluator hashes. A schema string by itself is never accepted as proof.

Old Y lacks the expanded Check-Raise tree contract. Turn Y additionally depends
on the old river teacher. These labels are **not** directly merged. Instead,
verified old public X is reconstructed, checked by feature roundtrip, and
relabelled on the current expanded signed-DCFR tree. This avoids repeating
expensive 64-particle trajectory collection for the recovered reservoir.
The river/turn NPZ weights have compatible residual topology and BB units;
they initialize the corresponding heads before **50 new cosine epochs**.
This is transfer initialization, not acceptance of old weights as a production CFV.
Old artifacts are never modified or given a fabricated current-source manifest.

The exact reconstructed v3 contract contains no private hand; placeholder
board-legal hole cards are irrelevant to full public-combination solving.
Historical states are synthetic Dirichlet states. New augmentation is 50% PBS /
50% Dirichlet; the complete recovered dataset is explicitly **not** labelled 50% PBS.

## Independent offline configuration and throughput

Remote calibration uses 768 fixed training states per street at 24 workers. It compares river
16/32 against a 64-iteration reference, turn 8/16 against a 32-iteration reference,
all with four chance samples. The turn calibration teacher is the audited trained
historical river head, never a fixture. The cheapest CPU-time configuration with
mean pot error <=0.02 and p99 row error <=0.08 versus reference is selected;
reference remains eligible. These are calibration tolerances, not changes to the
independent final CFV gate. Offline teachers remain HUD-free and linear root BB.

`LABEL_THROUGHPUT.json` records CPU seconds, wall seconds, labels/CPU-second,
labels/wall-second, before/after on identical current-tree states, historical
V23 throughput, and the collapsed run's saved conservative end-to-end rate if
available. It distinguishes relabelling from PBS collection. No speedup is claimed
until this benchmark runs on the Xeon. GPU training and mixed collection are
calibrated separately; online solver budgets are not an input to this choice.

## Data volume, leakage, budget, recovery

Preferred counts: **108,800 river = 100,608 recovered + 8,192 new**;
**72,192 turn = 68,096 recovered + 4,096 new**. Each fresh holdout has 2,048 rows.
If necessary the planner drops optional arm B first, then subsamples the old
reservoir to **80,000 / 50,000 total training rows**. It never trains smaller heads.
Both reservoirs are prechecked before any optimizer; actual row counts are
checked again before each head. The estimate prices offline labels, rejection
sampling for fresh validation, 50 CFV epochs, three distillation generations,
serial 200-match arenas per retained arm, benchmarks, and a safety reserve.
If even minimum counts cannot fit, the report is `STOPPED_CFV_DATA_BUDGET`.

The old run's observed synthetic-label throughput was about 68.8 river and 30.6
turn labels/sec (different, old tree). Merely replaying that old workload was
about 1.03 hours. This is historical context, **not** a timing prediction for the
expanded tree. The new run targets 8.5 hours after preflight with a cumulative
9-hour controller watchdog. The actual wall-time estimate is mandatory output
of remote calibration (`plan.predicted_seconds`); no unmeasured speedup is assumed.
Remote full pytest time is additional preflight time and is logged separately.

Fresh holdout seeds are independent. Canonical 52-bit public-board masks exclude
**every** old training board, old validation board, and new training board,
including historical rows omitted by budget downsampling. Entire PBS rollout
clusters are rejected on overlap; retained observations keep cluster identities
for the original holdout gate. All sets and exclusion results are saved.

Relabel shards (4,096 rows), new datasets, holdout pools, and accepted heads have
source/input-bound receipts. `--resume` retains them; completed stage receipts
still verify all file hashes. A rejected fit saves weights and diagnostics before
raising, and cannot be retrained on the identical split by restarting. An old
408a2ff output directory must never be resumed by this new source identity.

## Remote execution

Use `scripts/sprint24/launch_cfv_recovery.py` on Linux/Xeon/RTX5090, not locally.
It archives completed V23–V25 reports, runs the **entire unchanged pytest suite**,
and only then starts production: frozen PR85 benchmark; current 32 warm-up +512
request server benchmarks; audited import; offline quality/throughput calibration;
planning; 50-epoch CFVs and original holdout gate; G3–G5; trained-runtime gates;
paired A/C/D/B arenas; ranked V26–V28 packaging. Only arm B may be budget-dropped.

Inference is pinned to CPU1; its SMT sibling is also excluded from training,
tests and judge workers. The governor and resource reservation are recorded.
Use existing `/root/miniconda3/bin/python` (Torch2.8.0+cu128, RTX5090). Do not
install `requirements-alphanl.txt`: that is a different legacy TensorFlow stack.
If a dependency is missing, use a task-specific environment compatible with the
current server stack and record changes; never change existing test assertions.

Launch with a full immutable SHA and a new external output directory. Subsequent
invocations auto-resume when `CHECKPOINTS.json` exists; explicit `--resume` requires
it. An advisory lock prevents duplicate launchers. Heartbeats and at most two
transient-failure retries are automatic; mathematical/source/data gate failures
stop with evidence instead of repeating identical work.

Reports: `<out>/FINAL_REPORT.json` and `<out>/OVERNIGHT_FINAL_REPORT.json` (same
summary), plus `LAUNCH_STATUS.json`, `REMOTE_PYTEST.json`, `remote_pytest.log`,
`production.log`, and stage-specific compatibility/throughput/plan/validation
reports. Packages are reserved atomically under
`/root/autodl-tmp/sprint24/deliver/UPLOAD_V26_*`, `UPLOAD_V27_*`, `UPLOAD_V28_*`
if those version numbers are still free. Existing published versions are never
overwritten; the reservation advances if needed. A CFV gate failure produces
reports and rejected models, **no qualified upload**. Legacy is eligible only
after the trained CFVs and per-arm hard gates pass.

Local validation for this patch is limited to `py_compile` and the bounded
`smoke_cfv_recovery.py`; the full suite, timings, training, and arena belong to
the remote launcher. Do not interpret the local smoke as target-machine acceptance.
