# Local validation

The local host is macOS/arm64, using the existing isolated Python 3.11.17 /
NumPy 2.4.6 / Torch environment and the actual verified 192/4/6 champion.
OPENBLAS/OMP/MKL thread counts are one. No server training, checkpoint change,
upload or deployment was performed by this development task.

The phase-two suite checks the new check-raise leaves against an independent
chip-state ledger, including short stacks, all-in refunds and fold payoffs.
A sparse expanded-tree game is compared with an independent dense DCFR
implementation. The inherited Kuhn equilibrium, signed regret, own-reach,
zero-sum, public evaluator and real judge tests remain part of the regression.

New tests also cover exact 50/50 real-trajectory/exploration mixture counts,
parallel/serial labels, public Bayes updates and board masking, real value
targets and value-head updates, all 18 actual Astra candidates, significance
and rollback thresholds, checkpoint/portable frozen preflop routing, full/slim
guard export behavior, strict holdout quality, handoff ranking and integrity,
resource-budget floors and process-group descendant cleanup.

Final expanded regression: **145 passed in 167.63 seconds**. All 24 exported
portable modules also passed Python 3.6 grammar parsing, and `git diff --check`
passed. Grammar acceptance does not establish execution on Botzone's runtime.
After strengthening CFV/Astra confidence intervals to cluster correlated rows
by independent rollout, the affected regression suites passed **87 tests in
17.59s**, including all 18 phase-two cases. Added assertions reject a spurious
row-level significance result when only two independent rollouts support it.

Reproduction (use an interpreter with the repository's current dependencies):

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python -m pytest -o addopts='' -q tests/test_overnight_production.py tests/test_evening_production.py tests/test_evening_v2.py tests/math_repair tests/test_wizard_solver.py tests/test_search_entry.py tests/test_deployment_guard.py tests/test_stage2.py tests/test_stage2_botzone_native.py tests/test_stage2_traces.py tests/test_cfr_traversal.py tests/test_cfr_kuhn.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u scripts/sprint24/overnight_production.py --smoke --fresh --workers 2 --out /tmp/overnight-smoke-UNUSED
```

The source must be committed and clean for the second command; local development
may explicitly add `--allow-dirty`. Smoke still verifies the champion hash and
architecture and executes actual exported ZIP latency gates. Only CUDA identity,
production counts/epochs and CFV statistical acceptance are smoke-exempt.

Two complete development smokes reached COMPLETE. The second took 114.14s,
called Astra 18 times per generation and retained or rolled back each edit.
G3/G4 were accepted; G5 was rolled back by the independent final holdout.
Fixed-guard raw-blueprint maximum TV was 0.01918/0.01956/0; every generation's
preflop TV was exactly zero. These are small-sample influence/rollback checks,
not evidence of increased poker strength. Guard choices and match outcomes
from partial two-hand smoke matches are not production rankings.

The 256-wide expanded-tree fixture's actual ZIP pilot passed with approximately
19ms mean, 29ms p99 and 0.18s maximum fresh-process time in the first smoke.
These local fixture timings do not certify RTX-server throughput, trained
production CFVs, Botzone Python 3.6/NumPy 1.19, memory limits or playing strength.
The controller repeats all timing gates for every formal package and the full
evaluation; production 50-epoch CFV acceptance and all full matches remain
server-run gates.

An initial orphan-cleanup test exposed a macOS sandbox denial of a signal-0
process-group probe after the leader exited. Cleanup was changed to send
TERM followed by unconditional group KILL after the bounded grace interval,
without assuming that leader exit means descendant exit. The corrected real
descendant cleanup test passes. This was a discovered and repaired test failure.

Regression results, tested source hashes and development-smoke evidence are
recorded in `LOCAL_VALIDATION.json`. The final handoff additionally reports a
complete clean-source controller smoke at its locked commit.
