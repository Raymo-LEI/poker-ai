# Local validation, 2026-11-04

## Commands and results

- Expanded suite: **101 passed in 162.01s**. Command below includes portable
  math, generated entry, original Stage 2 export/guard/native/trace APIs, the
  100-hand synthetic referee protocol test and original external-sampling /
  Kuhn trainer tests. No online match or production training was launched.
- After the final raw-positive-undercall history, sampling-roundoff and
  fallback-diagnostic refinements: **45 tests passed in 3.07s**
  (`tests/math_repair`, `test_search_entry.py`, `test_wizard_solver.py`). These
  overlap the expanded suite and are not an additional 45 unique cases.
- NumPy-only mathematical suite after kernel cancellation stabilization:
  **36 passed in 2.30s**, Python 3.9 / NumPy 1.26.4. This overlaps the expanded
  suite and does not require Torch.
- **25 portable modules** passed Python 3.6 grammar parsing. This is syntax
  acceptance, not execution against Botzone's Python 3.6/NumPy 1.19 or its
  memory/CPU limits.
- `git diff --check` passed. All six audited original source/model SHA256
  identities were rechecked after the implementation and remain unchanged.

```
OPENBLAS_NUM_THREADS=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 <python-with-repo-dependencies> -m pytest \
  tests/math_repair tests/test_wizard_solver.py tests/test_search_entry.py \
  tests/test_deployment_guard.py tests/test_stage2.py tests/test_stage2_botzone_native.py \
  tests/test_stage2_traces.py tests/test_cfr_traversal.py tests/test_cfr_kuhn.py
```

The local dependency interpreter was Python 3.12.15, Torch 2.14.1, NumPy 2.5.4.
The checkpoint-loader environment variable is for existing trusted local test
checkpoints; no loader behavior was changed in this PR.

The first attempt at extended tests exposed an accidental `arraytorch` import
in the ported context's training `tensor()` method. That regression was fixed
before the successful expanded runs. An independently constructed limp test
also exposed premature preflop round closure; `pending` action accounting was
fixed and compared with the unmodified referee. Neither failure was hidden or
counted as a passing search.

## Numerical evidence

`numerical_report.json` records six dense-vs-prefix comparisons over three
boards and dense/sparse nonuniform ranges: maximum absolute equity error
**9.992007221626409e-16**, with matching reachability masks. A separate test
covers 1e-100 legal mass under a dominant blocking hand, and another compares
complete fast and dense sparse-range solves (tolerance 2e-12).

For Kuhn at 5000 iterations, value is **-0.05555318946277521**, versus exact
-1/18; independent enumeration of both players' 64 pure strategies gives
NashConv gap **0.0024853254171695593**. This certifies that tiny test only,
not the finite-budget poker abstraction. Forced +10BB terminal remains exactly
10BB. Four range-dependent river / enumerated-turn examples have maximum
weighted zero-sum error **8.88e-16**. The separately solved 100BB/200BB preflop
smoke games have error at most **1.11e-16**.

## Performance evidence and limits

`benchmark.json` is the authoritative raw timing / source SHA256 record. It
separates a 100-call precomputed-rank kernel measurement, three full 1081-combo
24-iteration solves (with/without rank construction), and three fresh
subprocesses loading the original frozen policy and reconstructing six history
actions before a real `respond()` decision. The source is development code
from this branch, **not** the V15 source ZIP. Filesystem caches may be warm; this was not a dedicated idle benchmark machine.

Local kernel latency is **0.072ms**, full solve **161.8ms**, and
model/history/process decision **9145.9ms**. These values must never be
interchanged. End-to-end latency remains a deployment blocker. No target
platform, production value network, production preflop chart, full-game GTO,
playing strength or improved win rate is certified.
