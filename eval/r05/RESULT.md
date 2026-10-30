# R05 result

Status: **INCOMPLETE**. CPU costs were measured. The GPU probe was not run. The budget proposal is ready for review and is not approved.

Implementation commit is recorded in `result.json` after this file is committed.

## What was measured

Host: cloud VM, Intel Xeon, 4 logical CPUs, one process, one thread. PyTorch `2.15.0+cpu`. `torch.cuda.is_available()` is false. CUDA was not used. GPU hours used: 0.

| Probe | n | Wall time | Rate / size |
| --- | --- | --- | --- |
| A5 kernel play (`botzone_a5_v2`) | 200 hands, 587 decisions | 0.366 s | 1605 decisions/s |
| A6 kernel play (`botzone_a6_v2`) | 200 hands, 660 decisions | 0.386 s | 1711 decisions/s |
| Cold import of kernel + schemas | 1 subprocess | 0.089 s | import only |
| Copy of frozen V6 npz | 6,060,293 bytes | 0.0026 s | rollback headroom assumed at 3 copies = 18,180,879 bytes |
| Peak RSS | VmHWM | 234,188 kB | this process only |

Teacher labeling was not timed. The snapshots are not in this tree, and no labels were invented.

## What is blocked

`training_gpu_budget` is `UNREGISTERED`. Peak VRAM and GPU step time are `null`, status `PENDING_USER_APPROVED_PROBE`. The request, not a cap, is 20 minutes on a GPU the user names, limited to one forward of the existing Stage2 policy at batch 32 with no optimizer step.

`budget_proposal.json` does not set `total_gpu_hours_cap`. CPU replay estimates for 1,000 decision states are derived from the measured per-decision time. Teacher, distill, PPO, solve, and final-acceptance GPU rows have null hour caps. The registered 200 paired segments / 20,000 hands were not reduced.

`experiment_registry.json` keeps the R01 seeds `101`, `202`, `303`, dev seed `404`, smoke seed `7`, the four primary opponent families, the pre-training thresholds, and the final-set access rule. FrozenHistory internal weights stay unlocked. The one-dev-look selection cap is a proposal, not a freeze.

Judge SHA256 remains `b70702fdcd28333f91087f1a3dc3ede1ca59737404622c64dc6dbf7e9206e544`.

pytest: `python3 -m pytest -q --override-ini='addopts=' --tb=line tests/test_r05_budget.py` → 3 passed.
