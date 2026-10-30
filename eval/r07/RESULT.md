# R07 result

Status: **INCOMPLETE**. The formal Teacher was not trained. No seed was selected. Distill is not authorized.

Plan commit: `51638a4aa99a7fd5990cc874b4021f6168c38a37`.

GPU hours used by this task: **0**. Registered total stays `0.0003791426171341704` of the 1000 hour cap. Remaining stays `999.9996208573829`. The cap was not raised.

## What this host did

`launch_teacher.py --write-plan` wrote six isolated manifests. A6 is the primary and keeps `quarter_pot` at width 6. A5 is the control at width 5. Seeds 101, 202, and 303 are each listed. Milestones are T=50/100/200/400/800. T=1000 is not scheduled.

Loss iteration weight power is 1. The strategy mixture is `iteration * own_reach`. A t² loss is not treated as a t² average. Burn-in iterations 1..49 are frozen out of scoring. Each run asks for a new reservoir. The production width-5 `RegretNetwork` / `SDCFRTrainer` path is forbidden: it is `botzone_5action_v1` and would drop `quarter_pot` and the A5 size table.

Dry-run and `--execute` exited 0 for `a6_seed101` and `a5_seed101`. The smoke head matched the schema width, rejected a `botzone_5action_v1` checkpoint, and did not start CUDA. That smoke network is not the formal Teacher body. Pytest: `eval/r07/pytest_r07.txt` (2 passed).

Learning curves, dev selection, holdout, weakness slices, certified river raw Fold, and A6-versus-A5 cost/benefit are **not measured**. Empty numbers were not filled in. LBR/ABR is not a full-game exploitability bound. Thin evidence is not a reason to train longer automatically.

## Parent (德扑训练) on the RTX 4090

This VM did not SSH to AutoDL. Formal iterations are still `PENDING_AUTODL_PARENT`.

On a machine where CUDA is visible, the same `--execute` **exits 2 and does not start a job**. That guard is so this cloud VM cannot spend GPU time. Do not point `scripts/train_cfr.py` at these manifests.

If a later schema-bound runner is used, the authorized serial order (`max_concurrent_training_jobs=1`) and the stop guards are:

```text
python3 eval/r07/launch_teacher.py --manifest eval/r07/manifests/a6_seed101.json --execute
python3 eval/r07/launch_teacher.py --manifest eval/r07/manifests/a6_seed202.json --execute
python3 eval/r07/launch_teacher.py --manifest eval/r07/manifests/a6_seed303.json --execute
python3 eval/r07/launch_teacher.py --manifest eval/r07/manifests/a5_seed101.json --execute
python3 eval/r07/launch_teacher.py --manifest eval/r07/manifests/a5_seed202.json --execute
python3 eval/r07/launch_teacher.py --manifest eval/r07/manifests/a5_seed303.json --execute
```

Hour caps, not measured iteration times:

- Each seed stops at **6 wall hours**, or sooner if the first iteration exceeds **1 hour**.
- Milestones 50/100/200/400/800 are checkpoints inside that one seed wall. There is no separate measured hour budget per milestone.
- T=1000 is not scheduled (`optional_t1000_hours=0`).
- Six seeds at the 6 hour stop is **36 hours** worst case. That is under the remaining budget and is not permission to spend the 400 hour soft hint.
- Debit train, label, and eval separately. Distill stays 0 until a Teacher passes the hard gates.
- Dev opponents stay Equity / TAG_Pro / Trapper_CR / FrozenHistory at 0.25 each. FrozenHistory internal weights are still unlocked. Dev seed 404. Final set is not for selection.
- Registered strength gates (not claimed): noninferiority lower bound > -20 mbb/hand, primary lower bound > 0, point estimate at least 20 mbb/hand, Bonferroni n=5. Certified river raw Fold ≤ 1% is not measured.

Judge SHA256 remains `b70702fdcd28333f91087f1a3dc3ede1ca59737404622c64dc6dbf7e9206e544`.
