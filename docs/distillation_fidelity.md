# Distillation fidelity and playing-strength acceptance

The fidelity pipeline supplements `distill_cfr.py`; the legacy collector and checkpoint format remain compatible. It addresses finite single-profile labels, uniform-opponent coverage, late-street loss imbalance, and final-step-only selection. It does not assume that rare teacher actions are noise or that lower static loss guarantees playing strength.

## Exact labels

For the frozen inclusive window T=150..350, every retained profile contributes:

```text
posterior(t | I) ∝ t × product of sigma_t(a | J) over the hero's earlier decisions J
target(I) = sum_t posterior(t | I) × sigma_t(I)
```

Opponent and chance reach are not multiplied again. Parents and chosen actions retain the full hero prefix, even if the network token sequence truncates history. All-zero teacher reach is explicitly flagged and excluded from fitting; it is never replaced by a naive local arithmetic average. Labels are exact with respect to the archived networks in FP32 inference, with FP64 reach accumulation. See the [SD-CFR average-strategy construction](https://arxiv.org/abs/1901.07621) and [Deep CFR](https://proceedings.mlr.press/v97/brown19b.html).

`audit_distillation_fidelity.py` replays a fixed prefix of the existing held-out six-bot student benchmark, checks every deck seed and payoff, labels those states with the full mixture, and reports teacher-to-student KL, argmax agreement, entropy, total variation and near-zero-action leakage by street, pot and opponent. Audit states are separate from training and selection data.

## Data and loss

Twenty percent of collected hands retain uniform guarded-legal opponent coverage. Other hands use unjittered random, calling-station, maniac, overbet, nit and equity bots. Teacher profiles stay fixed for complete hands. Some rule-opponent hands use the old student to expose its reached states; all labels still come from the exact teacher. Teacher and student actions never inspect opposing private cards.

Training and validation split by complete, disjoint hand ids. Their 128-bit deck-seed space is separate from the 64-bit duplicate-deal selection/acceptance seeds. Archive fingerprints are validated, and reusable datasets carry the complete teacher manifest.

The objective is state-weighted forward KL. Turn receives 2x, river 3x, and pots greater than 20 BB receive another 2x, capped at 4x combined. Thus a late large-pot state receives at most 4x, not 16x. The raw exact target remains available for auditing. Target pruning below 0.03 is an explicit candidate intervention, compared with no pruning; masks and at least one action are preserved.

## Selection and inference interventions

Fixed development duplicate deals against calling station and equity select `policy_best.pt` during fitting. Among candidates satisfying the development Call/Equity targets, their mean playing return determines selection; otherwise the best development mean is retained. Static held-out KL is recorded as a diagnostic, not the checkpoint-selection objective. The full six-bot acceptance set is evaluated only after selection.

Temperature 1.0, 0.90 and 0.85 are tested. A selected constant temperature is absorbed into the policy-head weights and bias, so the saved plain `PolicyNetwork`, Stage 2 frozen reference and portable inference use the same probabilities. Output cutoff 0.05 and temperature-plus-cutoff are tested as diagnostics; hard cutoff is not silently introduced into the Stage 2 action support.

## Acceptance and conditional Stage 2 start

Each standard opponent receives exactly the teacher's recorded 2,500 duplicate pairs / 5,000 hands, with fingerprinted raw logs and per-pair deck verification. Report paired student-minus-teacher intervals and all six individual returns. The user-requested point-estimate gate requires all three:

- Six-opponent equal-weight mean greater than +2,000 mbb/hand.
- Calling station greater than +1,800 mbb/hand.
- Standard equity greater than -3,300 mbb/hand.

Failure leaves Stage 2 paused and preserves all reports and candidates. No acceptance claim follows from a development result. Passing writes a separate accepted `policy.pt`, checks zero-residual reference parity, and may start Stage 2 only when both `--stage2-template` and `--stage2-out` were explicitly supplied. The output must be new; no active Stage 2 trainer may coexist. A fresh run uses alpha 1.0, beta 0.05 / 0.03 and the explicit finite update budget. No prior residual, optimizer, PFSP or HUD state is reused.

```bash
PYTHONPATH=src python scripts/train_distillation_fidelity.py \
  --strategies /absolute/frozen/strategies --reference /absolute/old/policy.pt \
  --teacher-report /absolute/teacher/report.json --out-dir /absolute/new/fidelity \
  --train-hands 20000 --validation-hands 3000 --steps 2000 --check-every 500 \
  --selection-pairs 500 --candidates warm_pruned,warm_raw --device cuda
```

Omitting the Stage 2 paths makes this a distillation-and-acceptance experiment only. Existing artifacts are never overwritten. CPU/synthetic tests establish contracts and math, not the requested real playing-strength recovery.
