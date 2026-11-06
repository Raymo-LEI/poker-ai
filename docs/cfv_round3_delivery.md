# Round 3 delivery-first handoff

Run the supervisor against the **existing** Round 3 directory. It adopts the current label and turn-fit processes, replaces only that run's queued post-gate continuation and arena, and uses the relabel launch receipt as the start of the original budget. Resume by running the same command from the same locked commit. Start the handoff before the training cutoff so that the final three hours are available for evaluation.

```bash
python -u scripts/sprint24/round3_delivery.py \
  --round3 /root/autodl-tmp/sprint24/cfv_turnround3_8014688 \
  --max-hours 24 --expected-commit "$COMMIT_SHA"
```

The supervisor records stage changes and 30-minute heartbeats in `delivery_first/STATUS.json`. The cutoff preserves the completed relabel shards and completed candidate receipts, and writes `delivery_first/BUDGET_CHECKPOINT.json`. The current training driver saves weights at candidate boundaries.

Generation 1 combines the frozen U250 blueprint with the selected turn and river pair. It evaluates exact staged package bytes against the verified V22 archive over 100 duplicate seat-swapped pairs per arm. A, C and D use the existing locked profiles. The first delivery uses V26–V28 when available. If another release has consumed that range, the shared counter allocates the next free range.

When a CFV gate fails or the fresh turn gate is still pending at cutoff, the fallback considers only real, completed, teacher-matched 50-epoch candidates with bound gate receipts. It picks the lowest recorded turn ratio among available compatible candidates. Holdout depths are recorded with each alternative. Diagnostic packages carry the failed value and keep V22 as the deployment recommendation. The gate thresholds and one-shot consumption are controlled by the existing turn process.

After the first three packages have been published, the supervisor may start one existing policy-distillation generation if the paired CFVs passed and time remains before the training cutoff. An accepted generation receives a separate arena and three new version numbers. Partial arenas retain completed match receipts and their counts and cannot promote a package. Exact-byte runtime qualification and statistically significant complete-pair improvement are required for promotion. The supervisor writes `delivery_first/FINAL_DELIVERY.json` and does not upload to Botzone.
