# Stage 2 Botzone-native run

Opt-in training config: [`configs/stage2/botzone_native.json`](../configs/stage2/botzone_native.json).
It does not change the encoder, the PPO clip/epoch/gamma defaults, the Pro Hunter
pool, or the v10 deployment guard. The new all-in mask defaults off for every
existing checkpoint.

## What the run changes

Voluntary all-in is removed from the legal mask before the softmax when all of
these hold:

- `botzone_native_allin_mask.enabled` is true (the config root and the `policy` block carry the same object; training refuses the run if they differ)
- SPR (`effective stack / pot`, the existing `CFRActionGuard.all_in_pot_ratio`) is strictly greater than 2.5
- the hand is not premium
- the all-in is not a forced call (stack already covered by the call, or all-in is the only legal action)

Premium hands:

- preflop: QQ+, AK suited or offsuit, or equity at least 0.72
- postflop: privately improved two pair or better (`strong_made_hand`), or equity at least 0.80

The illegal logit is the project sentinel `-1e9` (`ILLEGAL_LOGIT`), which the
softmax treats as probability zero. A literal `-inf` is not used: the existing
mask path documents that an all-`-inf` row becomes NaN under fp16. With the
flag on, the NumPy Botzone runtime applies the same snapshot mask before its
softmax, so this case does not depend on the v10 all-in-to-pot remap. The v10
guard still runs for its other fuses. With the flag off, observations match the
previous Stage 1 guard byte for byte.

The training pool keeps every current family (six scripted baselines, TAG_Pro,
Trapper_CR, and the fixed 10% reference / 15% history / 10% self quotas) and
adds three ladder replicas:

| Name | Behavior |
| --- | --- |
| `OffTree_Aggro_Boss` | Frequent preflop raises. Facing a 3-bet or 4-bet, a high chance of a 5-bet shove. Postflop, repeated bets at a sampled 0.35x–1.15x pot. Exact raise-to when the HUNL table accepts it; otherwise the nearest half-pot, pot, or all-in. |
| `Check_Shove_Trapper` | Mostly checks. Occasional cold all-in. Among shove attempts, 75% are value and 25% are bluffs; a mismatch checks instead of swapping modes. |
| `Flop_Raise_War_Bot` | Sees flops passively. On a flopped top pair, true two pair or better, or a strong draw, it raises and re-raises. |

In-training evaluation and `scripts/eval_stage2.py --opponents ...` use the same
duplicate 50-hand decks. `scripts/eval_pro_gauntlet.py --extra-opponents` scores
the replicas on that gauntlet's shared deck schedule and leaves champion
selection on the original script set.

## Initialization

This follows the Pro Hunter convention, not a resume of C200.

- `reference` is the fidelity student `runs/frozen_reference/policy.pt` (frozen backbone).
- `--warm-start` copies C200 residual weights only (`runs/frozen_stage2/update_0000200.pt`). Optimizer, PFSP, RNG and the update counter are new.
- The low-KL anchor is unchanged: `beta_kl` 0.05 for the early hands, annealing to `late_beta_kl` 0.03.
- Context stays the 14-field Pro Hunter HUD so the C200 warm start matches. The mask flag is not a weight; the destination config's flag wins, which is how a C200 checkpoint without the key can warm-start a masked run.

Budget: 250 PPO updates. `checkpoint_every` and `self_snapshot_every` are 50, so
`update_0000050.pt` through `update_0000250.pt` are the later champion-select
set. Evaluation is every 50 updates on the configured opponent list, including
the three replicas, and still stops if the reference retention gate fails.

## AutoDL launch

Use the machine's absolute paths. The config's `reference` and `strategies`
are the same fidelity-student and frozen-teacher locations as Pro Hunter.
Do not upload anything to Botzone from this command.

```bash
cd /root/autodl-tmp/poker   # repository root on the AutoDL disk
export PYTHONPATH=src
export OMP_NUM_THREADS=1
python scripts/train_stage2.py \
  --config configs/stage2/botzone_native.json \
  --warm-start /root/autodl-tmp/poker/runs/frozen_stage2/update_0000200.pt \
  --out-dir /root/autodl-tmp/poker/runs/stage2_botzone_native \
  --device cuda \
  --updates 250 \
  --experiment-kind formal \
  --experiment-name stage2_botzone_native
```

`--updates` overrides the config budget if a shorter probe is required. The
output directory must be new. `initial.pt` is update 0 after the warm start.
Passing checkpoints are also copied to `retained.pt`. A retention failure
leaves a diagnostic `train.pt` and stops.

Score a saved checkpoint later, on CPU or GPU, against the ladder replicas
and the baselines. Deals are the evaluator's fixed duplicate schedule:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python scripts/eval_stage2.py \
  --checkpoint /root/autodl-tmp/poker/runs/stage2_botzone_native/update_0000250.pt \
  --output /root/autodl-tmp/poker/runs/stage2_botzone_native/eval_u250.json \
  --device cuda \
  --opponents random,call,maniac,overbet,nit,equity,TAG_Pro,Trapper_CR,OffTree_Aggro_Boss,Check_Shove_Trapper,Flop_Raise_War_Bot,reference
```

Same-deck gauntlet cells, without changing Pro Hunter champion selection:

```bash
PYTHONPATH=src python scripts/eval_pro_gauntlet.py \
  --manifest /absolute/candidates.json \
  --out-dir /absolute/gauntlet_native \
  --extra-opponents OffTree_Aggro_Boss,Check_Shove_Trapper,Flop_Raise_War_Bot \
  --device cuda
```

Export still goes through `scripts/export_stage2_agent.py` / `tools/pack_botzone_stage2.py`.
A checkpoint trained with the mask stores `botzone_native_allin_mask` in its
policy settings, and the NumPy agent reads that block. This document does not
upload a bot.
