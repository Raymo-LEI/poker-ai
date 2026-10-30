# R06 result

Status: **INCOMPLETE**. Kuhn correctness and checkpoint restore ran on CPU. The A5 and A6 GPU pilots were not started.

Implementation commit: `14fe4b9dc4384a681b5d230b19ba363bb7de3d87`.

GPU hours used: **0**. The registered cap stays 1000. No strength gain is claimed.

## Kuhn (exact, not Hold'em)

Tabular external-sampling CFR, 1000 iterations, seed 1, 6 traversals per player.

- Exploitability fell from the uniform prior to about 0.0158.
- Player-zero value is about -0.055, against the equilibrium value -1/18.
- Payoffs are zero-sum in ante/bet units (fold wins 1, called bet wins 2).
- The check-bet infoset is weighted by that player's own check probability, not by the opponent's reach.
- Deep CFR approximation is a separate two-iteration CPU regret fit on a two-node toy game. It is not a Hold'em convergence result.

## Snapshot

A CPU checkpoint saved after iteration 1 restores to the same iteration, the strategy snapshot's iteration equals its weight, both seats have samples, and the resumed iteration matches a trainer that never stopped. Losses were finite.

## Pilots

`manifests/a5.json` and `manifests/a6.json` are isolated. A6 still requires `quarter_pot` and width 6. Each asks for at most 0.25 GPU hours, one job, under the 1000 hour cap.

`launch_pilot.py` dry-run and `--execute` now build an ActionSpec regret head. A5 is width 5 and A6 is width 6 with `quarter_pot`. A `botzone_5action_v1` checkpoint is rejected instead of reshaped. On this CPU host `--execute` runs one finite backward step and does not start CUDA. The 4090 commands for the parent, one job at a time, are:

```text
python3 eval/r06/launch_pilot.py --manifest eval/r06/manifests/a5.json --execute
python3 eval/r06/launch_pilot.py --manifest eval/r06/manifests/a6.json --execute
```

Each manifest caps itself at 0.25 GPU hours. This dispatch used 0.

No candidate was selected.

Judge SHA256 remains `b70702fdcd28333f91087f1a3dc3ede1ca59737404622c64dc6dbf7e9206e544`.
