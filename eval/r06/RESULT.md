# R06 result

Status: **INCOMPLETE**. Kuhn correctness and checkpoint restore ran on CPU. The A5 and A6 GPU pilots were not started.

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

`launch_pilot.py --execute` exits 2 and starts nothing. The production regret trainer is still the frozen five-action head, so running it would not be an A5 or A6 pilot. AutoDL still needs the parent after that head is bound to the schema. This dispatch did not SSH and did not debit GPU time.

No candidate was selected.

Judge SHA256 remains `b70702fdcd28333f91087f1a3dc3ede1ca59737404622c64dc6dbf7e9206e544`.
