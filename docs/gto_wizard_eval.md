# GTO Wizard eval scaffold

This is an offline prep tool. It compares a recorded action distribution with
another recorded distribution on the same A5 or A6 buckets and writes `kl`
and `l1` into `report.json`. It does not train, does not read AutoDL outputs,
does not upload a Botzone bot, and does not submit a leaderboard row.

Botzone is not the external benchmark for this path. The local SD-CFR checks
in [cfr_evaluation.md](cfr_evaluation.md) and the duplicate-deal matrix plus
optional Slumbot client in [pro_gauntlet.md](pro_gauntlet.md) stay as they
are. This script does not call either of them.

## Action buckets

`botzone_a5_v2` is fold, check/call, half-pot, pot, all-in. `botzone_a6_v2`
appends `quarter_pot`. Those names are the Route 1 labels. Chip amounts come
from the standard 200BB `HunlTable` (50/100 blinds, 20,000 chips, reset every
hand). Sized bets use that table's floor pot fraction. Quarter-pot adds
`pot // 4` and is legal only when that raise is legal on the table. An
illegal id or an off-tree amount fails closed and is not snapped to a
neighboring size.

`SnapshotPolicy` checkpoints are refused. The fixture distributions are JSON.
Nothing in this command opens `iteration_*.pt`.

## Dry-run

Request a researcher key at <https://benchmark.gtowizard.com/> if you want one
later. The API only plays hands; it does not expose a solver. The public
client repository does not publish a license, so this tree does not vendor
that source. See `eval/gto_wizard/NOTICE.md`.

The supported command does not use the network and does not need
`GTOWIZARD_API_KEY`:

```bash
python3 scripts/eval_gto_wizard_offline.py \
  --fixture eval/gto_wizard/fixtures/spots.json \
  --out-dir runs/gto_wizard_fixture
```

`report.json` fields:

| Field | Meaning |
| --- | --- |
| `hands` | Number of fixture spots compared. Not a dealt sample. |
| `kl` | Mean natural-log KL of the policy distribution against the reference. |
| `l1` | Mean L1 distance between those distributions. |
| `same_bucket_count` | Spots whose argmax bucket matches. Ties keep the lower action id. |
| `chips` | `null`. Not measured. |
| `aivat_bb_per_100` | `null`. Not measured. |
| `live` | `false`. |
| `leaderboard_submitted` | `false`. |
| `training_modified` | `false`. |
| `botzone_used` | `false`. |

A same-bucket miss still exits 0. The file is a measurement of the fixture,
not a pass/fail strength gate, and it is not a live GTO Wizard result.

A second small command compares the built-in synthetic snapshot for
`a6_flop_cbet` with that fixture. It also does not read a teacher directory:

```bash
python3 scripts/compare_teacher_fixture.py \
  --fixture eval/gto_wizard/fixtures/spots.json \
  --out-dir runs/gto_wizard_fake_snapshot
```

## Live

On the fixture command, `--live` is off unless you pass it. Without
`GTOWIZARD_API_KEY` the process exits 2 and does not open a socket. With the
variable set it still exits 2 and does not send a hand: live play is not a
gate for the offline script. `eval/gto_wizard/client.py` holds the `/hands`
and `/hands/{id}/act` request shape (`f` / `k` / `c` / `b` plus an integer
amount). The key stays in the environment, a state-changing act is not
retried, and this repo does not post to the leaderboard.

## Live smoke runner

`scripts/eval_gto_wizard_live.py` is the opt-in runner for that request
shape. It loads a frozen Stage 2 residual checkpoint through
`alphanl.stage2.train.load_checkpoint` and maps the 5-action head onto
`f/k/c/b` with the same `HunlTable` sizes as the fixture adapter. An abstract
size the table rejects, or a size outside the API `raise_range`, stops the
session instead of snapping to a neighbor. `model_structure/*.json` is not an
input and is not used to build another network.

Weights are not in this repo. A missing `GTOWIZARD_API_KEY` exits 2 before
any socket. The cap is 1000 hands. The command posts only `/hands` and
`/hands/{id}/act`. It does not call `/leaderboard`, and it does not retry an
act after a timeout or any other failure. `report.json` records
`live: true`, `leaderboard_submitted: false`, `training_modified: false`, and
`botzone_used: false`. Chip and AIVAT fields are filled only from values the
hand payload returned. A partial file is rewritten after each completed hand.

```bash
export GTOWIZARD_API_KEY=...
python3 scripts/eval_gto_wizard_live.py \
  --policy "$HOME/Documents/ChatGPT/Poker/botzone_native_v2_20261030/weights/u250_retained.pt" \
  --config "$HOME/Documents/ChatGPT/Poker/botzone_native_v2_20261030/weights/config.json" \
  --hands 1000 \
  --out-dir runs/gto_wizard_live_u250
```

Do not put the key in the command line. The sibling `config.json` is checked
against the checkpoint's all-in mask when that block is present. It does not
replace the checkpoint architecture. This command does not produce a
leaderboard row, and a report written here is not a claimed live score until
you run it with your own key.

## R08 handoff

`eval/r08/` is a Teacher handoff stub. `PLAN.md` and `RESULT.md` stay
`NOT_RUN` / `PENDING_TEACHER`. The scripts hash and schema-check
`manifests/export_from_teacher.example.json`, refuse `--execute`, and do not
invent a win rate. `distill_gate.py` prints a `scripts/distill_cfr.py`
command and exits 2 without running it. The example manifest leaves
`teacher_out_dir` empty and only names `eval/r07/manifests/a6_seed303.json`
as a hint. This branch does not modify that R07 file.
