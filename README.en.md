[中文](README.md) · **English**

# poker-ai

Research and engineering repository for heads-up no-limit Texas Hold'em. The tree currently contains standard 200BB SD-CFR, Transformer PPO, a 50-hand residual Stage 2, Botzone rule and NumPy inference bundles, and the Route 1 action contract.

**Status: training, rule diffs, checkpoints, export, and CPU smokes are wired up. The repository does not ship a 200BB strategy that has been trained for a long run and accepted in an independent match, and it has no Botzone rank or GTO Wizard leaderboard result.** A passing test establishes the contract and the wiring it covers. Round results live in `RESULT.md` under `eval/r02`–`eval/r06` and `eval/r08`. `eval/r07` is not in this tree.

Most of the linked notes under `docs/` are written in Chinese.

## Three tables

The three paths each carry their own blinds, action order, and action width. Read a checkpoint, a feature version, and a metric against the table that produced it.

| Path | Table | Actions |
| --- | --- | --- |
| SD-CFR, and the Stage 2 reference that reads it | Standard 200BB: 20,000 chips each, blinds 50/100, reset every hand | Five abstract actions. Stage 2 takes its feature version and network width from the reference checkpoint |
| Transformer PPO | `rlcard`, `pokerkit`, or a separate Botzone backend | Five bins. A checkpoint from one backend does not load into another |
| Route 1 (from R02) | The unmodified official Botzone referee, `Holdem.player_action` | `botzone_a5_v2` is fold / check-call / half-pot / pot / all-in. `botzone_a6_v2` appends `quarter_pot`. The old `botzone_5action_v1` is only the V6 contrast; a schema-hash mismatch refuses the load |

Standard HUNL, the older `rlcard` environment, and the early Botzone simple fixture in this repo also differ from one another. Kernel diffs from R02 onward are against the official referee; see the [R02 result](eval/r02/RESULT.md).

## Stage 1: standard 200BB SD-CFR

A [PokerKit](https://github.com/uoftcprg/pokerkit) table, five abstract actions, regret matching, external sampling, an independent reservoir per seat, iteration-weighted regression, historical strategy snapshots, and checkpoint resume. Sampling advances several trees in one process and replaces a finished tree with a new root immediately. Rules: [HUNL environment](docs/hunl_environment.md). Algorithm: [SD-CFR notes](docs/sd_cfr.md).

Pieces that are wired up and have their own notes:

- An all-street deep-stack all-in guard, and shared position-conditioned training: [SPR guard](docs/cfr_spr_guard.md), [unified SD-CFR](docs/cfr_unified.md).
- The T=150→200 budget comparison, a shared training probe, and same-iteration evaluation: [Stage 1 selection](docs/cfr_stage1_selection.md).
- An optional full profile: equity and pot-odds inputs, a 1.49M model, one 8M reservoir per seat, polynomial regret regression, and a 15% fixed-opponent root mixture. CPU wiring and loss diagnostics on the frozen T200 snapshots are in the tree. This profile has not been deployed or accepted on CUDA. See [full profile](docs/cfr_fullupgrade.md).
- The RTX 4090 config turns on CUDA BF16, fused Adam, and TF32 explicitly. Measurement commands: [server efficiency](docs/server_4090_efficiency.md). Throughput stages: [P0–P3](docs/cfr_performance_rollout.md). No 4090 timing in this repo is written up as a published strategy-quality result.
- While writing a regret sample, external sampling also records the current strategy when the buffer provides `add_strategy`. The weight is `iteration * traverser_reach`. Parallel workers place that strategy and weight in the same shared slot as the regret. The parent keeps the rows when the buffer provides `strategy_sink`, so a later strategy fit can reuse the traversal. See `src/alphanl/cfr/traversal.py` and `src/alphanl/cfr/parallel.py`.

Paired matches, 50-hand matches, and self-play health checks go through `scripts/eval_cfr_readiness.py`. `scripts/eval_cfr_periodic.py` freezes a strategy when a new milestone arrives, skips work that already finished, and can resume after an interrupt. Config and systemd examples: [offline and periodic evaluation](docs/cfr_evaluation.md).

## Average-strategy distillation

Offline compression: [distillation prep](docs/cfr_distillation.md). It covers frozen-archive checks, own-reach-preserving trajectory labels, one policy student, held-out diagnostics, and a portable checkpoint. Playing strength, Stage 2, and Botzone acceptance are later gates.

[Exact-mixture fidelity distillation](docs/distillation_fidelity.md) adds in-window posterior labels, street-and-pot weighted forward KL, development-set checkpoint selection, and a six-opponent duplicate-deal strength bar. CPU and synthetic tests lock the contract and the arithmetic. Stage 2 stays paused when the bar is missed. After a pass, a new Stage 2 run starts only when both `--stage2-template` and `--stage2-out` are supplied.

## Transformer PPO

A betting-history token encoder, policy and value heads, self-play, and PFSP. Backends are `rlcard`, `pokerkit`, and the Botzone rules. A convolution control, behavior cloning, and an adapter for the older [AlphaNLHoldem](https://github.com/bupticybee/AlphaNLHoldem) baseline are included. See [Transformer path](docs/step3_transformer.md), [controls and baseline](docs/bc_and_baseline.md), and [Botzone milestones M1–M12](docs/botzone_transformer_milestones.md).

Configs live in `configs/transformer/`. Resume: [training notes](docs/training.md). After a killed process on Windows, `tools/train_autoresume.ps1` restarts only from a fully written PPO checkpoint. See [autoresume](docs/train_autoresume.md).

`src/alphanl/exploiter/` is the earlier 50-hand residual prototype. It reaches CPU wiring and a training smoke. The current 50-hand adaptation is Stage 2, in the next section. Prototype notes: [exploiter](docs/exploiter.md).

## Stage 2: 50-hand residual PPO

A frozen Stage 1 student is the reference. The residual policy trains with PPO on public action statistics across a 50-hand match. The shared input contract, PFSP, single-file NumPy export, and CPU smoke are in the tree. These entry points do not start a GPU run by themselves. See [Stage 2 handoff](docs/stage2_match_residual.md).

Later optional layers each have their own switch and note:

- Four-layer MatchPool, match-locked style jitter, and a fixed 35% defensive quota: [MatchPool](docs/stage2_matchpool.md).
- TAG / check-raise probes, frozen-policy Slumbot evaluation, and cross-generation duplicate deals: [Pro Gauntlet](docs/pro_gauntlet.md) and the [2026-10-29 report](docs/pro_gauntlet_report_20261029.md). External candidates for a future training pool: [open-source opponents](docs/open_source_stage2_opponents.md).
- TAG/Trapper training families, a trap HUD, C200 weight-only warm start, and residual-output blends: [Pro Hunter](docs/stage2_pro_hunter.md).
- `configs/stage2/botzone_native.json`: a pre-softmax voluntary all-in mask and three ladder-replica opponents. With the flag off, Pro Hunter weights and the v10 guard stay as they are. See [Botzone-native](docs/stage2_botzone_native.md).
- The deployment guard is stored in `meta_json.deployment_guard` inside `stage2_model.npz`. Old weights with no such field leave the guard off. See [v10 guard](docs/botzone_v10_guard.md).

The NumPy runtime calls `alphanl.stage2.wizard.fuse` before it emits action probabilities. The 1.49M blueprint weights are read only. With no `vleaf.*` leaf head, the function returns the blueprint probabilities unchanged (`wiz=0`). With a leaf head, preflop still returns the blueprint. Postflop search runs when the pot is at least 6BB or the hero is facing a bet. Search uses 8 range buckets, 2 runouts, and 4 CFR iterations, then mixes the result back into the blueprint. An exception falls back to the blueprint (`wiz=err`). Search is over the five action ids, and it is exported with the Stage 2 source bundle. The contract test is `tests/test_wizard_solver.py`. There is no separate strength report for this hook.

## Botzone bundles

| Bundle | Role | Notes |
| --- | --- | --- |
| `botzone_bots/rulebot.zip` | Rule bot | [Upload notes](docs/botzone_rulebot_upload.md) |
| `botzone_bots/model.zip` | Torch-free NumPy forward | Structure was checked with a development random initialization. The repo does not include trained weights that are ready to play. [Upload notes](docs/botzone_model_upload.md) |
| `botzone_bots/stage2.zip` | Stage 2 forward for Python 3.6 + NumPy. Weights go in user storage as `stage2_model.npz` | Upload steps, and an earlier single-core, 256MB, full 50-hand record, are in the [upload notes](docs/botzone_stage2_upload.md). The current bundle includes the blueprint subgame hook. With no leaf head, actions still come from the blueprint. That recorded match predates v10 and this hook |

Pack with `tools/pack_botzone_rulebot.py`, `tools/pack_botzone_model.py`, and `tools/pack_botzone_stage2.py`. The v10 guard is written at Stage 2 pack time with `--v10-guard` or `--guard-config`. Split the outer delivery ZIP first: upload the inner source ZIP as code, and upload `stage2_model.npz` to user storage.

## Route 1: referee kernel and the A5/A6 contract

This line pulls training and evaluation onto the official referee and an explicit action schema. Status for R02–R06 and R08 is taken from each round's `RESULT.md`.

| Round | Status | What is in the tree |
| --- | --- | --- |
| [R02](eval/r02/RESULT.md) | PASS, 0 GPU hours | A branchable kernel steps with the unmodified official referee. Recorded diffs are 0 for legal/illegal decisions, next actor, pot, stacks, street, and showdown payoff |
| [R03](eval/r03/RESULT.md) | PASS, 0 GPU hours | Four-layer attribution and counterexamples on one V6 deployment pack. The teacher snapshots and the frozen student `policy_best.pt` are not included; those two layers are marked MISSING |
| [R04](eval/r04/RESULT.md) | PASS, 0 GPU hours | A5 and A6 share rules, public features, and a zero training budget. A6 only adds `quarter_pot`. The boundary measurement reports `quarterpot_effective_new_coverage = 0.224`. The result file states that this does not establish a small-bet capability |
| [R05](eval/r05/RESULT.md) | INCOMPLETE | CPU timings exist for kernel play, import, and copy. The GPU probe was not run, the budget proposal is not approved, and `total_gpu_hours_cap` is still empty |
| [R06](eval/r06/RESULT.md) | INCOMPLETE | Kuhn correctness and checkpoint restore ran on CPU. The A5/A6 GPU pilots were not started. On a CPU host, `eval/r06/launch_pilot.py --execute` takes one finite backward step and does not start CUDA |
| R07 | Not in this tree | The monitor can read `learning_curve_live.json` from a run directory. `eval/r07` and its manifests are not in the repository |
| [R08](eval/r08/RESULT.md) | NOT_RUN | Teacher remains PENDING_TEACHER. `distill_gate.py` prints a command and exits 2. Distillation, SD-CFR, and PPO are not started from here |

The GTO Wizard comparison is an offline fixture plus a live smoke that needs a key. The offline command does not use the network and does not read a teacher directory:

```bash
python3 scripts/eval_gto_wizard_offline.py \
  --fixture eval/gto_wizard/fixtures/spots.json \
  --out-dir runs/gto_wizard_fixture
```

`kl` / `l1` are distances between the fixture distributions. Chips and AIVAT are empty in the offline report. The live entry is `scripts/eval_gto_wizard_live.py`. Put the key in `GTOWIZARD_API_KEY`. It requests only `/hands` and `/hands/{id}/act`, and it does not submit a leaderboard row. The upstream client repository ships no license, so this tree does not vendor its source. See [GTO Wizard notes](docs/gto_wizard_eval.md) and `eval/gto_wizard/NOTICE.md`.

## Overwatch training console

Read-only over existing logs. It does not load a model. Python 3.10+ standard library is enough; training dependencies and npm are not required.

```bash
python3 scripts/monitor_training.py --runs-dir /absolute/path/to/runs --open-browser
```

The local page defaults to `http://127.0.0.1:6008`. Repeat `--runs-dir` for each run or for that run's direct parent. Discovery does not recurse into deeper children. Remote viewing, `--source-label`, `learning_curve_live.json`, and `?scope=all` (unlabeled runs are shown and stay non-formal) are in the [monitor entry](monitor/README.md). Workbench pages and Stage 2 log fields: [console notes](docs/training_monitor.md).

A training process reports its own identity with `--experiment-kind`: `formal`, `development`, `smoke`, `benchmark`, or `evaluation`. The console filters on that field in the logs. A directory with no label is not rewritten as formal.

On macOS, double-click `打开训练监控.command` or `连接服务器监控.command` in the repository root.

## Quick start

Run from the repository root on Python 3.10+. Install the CPU or CUDA PyTorch wheel that matches the machine. `requirements-cfr.txt` is the current CFR path, and it includes `requirements-transformer.txt`.

```bash
git clone https://github.com/Raymo-LEI/poker-ai.git
cd poker-ai
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-cfr.txt
export PYTHONPATH=src

# Check the config and the model size only
python scripts/train_cfr.py --dry-run

# Short CPU smoke: real-table sampling, one fit, one checkpoint
python scripts/train_cfr.py --config configs/cfr/hunl_200bb.json \
  --device cpu --iterations 1 --traversals 1 --train-steps 1 \
  --out-dir runs/cfr_smoke
```

The full training command is `python scripts/train_cfr.py --config configs/cfr/hunl_200bb.json`. Output includes `metrics.csv`, `performance.csv`, `strategies/`, `train.pt`, and `checkpoint_data/`. When moving a resume point, keep `train.pt` and `checkpoint_data/` together. Resume an existing directory with `--resume`.

The 4090 starting point is [hunl_4090.json](configs/cfr/hunl_4090.json). The multi-worker starting point is [hunl_4090_parallel.json](configs/cfr/hunl_4090_parallel.json).

The same environment can check the PPO config and the early exploiter prototype:

```bash
python scripts/train_transformer.py --config configs/transformer/laptop_8gb.json --dry-run
PYTHONPATH=src python scripts/train_exploiter.py --dry-run
```

`train_exploiter.py` looks for `runs/cfr_s2_ref/strategies` by default. When that directory is absent, the script reports that it is using an untrained uniform-strategy stand-in.

Stage 2 has no dry-run. `scripts/train_stage2.py` requires `--out-dir`, defaults to `configs/stage2/botzone_50hand.json`, and reads the reference checkpoint and strategy directory named in that config. The interface smoke is a separate command. It writes untrained synthetic teachers into a new directory, runs two quota blocks, then exports. The output directory must not already exist.

```bash
PYTHONPATH=src python scripts/smoke_stage2.py --out-dir runs/stage2_smoke
```

That smoke states that it checks interfaces and CPU execution only. Its output is not a Stage 1 or Stage 2 strength result.

The older AlphaNLHoldem stack is a separate Python 3.7 / TensorFlow 1.15 environment, with `requirements-alphanl.txt`. To fetch the upstream code or the `c_1048` weights:

```bash
git submodule update --init --recursive
```

The 50BB-capped Slumbot client is documented in [eval/SLUMBOT_CLIENT.md](eval/SLUMBOT_CLIENT.md). It is a different stack from the 200BB SD-CFR path above.

## Checks and layout

`pytest.ini` already sets `pythonpath` to `src`.

```bash
python -m pytest -q
PYTHONPATH=src python -m alphanl.cfr.kuhn --iterations 1000
```

Kuhn is a tabular external-sampling CFR toy, used to check sampling and an exploitability drop. Its numbers are not a Hold'em convergence result. R06 records one CPU run of 1000 iterations at seed 1.

- `src/alphanl/cfr/`: SD-CFR, reservoirs, parallel sampling, historical strategies, distillation, evaluation, and checkpoints.
- `src/alphanl/transformer/`: table adapters, observation encoding, PPO, PFSP, and the model.
- `src/alphanl/botzone/`: official-referee kernel, legal responses, the A5/A6 schema, and the rule bot.
- `src/alphanl/stage2/`: the 50-hand environment, residual policy, opponent pool, export, deployment guard, and blueprint subgame search.
- `src/alphanl/exploiter/`: the earlier 50-hand residual prototype.
- `scripts/`: training, evaluation, benchmarks, the monitor, and smokes.
- `configs/cfr/`, `configs/transformer/`, `configs/stage2/`: configs for those paths.
- `eval/`: the Slumbot client, the GTO Wizard scaffold, and the R02–R06 and R08 results and manifests.
- `botzone_bots/`, `tools/`: uploadable entries and packers.
- `docs/`: rules, algorithms, training, and deployment notes. The [survey](docs/reference_survey.md) records the early baselines.
- `README.md`: the Chinese edition of this page. The link at the top switches to it.
- `monitor/`: the console's own notes. Frontend sources are in `tools/training_monitor/`.

## License and third-party code

The repository root currently has no `LICENSE` file. `third_party/AlphaNLHoldem` is its own Git submodule. When using that code or the other dependencies, read the upstream license for each project. The public GTO Wizard client likewise has no license shipped with this repository, so its source is not vendored here.
