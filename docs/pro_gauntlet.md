# Independent Pro Gauntlet

This is an evaluation-only addition. It does not change training defaults,
PFSP weights, stop conditions, checkpoints or any running experiment. The two
new scripts are style probes, not demonstrated Botzone ladder professionals.
A confidence interval containing zero does not prove parity or profitability.

The [implementation and evaluation report](pro_gauntlet_report_20261029.md)
includes the five-candidate 100,000-hand matrix, evidence fingerprints,
selection outcome and proposed next experiments.

## Local tournament

Copy `configs/cfr/pro_gauntlet.example.json` and point it at immutable local
artifacts. Paths resolve relative to the manifest, not the current directory.
Required missing candidates stop the run with a partial report; optional
missing candidates are listed explicitly. Existing output directories are
never overwritten.

```bash
python scripts/eval_pro_gauntlet.py \
  --manifest configs/cfr/pro_gauntlet.example.json \
  --out-dir runs/pro_gauntlet_seed1 --pairs 2500 --seed 20261030
```

Use `--pairs 1000` for the smaller full budget, or `--pairs 100` for plumbing
smoke only. Every count must be divisible by 50 and at least 100. CPU inference
with one Torch thread is the default; no remote compute is launched.

Each candidate faces the unchanged six standard bots (`random`, `call`,
`maniac`, `overbet`, `nit`, `equity`) and `TAG_Pro`, `Trapper_CR`. Every pair of
candidates also plays a direct match. Six candidates produce 48 scripted
matchups and 15 direct matchups: 315,000 hands at 2,500 duplicate deals per cell.
The mirrored round-robin cell is derived by exact zero-sum negation; it is not
a second independent measurement. `reference` is represented by direct games
against Candidate B rather than an extra vote for the same model.

Outputs are `report.json`, `matrix.csv`, `matrix.html`, `deck_seeds.json`, and
one flushed JSONL per actual matchup. JSON contains all pair totals, per-leg
behavior audits, source hashes, artifact hashes, exact teacher window and
checkpoint update identities. The HTML matrix is self-contained and escapes
candidate names. Raw logs identify shared deck seeds and sampled teacher
iterations. Common-deal candidate gains against each script have paired CIs.

The common deck schedule is shared across *all* cells. Each block consists of
50 hands with alternating positions, played again with identities swapped.
Both actors receive their own perspective on every public action and their
own signed settlement, including hands ending without a policy decision.
HUDs reset at block boundaries. CIs cluster the correlated 100-hand pair;
they never count the individual hands as independent. These normal intervals
are approximate, especially at smoke budgets.

## Candidate formats and historical policies

- `teacher`: an SD-CFR snapshot directory with **every** iteration from
  `start` through `end`, inclusive. A uses 150..350, weighted by iteration,
  sampling one network per complete hand. No local probability averaging.
- `student`: a distilled `policy.pt` with its original tokenizer, numeric
  version and action guards. No implicit temperature or cutoff changes.
- `stage2`: a frozen residual checkpoint, including its embedded reference.
  An explicit `update` must match the internal update exactly.
- An optional 0.59M legacy policy is accepted only in one of these compatible
  formats. Use `teacher` for a legacy SD-CFR archive. A legacy PPO or Botzone
  simple checkpoint requires its own explicit rules/observation adapter and
  is rejected here rather than silently reinterpreted.

When per-update Stage 2 files were not retained, the self-play bank may contain
the exact required historical residual tensors. Recover evaluation-only
artifacts from a **local copy**, without altering the original checkpoint:

```bash
python scripts/extract_stage2_eval_candidates.py \
  --checkpoint runs/local_copy/retained.pt \
  --out-dir runs/local_copy/recovered --updates 100 200 300
```

The extractor requires every non-reference state tensor. It combines those
with the permanently frozen reference, checks strict loading and exact tensor
equality, and records source and output hashes. Current-update recovery is
also checked against the original full model. Optimizer, pool and training
RNG are omitted: these files are for evaluation, not training continuation.

## Script behavior

`TAG_Pro` ranks all 1,326 starting combinations using the existing Monte Carlo
preflop table: top 35% raises from SB, the next 30% limps; BB defense tightens
with raise size and raise count, with strong-value and occasional suited-ace
reraises. The rank is a reproducible heuristic, not a solved 200BB opening chart.
Postflop it uses the visible board, last preflop aggressor, dry-board half-pot
c-bets, strong-draw semi-bluffs, value betting and randomized river bluff-catcher
folds. Small river bluffs are included. Raw equity is against uniform unknown
cards; the script is not a range solver or an equilibrium strategy.

`Trapper_CR` shares that framework, but OOP at the start of a postflop street
checks 50% of sets, privately improving straights/flushes or better, and top
two-pair hands. It then raises a bet on that same street when legal. It also
checks some strong draws to semi-bluff raise. Board-only river monsters are
excluded. Both scripts consume only `HandSnapshot`, never the table's hidden
cards or predealt future board. Counters record traps, c-bets and river folds.

The original `OPPONENTS` tuple and Stage 2 pool weights are unchanged. New names
can be explicitly instantiated through `RuleActor` for separate experiments.
Adding them to training would consume this evaluation holdout; keep separate
seeds and independent external opponents for final model selection.

## Slumbot

```bash
python scripts/eval_pro_slumbot.py --kind student --path runs/student/policy.pt \
  --hands 1000 --out-dir runs/slumbot_student
python scripts/eval_pro_slumbot.py --kind stage2 --path runs/stage2/update_0000300.pt \
  --update 300 --hands 1000 --out-dir runs/slumbot_stage2
python scripts/eval_pro_slumbot.py --kind teacher --path runs/teacher/strategies \
  --start 150 --end 350 --hands 1000 --out-dir runs/slumbot_teacher
```

Alternatively add `--eval-slumbot --slumbot-hands 1000` to the tournament. This
evaluates each frozen candidate separately online after the local tournament.
Use at least 5,000 online hands for a larger external calibration. Four hands
are suitable only for connectivity and replay smoke.

The adapter reuses the tested exact-chip standard-HUNL replay, with full 200BB
stacks and no legacy 50BB cap. Tokens remain in memory and raw logs use an
explicit field allowlist. Non-idempotent HTTP actions are never blindly retried
after a timeout. Partial results and failure type are preserved. Opponent
off-tree bets are counted once using the actual abstract targets at the time
of the exposed decision. The residual sees public action history and a HUD
reset every 50 online hands. Slumbot chooses deals and positions; these are
**not duplicate games**, and its score stays separate from the local matrix.
Online CIs cluster complete 50-hand HUD blocks; fewer than two blocks have no CI.

The monitor now reads these standalone `report.json` files into the existing
Slumbot row. To inspect them, start this checkout's monitor against the parent
evaluation directory and select all experiments (the reports are not formal
training runs). This does not deploy/restart the current monitor, attach scores
to another checkpoint, or overwrite a live run's logs.

## Selection

Large wins against extreme bots must not dominate by sheer chip magnitude.
The descriptive ranking uses equal-opponent percentile ranks, prioritizing
the weakest three axes; mean rank breaks ties. Round-robin contributes one
axis. Selection requires >=1,000 duplicate deals and two candidates. The report
can recommend the best *available* candidate without certifying a champion.

`champion` is populated only if exactly one candidate also has approximate
Bonferroni simultaneous lower bounds >=0 against all eight scripts and >0
against every other candidate. Otherwise it remains null. These predeclared
conditions are conservative experimental criteria, not proof of equilibrium,
professional strength or online ranking. Confirm any selection on a fresh
seed before deployment. No tournament winner is automatically exported.
