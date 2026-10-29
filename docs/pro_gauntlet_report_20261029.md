# Pro Gauntlet: implementation and evaluation report

Date: 2026-10-29 (Japan time).

## Decision

Keep update 100 as the **provisional candidate under this tournament's
predeclared ranking**, and retain updates 200 and 300 for confirmation.
No candidate satisfies the simultaneous-bound champion criterion. This is
not a deployment approval, a Botzone ranking claim, or evidence of Nash
exploitability. The new scripted opponents expose weaknesses that the
original extreme-style pool did not test directly.

The implementation and this report cover independent evaluation. The original
Stage 2 training process, configuration, PFSP weights and checkpoints were
not modified. That run completed its existing budget of 300 updates, 6,000
matches and 300,000 hands. No new training or external-bot integration was
started.

## What was implemented

- `TAG_Pro`: combination-weighted preflop ranges, position/raise-size-aware
  defense, continuation betting, strong-draw semi-bluffs and randomized river
  bluff-catcher folds.
- `Trapper_CR`: the same framework with a 50% strong-hand trap check when
  first to act postflop, followed by a legal check-raise, plus some draw traps.
- A frozen-policy tournament for a contiguous CFR teacher window, distilled
  students and Stage 2 residual checkpoints. Optional legacy candidates must
  use a compatible checkpoint format and rules contract.
- Full-200BB Slumbot evaluation for all three policy formats, including public
  HUD updates for the residual, exact off-tree chip replay, bounded requests,
  secret-free logs and preserved partial results.
- Strict extraction of historical residual policies from retained self-play
  banks, with a frozen reference, complete tensor checks and source hashes.
- Monitor parsing for standalone Slumbot reports and labels for the two new
  scripted styles. The existing live monitor was not restarted or deployed.

The original six-bot list and training-pool defaults remain unchanged.
The scripted styles have not been calibrated against professional players.

## Evidence included with the PR

- [Machine-readable summary](benchmarks/pro_gauntlet_20261029/summary.json):
  all 60 displayed cells, confidence intervals, match-pair totals, policy and
  raw-log hashes, source fingerprints and the selection rule.
- [Complete matrix as CSV](benchmarks/pro_gauntlet_20261029/matrix.csv).
- [Standalone HTML matrix](benchmarks/pro_gauntlet_20261029/matrix.html):
  download/open locally to view the formatted matrix.
- [Common deck schedule](benchmarks/pro_gauntlet_20261029/deck_seeds.json).

The evidence bundle omits model binaries, raw per-hand logs, server paths and
online session tokens. Its source-report SHA256 binds it to the full local
report; each cell also records the raw-log fingerprint. Included match-pair
totals are sufficient to recompute the reported nominal intervals and ranking.
Full decision replay additionally requires the original policy artifacts and
raw logs, which are not distributed in this PR.

## Tournament protocol

Five verified candidates participated:

- A: Stage 1 `w150_350`, with every snapshot 150..350 inclusive. One snapshot
  is drawn per hand with linear iteration weights and retained for that hand.
- B: the accepted high-fidelity distilled student, with its original policy
  temperature already embedded in the checkpoint.
- C100, C200 and C300: exact update 100, 200 and 300 residual policies recovered
  from the retained self-play bank and the immutable reference.

The optional 0.59M v2 candidate was excluded because its compatible artifact
identity was not verified. It was not replaced by a guessed legacy policy.

Each candidate faced the original six standard scripts plus `TAG_Pro` and
`Trapper_CR`. Every candidate pair also played a direct match: **40 scripted
matchups + 10 direct matchups = 50 actually played matchups**, totaling
**100,000 hands**. The ten mirrored round-robin cells are exact zero-sum
derivations, not extra games.

Settings: seed `20261030` (an RNG identifier), 1,000 duplicate deals per
matchup, 64 equity samples and no opponent jitter. Each matchup contains 20
duplicate 50-hand match pairs. Seats alternate within a leg and identities
swap for the other leg. All cells share the same full deck schedule.
Both neural actors receive their own public-event HUD and signed settlement;
HUDs reset at each 50-hand boundary. Intervals use the 20 correlated match-pair
means, not 2,000 independent hand observations. The normal approximation is
still imprecise at this budget.

B/C games ran on local CPU. After the original training finished, A games ran
with one remote CPU thread and read-only model access, streaming outputs to
local disk without creating remote files. Deck-schedule hashes matched across
the two execution environments; model binaries and hardware differ, and no
cross-hardware bitwise-logit equivalence is claimed.

## Results and interpretation

The complete cell values and nominal 95% intervals are in the linked matrix.
The selection result is deliberately separated from the equal-weight mean:

- **C100** ranks first by weakest-three opponent-rank scores. Its eight-script
  mean is **+1,222.0 mbb/hand**. TAG is −925.0 [−1,798.1, −51.9]; Trapper is
  −428.25 [−1,274.2, +417.7]. It remains a provisional choice, not a proven
  strongest policy.
- **C200** has the highest eight-script point mean, **+1,538.47 mbb/hand**,
  but Trapper scores −1,297.5 [−2,212.3, −382.7]. Its Equity result is −304.0
  [−1,760.9, +1,152.9], which does not establish positive expected value.
- **C300** has an eight-script mean of **+1,349.78 mbb/hand**, with negative
  point estimates against both new styles. A larger update number does not
  automatically produce the most balanced candidate.
- **A** has an eight-script mean of **+702.63 mbb/hand**; **B** has −159.97.
  These values come from this specific shared-deal tournament. They must not
  be subtracted directly from earlier 2,500-pair acceptance reports with
  different deals, RNG trajectories or scheduling to assert a regression.

Ranking first prioritizes the average of each candidate's three weakest
equal-opponent percentile scores, then the mean score. Round-robin contributes
one additional axis. Huge chip wins against a maniac therefore cannot dominate
selection solely through payoff magnitude. The script means above are
descriptive averages, not additional confidence statements.

A champion additionally requires approximate Bonferroni simultaneous lower
bounds >=0 against all eight scripts and >0 against every candidate. The
familywise alpha is 0.05, with z approximately 3.3415 for this five-candidate
matrix. **All five candidates fail that criterion; `champion` is null.**
The nominal intervals quoted above are not the adjusted champion intervals.

## Slumbot and regression validation

The accepted student and C300 each completed **four real online Slumbot hands**.
This checks connectivity, full-200BB observation replay and legal inference;
it does not estimate playing strength. The actual online deals are chosen by
Slumbot, so these scores are separate from the local duplicate matrix.

The affected Python regression set covers 78 tests; all passed, including
the two additional teacher-window and report-validation checks. Ten frontend
tests also passed. The monitor HTTP test requires an available loopback port.
The full tournament evidence was checked for 50 unique raw-log hashes,
100,000 complete hands, matching deck schedules, exact per-hand-to-match
settlement sums, zero-sum mirrored cells and 201 teacher snapshot identities.

## How to run it

Follow the [operator guide](pro_gauntlet.md) for the candidate manifest,
historical extraction and standalone Slumbot commands. A typical tournament is:

```bash
python scripts/eval_pro_gauntlet.py \
  --manifest configs/cfr/pro_gauntlet.example.json \
  --out-dir runs/pro_gauntlet_confirmation --pairs 2500 --seed 92031
```

Replace the example paths with frozen artifacts first. Add `--eval-slumbot`
to run a separate online campaign after the local tournament. Existing output
directories are refused rather than overwritten. Do not point output at a
training directory.

## Proposed next steps

1. Confirm the current shortlist on fresh deals at 2,500 duplicate pairs.
   This ranking used one selection seed and needs independent confirmation.
2. Run a larger Slumbot campaign with complete 50-hand HUD blocks. The four-hand
   smoke is not sufficient for model selection.
3. Investigate DecisionHoldem blueprint-only artifacts first, followed by the
   published HU200 v0.5.5 model. Check assets, rules, information boundaries,
   fallback frequency and latency before adding either to a new training pool.
4. Keep a separate evaluation policy/deal holdout if TAG/Trapper or an external
   policy becomes a training opponent. Changing a later pool requires a new
   explicitly identified experiment, not mutation of the completed run.

The [open-source opponent assessment](open_source_stage2_opponents.md) includes
upstream links and the practical differences: DecisionHoldem's blueprint-only
strength remains unverified; the HU200 model has substantial assets and an
upstream-engine licensing gap; GTO Poker Overlay ships a 100BB blueprint that
cannot faithfully substitute for a 200BB policy. None is integrated by this PR.
