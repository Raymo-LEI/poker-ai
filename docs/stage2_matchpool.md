# Stage 2 MatchPool and independent acceptance evaluation

This extends the 50-hand residual PPO implementation. It changes training code and diagnostics; it does not establish profitability against any opponent. No final Stage 1 checkpoint, remote GPU training, or Botzone deployment was used for implementation acceptance.

## Match boundary and observations

`BotzoneMatchEnv.reset()` samples exactly one actor from `MatchOpponentPool`. That instance, its rule parameters, and its neural snapshot id stay fixed for all 50 hands. Unfinished matches cannot reset. Neural opponents receive their own independent public-action HUD, reset at the next match. No opponent type, jitter parameter, snapshot id, future board, or hidden opposing cards enter either policy's observation. The existing 12-field `MatchCtx` and exported agent contract stay unchanged.

An update now collects 20 concurrent complete matches (1,000 hands), or a configured multiple of 20. Each shuffled block reserves exactly two distilled-reference environments, three CFR-history environments, two PPO-history environments, and thirteen rule environments. This is an exact 35% defensive allocation per block; it is not a multinomial expectation. Finishing matches wait until the block ends. Sampling counts and the actual defensive fraction are logged separately from target probabilities.

`collect_matches()` groups pending opponent decisions by the shared immutable model instance, following `transformer.env.PolicyActor.prefetch`. It runs one forward pass per snapshot id and keeps action RNG and public HUD state private to each match. Hero decisions are batched too. GAE carries across hands and resets at each 50-hand terminal, including passive hands with no hero decisions.

## Four-tier composition

Initial expected allocation:

- Equity family: 35%. Standard, tight, and aggressive variants are equally likely within this family. Standard calls the unchanged `RuleActor('equity')` implementation with identical RNG behavior and the same default 64 equity samples. Its equity is the existing Monte Carlo estimator, not a newly claimed exact enumerator. Tight adds a fixed, match-local 0.03–0.08 equity margin to calling and raising thresholds. Aggressive prefers pot-sized bets at a jittered strong-equity threshold and shoves above 90% equity. The existing five-action tree has half-pot, pot, and all-in; it has no arbitrary overbet size.
- Biased scripts: 30%, consisting of nit 12%, calling station 6%, aggressive 3.5%, overbet 3.5%, and new push/fold 5%. Nit thresholds move by ±0.02 equity. Stations retain broad calls and occasional strong value raises above a sampled 0.82–0.90 threshold. Aggressive/overbet action frequencies vary from 0.88–0.98. Push/fold shoves with a sampled 0.80–0.98 frequency, including preflop and flop; otherwise it folds to a bet or checks. All parameters are drawn only at match creation. Baseline evaluation bots remain unchanged and receive no jitter.
- Stage 1 defense: fixed 25%, split into 10% distilled-reference mirror and 15% CFR history from T=100–350. Equal-weight, disjoint buckets emphasize T=140–160, T=240–260, T=320–350, and the remaining T=100–350 snapshots. Required empty buckets fail at startup. The archive's selected weights and semantic contracts are validated and fingerprinted before PPO starts.
- Stage 2 history: fixed 10%. Before the first PPO update, this uses an explicit frozen zero-residual bootstrap. Every `self_snapshot_every=25` updates, the current residual weights enter a bounded 16-snapshot pool. Old residual snapshots share the permanently frozen reference backbone and maintain their own public HUD when sampled.

Stage 1 inference modules stay resident on the rollout device. `history_cache_size` accepts 8–16; the default is 16. The working set rotates every 100 sampled matches at a complete quota boundary. Bucket selection stays balanced regardless of bucket size, and repeated sampling of a resident snapshot causes no disk load. Model construction on cache misses preserves the training Torch RNG, so resumed runs do not diverge merely because their inference caches start cold.

Inference remains FP32 to preserve existing checkpoint numerics. A 1.49M-parameter model is approximately 6 MB in FP32, so 16 historical models take about 96 MB of parameter storage, plus runtime activations and allocator overhead. The often quoted 3 MB/model assumes FP16; no unvalidated half-precision conversion is applied here. Resident counts and cumulative disk loads are logged. Partial synthetic archives use an explicit bucket override in the smoke script.

## Dual-track weakness sampling

Each rule family maintains a sliding window of the latest 100 completed matches. Its return is `sum(hero_net_bb) / (50 * matches) * 1000` in mbb/hand. Match Win is the fraction with strictly positive net BB; ties are not wins. Lifetime counters are separate from the sliding window.

For each rule family with observations, priority is 3 when either return is below +500 mbb/hand or Match Win is below 60%; otherwise it is 1. Unseen families start at 1, preserving the requested initial mix. Multiply initial family weights by this priority and renormalize within the 65% exploitation allocation. Defense remains exactly 10% reference + 15% CFR history + 10% PPO history. Rules retain nonzero probability throughout. Equity variants share family statistics; independent standard-equity evaluation is never mixed with jittered training results.

Checkpoint state includes sliding windows, pool RNG, remaining quota slots, resident ids/buckets, immutable archive fingerprints, and bounded residual-history weights. Changed pool contracts or archives are rejected on resume. Old PFSP checkpoints require a fresh run because they do not contain the new quota/history contract. Optimizer, action RNG, total matches, and cached frozen-baseline evaluation results are also restored.

## Independent evaluation and retention gate

`evaluation.every_updates` defaults to 100. At that interval, evaluate the current candidate and frozen Stage 1 reference against standard equity, nit, calling station, aggressive, overbet, and the distilled mirror. Use the same held-out decks, seats, action seeds, and unchanged baseline bot parameters for both policies. Cache the fixed Stage 1 results across updates and resume. Evaluation preserves the training Torch RNG and never updates PFSP training windows.

The requested budget contains a counting ambiguity: **2,500 duplicate deck pairs means 50 paired 50-hand matches, hence 100 complete matches / 5,000 played hands per policy and opponent**. Fifty complete matches alone would be 2,500 played hands, not 2,500 pairs. The implementation uses the larger, explicit paired budget. Each pair repeats the same 50 full decks in both legs and swaps hero/opponent seats on every hand. HUDs reset between legs; opponents remain fixed within each complete match. Candidate and baseline reuse the same deals. Match-pair means, not individual correlated hands, are the uncertainty units. Reports include raw paired match totals, counts, tie counts, 95% mean intervals, and paired gain intervals. Match Win reports each complete match's outcome; its interval also clusters by pair.

The Stage 2 monitor shows:

- Standard equity mbb/hand (target >0), its confidence interval, and 50-hand Match Win.
- Stage 2 minus frozen Stage 1 gain in mbb/hand for nit, calling station, aggressive, and overbet.
- Candidate versus distilled reference return, confidence interval, and the `>= -20 mbb/hand` retention gate.
- The evaluation update and actual deck-pair/match counts, plus the existing KL, public HUD, training return, and nine pool weights.

Training rewards never populate these independent evaluation cards. Between evaluation updates the latest complete evaluation remains visible, with its original update number. Before the first evaluation, these fields show missing values.

`stop_on_retention_failure=true` stops PPO when the measured mirror return is below -20 mbb/hand. The failed candidate remains in diagnostic `train.pt`, and its report and metrics are written before stopping. Only passing evaluations update `retained.pt`; a failed diagnostic checkpoint is refused for resume. KL remains `KL(final || frozen reference)` with the existing per-hand coefficients. Neither KL nor the sampled retention gate mathematically guarantees a true expected return floor or profitability between evaluations. Sampling uncertainty is visible in the reports. Nit stealing, zero-bluff station exploitation, and trapping maniacs remain learning objectives to verify on trained weights.

Run a separate evaluation without training:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python scripts/eval_stage2.py \
  --checkpoint /absolute/stage2_run/train.pt \
  --output /absolute/new_stage2_acceptance.json --device cuda
```

The script exits 2 for a measured retention failure. Smaller `--match-pairs` and `--equity-samples` are available for pipeline checks and must not be presented as full-budget acceptance. Default reports use 50 match pairs and the unchanged 64-sample rule baseline.

Training entrypoint and config remain `scripts/train_stage2.py` and `configs/stage2/botzone_50hand.json`. Choose actual distilled-reference and archive paths and an explicit training budget. The default is still two CPU updates, now forty complete matches. Importing modules or opening the monitor starts no training.

## Implementation validation

Tests cover exact quotas despite weakness reweighting, threefold priority and window expiration, standard equity baseline equivalence, fixed jitter, hidden opponent identity, full-match rewards/dones and HUDs, grouped inference, resident cache reuse/rotation, frozen residual snapshots, checkpoint schedule/RNG replay, common-deal evaluation gains, and monitor retention/gain rendering including real zero values.

The synthetic 1.49M CPU smoke performs distillation, two 20-match PPO blocks, self-snapshot insertion, and a NumPy-only single-file export. These checks validate code paths, not Stage 1 or Stage 2 playing strength. No full 2,500-pair acceptance or GPU benchmark is claimed.
