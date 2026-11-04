# Mathematical repair and deployment-path review

This branch repairs the canonical repository source. It does **not** modify or
recertify the immutable V13/V14/V15 ZIPs/NPZs, start production training, create
an upload release, upload, deploy, run an online match, or merge a PR.

## Ownership and evidence

The owning repository is `raymo-lei/poker-ai`, in `poker-ai-1`, with base
`0f6a096` (`origin/main`, fetched on 2026-11-04). The outer Poker repository has
no commits or remote; `worktrees/botzone-revision7` is not a worktree of this
repository. The original checkout's dirty wizard and untracked deep solvers /
sprint scripts were preserved. Work is isolated on
`codex/math-contract-search-repair`.

`legacy_identity.json` verifies all six original release SHA256 identities
against the audit manifest. `scripts/math_repair/verify_legacy.py` imports the
read-only extracted V15 code without writing bytecode and independently
reproduces its rank inversion (70 versus 973972) and forced 10BB value
0.0004516507836141096. The new terminal returns exactly 10BB. No result here is
attributed to running a repaired original ZIP.

The larger guard/context delta is a **port of the audited R7 deployment code**
into the actual owning repository, including its blind-lock / shove-evidence
dependencies. It is required to test real release metadata rather than a
simplified main-branch guard. These empirical rules are not new poker formulas.
They run once, after search. The broken source-transplant packers and the
standard-HUNL tournament evaluator have been retired; supplied Botzone payloads
are evaluated through the same `respond()` as production.

## Audit issue mapping

### A — formal path: repaired

`respond -> Agent.act -> Agent.policy -> wizard.fuse -> public-tree DCFR` is the
normal path, even when victory locking is configured. Only an actual own lock
or explicit opponent probe skips search, with a reason. Other guards run once
at the end, so forbidden final actions cannot be revived by later fusion.
The final policy records whether empirical guards changed the DCFR average.
Cache keys include model tensor/config identity, solver contract, release and
the complete public request (including scores/schedule); cached actions are
revalidated. Numeric/range downgrade is `error_blueprint`, not a search pass;
strict mode raises. Unexpected failures reach the visible entry fallback.

`tests/test_search_entry.py` calls the real generated `respond()`, counts one
search and one guard, compares direct Agent decisions, checks lock/cache/error
branches and final forced actions. The process evidence in `benchmark.json`
loads the audited V15 NPZ **as a frozen policy only**, with new development
source. Debug identifies both, starts `py`, and records `search=active`, solver,
value contract, legacy rejection and the final guard policy. No beta/KL
calibration or unverified value network is used.

### B — game, information sets and chips: repaired within a declared game

Fixed-width rank encoding preserves all nine categories, kickers, wheel and
board ties, including 5s6h versus AsQh on 2s3h4d9cKs. The same portable evaluator
is always used; correctness does not depend on an optional package.

Public successor nodes reuse `reconstruct.judge_accepts`, `min_raise_extra` and
`_put`. Terminal nodes have no response slots. Check/bet/raise/call/fold paths
are actual nodes; per-own-hand strategies are shared over hidden opponent
hands. There is no per-pair maximum. Calls use the outstanding additional cost
once; short all-ins and uncalled chip refunds are settled explicitly. Tests
compare root chip differences to the unmodified referee for check, bet/call,
raise/call, fold and equal-stack all-in. Unequal-stack refunds are an explicitly
specified general heads-up extension: the Botzone referee resets equal stacks
and does not implement general side pots. SB limp retains BB's option.

Root actions are the supplied executable chip mapping. Duplicate buttons for
the same chip amount have identical subtrees, not maximum Q across other sizes.
Continuation half-pot/pot/all-in actions use the platform minimum and stack
bounds. A raise cap removes later aggressive branches; it is an **action
abstraction**, not a claim to solve the full no-limit tree. The platform's
unusual accepted positive undercalls are deliberately excluded from this
abstraction; no such amount is emitted by the canonical button mapping.

### C — DCFR: repaired and independently tested

[CONTRACT.md](CONTRACT.md) specifies signed DCFR(1.5,0,2), iteration-start history
discount, a single frozen strategy snapshot per iteration and no pseudocount in
the average. Negative history receives 1/2, and clipping occurs only in regret
matching. Recursion carries opponent/chance reach into regret and
`t^2 * own_reach * sigma` into averaging. Legal children are enumerated; no
sentinel utility for impossible actions exists. Returned search policies are
averages, not KL responses. Shared information sets reject inconsistent own
reach histories. There is no claimed historical-regret warm start.

`numerical_report.json` includes independent pure normal-form best responses
for Kuhn, not just a probability-sum or EV assertion. A hand-derived two-iteration
multilayer example tests own reach and snapshot indexing. The independently
correct external-sampling MCCFR trainer was **not modified**.

### D — beliefs and buckets: repaired, with a declared observation approximation

Each candidate has its own frozen-policy forward at the historical event's
public board, actor, stacks, pot, owed amount and legal action domain. The legal
probabilities normalize together. The current hero prior is never copied to
opponent types. Whole-history replay on combo weights preserves posterior mass
across streets; river uses the history too. Public cards filter the combo
support, and incompatible private pairs get zero joint mass. Individual
candidates can share cards; every candidate is exactly a two-card hand.

The observation model is **neutral-context frozen policy, size-coarsened
aggression**. It is not a calibrated model of the actual opponent, and does not
reconstruct unavailable historical match HUD. Arbitrary observed sizes are
modelled by the sum of legal aggressive categories, not an invented precise
size density. This replaces unnormalized hand-written likelihood tilts with a
verifiable conditional policy, without claiming the model assumption is true.

Old 12/32-bucket aliasing, potential bits and empty-bucket pseudomass are removed
from the new runtime architecture. Combo support has no empty buckets. The
optional aggregation helper accepts independent maps for either side and keeps
zero buckets zero. Invalid/empty posterior mass is explicit. Sampling before
river uses uniform-without-replacement inclusion probabilities and inverse
inclusion masses; the hero row is a separately accounted stratum. This finite
support changes the solved game and has no unbiased equilibrium guarantee.

### E — river operator: repaired

All 1081 public-board-legal combos are supported; each hero conditions on its
990 compatible opponents under a uniform range. Both constants and showdown
terms use the same Z(h). Rank groups, strict-lower prefixes, tied mass,
single-card inclusion/exclusion and the own-combo correction are implemented.
Zero Z returns an unreachable mask and NaN internally; serialized targets use
null plus masks. Sparse numerical cancellation invokes a dense recomputation,
using a floating-sum error bound, without adding probability mass. Both the
single operator and the full sparse-range solve are checked against independent
dense calculations.

`benchmark.json` separates kernel, complete solve with warm/cold rank building,
and fresh subprocess/model/history/decision timings. It is local macOS data,
with warm filesystem caches, not target Botzone CPU/memory/Python 3.6 evidence.
The multi-second full process is a deployment blocker. There is no <5ms claim.

### F — values and labels: generator repaired; production training pending

`combo_value_v1` uses actual board IDs/street, actor/order/pending decisions,
per-street contributions, platform raise bound, chip unit, pot/BB, stacks/BB,
both combo supports and both actual input ranges. It has no backbone feature
placeholder; the 192-dimensional policy trunk is not wired into an old
128-dimensional interface. The old dead block was 72:200, not the first 128.

Targets are conditional **centered root BB**, with explicit validity and
range-weighted zero-sum checks under the compatible joint deal. Scaling is
identity and shared by training-data construction and inference helpers. The
root/leaf translation is tested, including a facing-bet call. Runtime uses no
neural leaf; exact terminal / named check-down operators use actual successor
chips, and tiny turn labels enumerate each legal river **then river betting**.
Changing input ranges changes the targets. No river target is copied onto a
turn input. `smoke_labels.json` contains the four small accepted examples.

Every old `vleaf.*`, `cfv_turn.*`, `cfv_river.*`, and `preflop_freq` artifact is
explicitly rejected for value/table use. A copied schema string or matching
shape cannot activate a network: no trained v1 decoder is shipped. Existing
policy parameters remain a statistical observation prior, not a certified GTO
solution. Large label generation, architecture fitting and production value
training remain unexecuted. Runtime pre-river continuation is **check-down**,
not the small generator's complete finite-support turn game.

### G — preflop and empirical interventions: repaired / classified

The corrected generator uses actual combo masses (169 classes aggregate to
1326 combos; pair/suited/offsuit sizes 6/4/12), actual blocked joint deals and
the same recursive reach-aware solver. Caller selection reaches the leaf
through action probabilities, rather than a global call-frequency multiplier.
There is no playability equity rescaling, class-index overwrite, utility jam
tax or arbitrary half neural/half showdown leaf. Monte Carlo runouts are paired
by unordered private deal, so equity is complementary when player roles swap.

Tables are addressed by actual actor/action/additional-chip public histories,
not raise counts. All-in calls are calls. 100BB and 200BB are solved separately;
`smoke_preflop.json` summarizes those independently generated finite-support
games. Runtime never reads the old six-node table, and never transfers illegal
jam probability into a different bet. No trash-hand prohibition was added.

## Retained guard classification and enablement

All legacy knobs retain their explicit metadata names and original defaults.
The R7 model metadata continues to enable its interventions. None is represented
as a game-theoretic identity. `guard_changed` and `last_guard` show actual use.

- **Rules/platform constraints:** legal integer responses, stack bounds,
  additional chips, 50/100 blinds, 20000-chip resets, at most 50 hands, explicit
  hard final action constraints. These are not removable tuning constants.
- **Authorized match objective:** `victory_blind_lock_enabled`, own-score bound
  and blind budget. The own bound has a public-score proof. Opponent lock
  confirmation/probe/transfer remains an **opponent-behavior heuristic**, not a
  guaranteed chip-EV or match-win theorem.
- **Statistical assumptions:** context Beta priors, equity Monte Carlo
  seeds/budgets, frozen-policy observation model. These have named assumptions,
  not exact posterior/strength claims about the actual opponent.
- **Empirical probability interventions:** `micro_stack_commitment_enabled`,
  `shove_sb_discipline_enabled`, `shove_runway_enabled`, `adaptive_anti_shove`,
  `effective_shove_enabled`, `adaptive_call_protection_enabled`, legacy giant /
  `big_call_enabled` fuses, `safe_shove_remap`, `passive_probe_enabled`,
  `pot_control_enabled`, `late_pot_control_enabled`, `raise_redirect_enabled`,
  `dominant_fold_enabled`, `late_weak_call_enabled`, `small_probe_defense_enabled`,
  `preflop_facing_raise_fold_enabled`, `preflop_weak_reraise_enabled`,
  `flop_highcard_fold_enabled`, `free_flop_weak_bluff_enabled`,
  `strong_hand_price_exception_enabled`, `good_hand_blind_protection_enabled`,
  `free_weak_tail_enabled`, `station_control_enabled`,
  `postflop_raise_gate_enabled`, `strong_cap_no_fold_enabled`,
  `private_two_pair_strong_enabled`, `strong_fold_floor_enabled`,
  `made_hand_redirect_call_enabled`, `thin_call_fold_enabled`,
  `overbet_made_hand_exempt_enabled`, `prune_below`, `purify_threshold`, plus
  optional `safety`. Their purpose is legacy risk/price/tail/opponent response
  control; they lack a derivation from the centered chip utility. They are
  preserved for compatibility **as a named final empirical layer**, including
  size transfers that are not EV-preserving. They are outside the solver game.
- **Failure policy:** entry's >3500-chip fallback call limit is retained as an
  explicit platform emergency policy, with `fallback` diagnostics. It is not
  counted as successful search or evaluated as a poker equilibrium strategy.

Applying guards to the search result is an intentional behavioral change from
applying them to blueprint logits and then reviving actions with fusion. The
final guarded output is not advertised as an equilibrium strategy.

## Reproduce / follow-up gates

Run the portable numerical suite with Python+NumPy+pytest:

```
OPENBLAS_NUM_THREADS=1 python3 -m pytest tests/math_repair
python3 scripts/math_repair/numerical_report.py NEW_REPORT.json
python3 scripts/math_repair/generate_labels.py --output NEW_LABELS.json --iterations 100
python3 scripts/sprint24/solve_preflop.py --output NEW_PREFLOP.json --iterations 100
```

The labels CLI uses exclusive creation, so old data cannot be overwritten. The
source-only tests and real entry integration require the repository's Torch /
poker dependencies. On the local Torch 2.14 environment,
`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` is needed by the existing trusted-checkpoint
loader (this PR does not change checkpoint security behavior).

Before any deployment: decide the desired game/utility and abstraction budget;
replace check-down with validated continuation as needed; generate versioned
range-conditioned production labels; train and independently hold out/evaluate
new value networks; bind their exact schema/provenance; measure approximation
and finite-iteration errors; optimize the end-to-end likelihood/replay path;
measure the actual Python 3.6/NumPy 1.19 Botzone CPU/memory limits; run authorized
same-entry evaluations. Only then allocate an unused increasing Vn and create
all AGENTS.md source/model/hash/debug/replay-verification artifacts. This PR
provides no new production-trained value net, full HUNL Nash certificate,
playing-strength evidence, target performance certification or upload package.
