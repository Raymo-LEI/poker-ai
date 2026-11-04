版本：V21（内部 R1），release=V21_EVENING_V2_20261104_SMOKE

# Evening V2 repair and validation

The later [overnight phase-two controller](../overnight_production/PRODUCTION.md)
adds opt-in check-raises, trajectory PBS mixtures, expanded joint policy/value
distillation and actual gated Astra candidate searches. Its
[validation](../overnight_production/VALIDATION.md) is separate from the
historical V21 smoke evidence below.

This PR integrates the second Evening audit repair onto main after #80
(`5851d88`). The existing `Agent.policy()` / `respond()` pipeline, authoritative
judge state, final guard ordering, decision cache identity and old search path
remain in place. Versioned CFV artifacts select the new strict Evening solver.
The retired ZIP transplant entry points remain retired; `pack_evening()` uses
canonical `stage2.export` with its existing `release_id` API.

## Repair coverage

- R1–R3: fixed-width fallback rank encoding, complete public combo support,
  preserved prefix/removal/tie kernel, explicit zero-compatible-mass handling.
- R4–R7: one finite betting tree throughout solving and policy evaluation,
  separate per-bet opponent responses, both players' counterfactual reach,
  signed DCFR(1.5,0,2), executable root amounts, terminal Calls and refunds.
- V1–V3: deterministic public buckets built before solving; both input ranges
  condition the labels; real two-player turn/flop betting trees use updated
  branch ranges, pot and stacks at neural leaves. Chance transitions retain
  exact hand-combo identity, remove new-card conflicts, and correct sampling
  mass. All-in leaves sample runouts and settle showdown directly.
- V4–V5: `cfv.py` owns the versioned 136-feature schema and exact three-block
  residual GELU forward. `TAIL_NAMES` names all 20 public scalar features;
  the first 64 entries are both ranges and the next 52 are board bits.
- C1–C5: validated CFV loading, package imports, correct SlayerActor import,
  average-policy labels and search-driven collection using actual engine
  button amounts. No existence-only wait or `leaf=None` shortcut.
- C6–C7: disjoint match seeds for training, preflop anchors, validation and final
  holdouts; frozen source residual routing for every preflop decision,
  persisted through checkpoints and NumPy export; rejected edits roll back.
  A deliberately bad surgery candidate exercised the actual rejection path.
  Full Astra grid search was not run; optional candidates enter via `surgery_fn`.
- E1–E5: evaluation executes the ZIP's `runtime.respond()` with canonical
  release/cache identity and safety settings; bundles contain CFV and solver
  modules; per-match win rates use `summarize_match_pairs`; blind budgets
  alternate correctly and matches are never truncated on a lead estimate.
- U1–U3: normal CDF uses `sigma*sqrt(2)` and nonlinear utility is evaluated
  separately at terminal win/tie/loss outcomes. Opponent utility is its negative.
  CFV training stays linear zero-sum BB. Non-river utility requests are rejected
  because expected chip CFVs do not encode terminal payoff distributions.
- P1: atomic artifact/checkpoint publication; manifests bind schema, BB units,
  counts, epochs, datasets, sources and array hashes. Embedded CFV validation
  rejects stale sources, malformed layers or mismatched array identity.

## Value and abstraction contract

`pot` already includes the outstanding opponent bet. With `P=10BB`, `c=4BB`,
`equity=.5`, Call returns `.5*(10+4)-4 = +3BB`. There is no response after a
river Call. Unmatched chips are refunded before settlement.

Training stores centered root BB: incremental value minus half the root pot.
Thus the example's centered value is -2BB. Cross-street translation is
`V_leaf + leaf_pot/2 - H - root_pot/2`, in BB.

The finite tree allows executable root buttons; after Check the opponent can
Check/half-pot/pot/Jam and Hero Fold/Call; after a root bet the opponent can
Fold/Call/Jam-raise and Hero Fold/Call. This is an explicit betting abstraction,
not the complete no-limit action space. Buckets assume uniform combo mass
within a bucket; neural leaves and finite public chance samples are approximate.

## Validation

Combined suite, including the math-repair and production-entry tests from #80:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python -m pytest -q \
  -o addopts='' tests/test_evening_v2.py tests/math_repair \
  tests/test_search_entry.py tests/test_wizard_solver.py \
  tests/test_botzone_stage2.py tests/test_stage2.py \
  tests/test_botzone_referee_diff.py
```

Result: **91 passed, 2 skipped in 167.82s**. The skips are actual Python 3.6
execution and Linux platform-limit acceptance. `git diff --check` also passed.

The independent self-test generated 4 river and 3 turn training rows, with
2 holdouts each, and trained each head for one epoch. Two distillation
updates used 4 labels each and separate two-row anchor/validation/final sets.
It built a fresh V21 smoke release without overwriting V20. Both final
preflop drift checks were zero. Torch/NumPy forward maximum errors were
6.95e-08 BB (river) and
4.91e-08 BB (turn).

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  python scripts/sprint24/selftest_evening_v2.py \
  --out eval/evening_v2_repair/repeat \
  --deliver /path/to/shared/poker/upload/root
```

The shared upload root must contain previous versioned upload directories so
`pack_evening()` allocates the next increasing version. Rerunning creates a
new release. Runtime requires NumPy; generation/tests additionally need the
project's Torch, phevaluator, PokerKit and pytest dependencies.

Final ZIP probes blocked `phevaluator` import and exercised preflop, flop,
turn and river through the real ZIP entry. Both flop and turn debug reported
`cfv=18/dim=136`. River uses exact terminal settlement and explicitly reports
`cfv=0/reason=exact_river_terminals`. Both complete 50-hand judge matches
finished with zero protocol errors (301 Hero decisions).
The evaluator also reproduces canonical release identity in saved-decision caching.

The other independent checks cover all hand categories/kickers/wheel/ties,
3,000 fallback-vs-phevaluator comparisons, dense/sparse/tied/unreachable
prefix-kernel comparisons, PokerKit successor ledgers including unequal-stack
refunds, Kuhn's -1/18 value and best responses, range sensitivity, cross-street
combo identity, frozen preflop checkpoint roundtrip, teacher-action collection,
atomic corruption rejection and native replay-HTML release verification.

See [machine-readable smoke summary](smoke_summary.json).

## Release identity and limits

- Release: `V21_EVENING_V2_20261104_SMOKE` (smoke pipeline acceptance only).
- Source SHA256: `09053e8b3a0eae254cf62891e75884cc90ec44fc311e5e447ff5d6402cdf81d3`.
- Model SHA256: `29c4a4d6180baad83f409a8b2e6d8342c53fc8439d8a52e7ad89ccd4caf0c907`.
- Trained checkpoints, CFV weights, datasets and upload directories are local
  validation artifacts and are intentionally excluded from this PR.
- The generic committed `botzone_bots/stage2.zip` is a reproducible runtime
  source fixture, not an allocated upload release.
- Python 3.6 grammar parsing passed. The execution host was macOS,
  Python 3.11.17 and NumPy 2.4.6. Python 3.6.5/NumPy 1.19.5 execution and
  Linux single-core/256MB/6-second acceptance remain unverified.
- Tiny training counts and large held-out CFV errors do not establish model
  quality or playing strength. No production training or deployment was run.
