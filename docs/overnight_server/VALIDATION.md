# Validation evidence

No existing test was edited, deleted, weakened or newly skipped.

The complete local suite passed **835 tests, 8 existing environment skips, in
309.29s**, after rerunning outside the macOS shared-memory/socket restriction.
That collection included the first 34 new cases. The final three added cases
and all related guard/context/export contracts then passed together:
**85 passed in 2.65s**. The new module has **37 tests**, including 16,384 seeded
random guard states/distributions, exemptions, symmetric call/jam caps, strict
latency boundary cases, real-state sampling, production-CFV rejection,
HUD/range bounds, serial-inference budgeting, checkpoint corruption/restart,
hard-arm isolation, RPC diagnostics, version reservations and crash-safe
publication/exclusion receipts.

The first sandboxed full run had ten host permission failures (POSIX shared
memory and localhost bind) and detected the stale development ZIP. The ZIP was
regenerated and the complete allowed-environment rerun passed. These failures
were resolved through the correct environment/artifact, not by changing tests.

All 25 portable modules pass Python 3.6 grammar. This is a grammar check, not
an actual Botzone Python 3.6/NumPy 1.19 platform acceptance claim.

`pr85_benchmark/` retains actual measurements on the remote Xeon 8470Q,
CPU1, governor `schedutil`: 32 warmups, 512 timed real-state draws and 12 cold
starts. The initial corpus retained genuine MatchCtx checkpoints with only the
current request; the final mandatory startup stage uses full cumulative
requests/responses/data and retains all raw timing samples and state IDs.

- Original PR85: mean 59.512ms, p50 49.990ms, p99 178.478ms, max 184.980ms,
  cold max 0.422s. Its hard p99 gate fails.
- Original PR85 with HUD enabled: mean 60.438ms, p50 50.434ms, p99 180.558ms,
  max 201.678ms, cold max 0.416s; 355 calls used a nonzero HUD prior and 128
  timed river calls used all 1,081 combos. Its hard p99 gate also fails.

These are measurements of an unqualified calibration CFV fixture and prove a
runtime bottleneck, not playing strength or final-arena acceptance. Production
50-epoch CFVs, all 200-match arms, confidence intervals and V26-28 packages
must be generated and checked by the locked remote production run. No fixture
result is used to promote or rank a formal package.

## Final handoff status (user-requested validation delegation)

The prior locked revision completed 839 tests with 8 unchanged environment
skips. The final addendum adds entry-only NumPy HUD/sigma caches, a separate
NumPy-only inference service with RSS measurement, fixed 8/8/16 iterations and
2 chance samples, and 600/1500ms/3s/220MiB hard limits (160ms mean warning).
These final changes have NOT completed regression or server acceptance.
At the user's request, further testing and final Arm D verification are
delegated to the remote server agent. Do not treat earlier measurements as
acceptance of the final deeper configuration. Existing tests remain intact;
new addendum cases are in test_full_power_server.py.
