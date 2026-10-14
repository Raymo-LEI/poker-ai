# T-C1a cuda determinism checks (test script only)

Only `scripts/test_t_c1a_equivalence.py` changed (patch: `det.diff`). No training code touched.
Default behaviour (no new flags) is unchanged.

New flags:
- `--deterministic`: `torch.use_deterministic_algorithms(True)`, `cudnn.deterministic=True`, `cudnn.benchmark=False`
  (in the parent and in every child process). On cuda it exits with an error unless
  `CUBLAS_WORKSPACE_CONFIG` is `:4096:8` or `:16:8`. It has to be set in the shell before python starts.
- `--legacy-twice`: control run. It runs the legacy (non-cached) path twice from the same seed and prints
  `[legacy-twice] IDENTICAL` or `[legacy-twice] DIVERGED n=..`. If it diverges, the nondeterminism is not from the cache.
- `--per-step K`: a base legacy run saves a checkpoint after every update. In trial t, both paths resume from the
  same checkpoint (update t-1; trial 0 starts fresh). Each checkpoint holds the model, Adam, pool, tables, actor RNG,
  and python/numpy/torch-cpu/cuda RNG. Each trial runs 1 rollout and 1 PPO update.
  A trial passes iff learner and opponent actions are 100% identical and the max relative param error is <= 1e-5
  (per tensor: max|a-b| / max|a|).

## Laptop (PowerShell, repo root)
```
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$env:PYTHONPATH='src'
python scripts/test_t_c1a_equivalence.py --updates 5 --device cuda --deterministic
python scripts/test_t_c1a_equivalence.py --updates 5 --device cuda --legacy-twice
python scripts/test_t_c1a_equivalence.py --updates 5 --device cuda --legacy-twice --deterministic
python scripts/test_t_c1a_equivalence.py --per-step 5 --device cuda --deterministic
```
(The second command reruns the control without determinism to reproduce the original divergence.)

How to read the results:
- If the first command prints `[equiv] PASS`, T-C1a passes.
- Otherwise, if `--per-step 5` prints `[per-step] PASS`, T-C1a passes under the fallback criterion.
  `[legacy-twice] DIVERGED` is the evidence that the cache is not the cause.

## Box cpu results (torch 2.14.1+cpu)
- `--updates 5 --deterministic` gives [frozen] OK and [equiv] PASS. Without the flag it also gives PASS.
- `--legacy-twice --deterministic` gives IDENTICAL (1289/1289 opponent moves).
- `--per-step 5 --deterministic` gives PASS. Every trial had 0 action diffs and a relative error of 0, with params bit-identical.
- cuda with `CUBLAS_WORKSPACE_CONFIG` unset exits with a clear error (checked).
