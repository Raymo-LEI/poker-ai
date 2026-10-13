# Slumbot client: how to run it

Files:
- `eval/slumbot_client.py`: the client. Modern Python 3.9+ with `requests`. It plays Slumbot (200bb), applies the 50bb cap, logs to jsonl, resumes, and summarizes.
- `eval/alphanl_worker.py`: AlphaNLHoldem inference in a separate Python 3.7 / TF1.15 / Ray 0.8.3 process. It talks JSON lines over stdin/stdout and loads `src/alphanl.AlphaNLHoldemAgent` from poker-ai PR #2.
- `tests/fake_repo/`: a stand-in AlphaNLHoldemAgent that needs no TF. It is only for testing the subprocess path.

## Stack mapping (`--stack-bb 50`, the default)
- Both players' effective stack is capped at 5000 chips (50bb). The agent never sees, and can never bet, past `cap - committed`.
- If Slumbot has committed >= 50bb, the agent sees Slumbot as all-in and gets only fold/call. Calling pays the real amount Slumbot bet, and the result comes from Slumbot's real `winnings`.
- For AlphaNL, the observation is rebuilt in rlcard-1.1.0 units (1 chip = 50 Slumbot chips, so stakes are 100 at 50bb). Abstract actions are 0 fold, 1 check/call, 2 half-pot, 3 pot, 4 all-in(to cap). Slumbot's bet sizes are bucketed into these by size relative to the pot (ratio < `--half-pot-max`, default 0.75 → half-pot; < `--pot-max`, default 1.6 → pot; otherwise all-in). Our bets are clamped into Slumbot's legal range. A half-pot raise that is below Slumbot's minimum raise is raised to the minimum.
- Per-hand tags: `truncated` is true when at some decision the agent's view differed from reality (Slumbot above the cap, or the agent at the cap while chips remained). `past_cap` is true when either side's final commitment was over 50bb.

## Resume, rate limit, concurrency
- `--hands` is the TOTAL target for the log file. On start, the client counts the valid json lines in `--log` and plays only the rest, so re-running the same command after a crash, reboot or Ctrl-C continues where it stopped. Each line is flushed and fsynced after its hand. A half-written last line is ignored and rewritten. A hand that is cut off mid-play never gets logged, so it is replayed as a new hand.
- `--workers` defaults to 2 and is capped at 4. Each worker runs its own Slumbot session and, for alphanl, its own model subprocess.
- `--min-interval` (default 0.2s) is a global gap between requests. HTTP 429/5xx and network errors back off exponentially (up to 60s) with retries. A hand that fails is dropped and a fresh session starts.

## Commands
```
python eval/slumbot_client.py --hands 100 --agent random --log runs/random.jsonl
python eval/slumbot_client.py --summarize runs/random.jsonl     # mbb/hand, 95% CI, by position, truncated vs not
```
Agents: `random`, `call`, `abstract-random` (random over the 5 AlphaNL actions; same mapping path as the model), `alphanl`.

## Laptop setup (Ryzen 9 5900HX / RTX 3070 Laptop 8GB / 32GB, multi-night runs)
The model runs on CPU. The run is limited by the network (~1–5 s/hand per worker), so GPU makes no difference. TF1.15 does not support Ampere anyway (it would need nvidia-tensorflow).

1. Client env (any Python 3.9+): `pip install requests`
2. Model env (conda, from the poker-ai repo checked out at PR #2):
   ```
   git clone https://github.com/Raymo-LEI/poker-ai && cd poker-ai && git fetch origin pull/2/head:pr2 && git checkout pr2
   git submodule update --init --recursive
   conda create -n alphanl python=3.7 -y && conda activate alphanl
   pip install -r requirements-alphanl.txt
   PYTHONPATH=src python eval/eval_step1.py --hands 200     # sanity check from PR #2
   ```
   Get that env's interpreter path: Linux `conda run -n alphanl which python`, Windows `conda run -n alphanl where python`
   (e.g. `C:\Users\<you>\miniconda3\envs\alphanl\python.exe`).
3. Smoke test (20 hands):
   ```
   python eval/slumbot_client.py --agent alphanl --hands 20 --alphanl-python <py37 path> --alphanl-repo <poker-ai path> --log runs/alphanl_smoke.jsonl
   ```
4. Unattended, resumable run (5000-hand checkpoint, then extend `--hands` to 50000 later with the same `--log`):
   - Linux:
     ```
     nohup sh -c 'until python eval/slumbot_client.py --agent alphanl --hands 5000 --workers 4 \
        --alphanl-python ~/miniconda3/envs/alphanl/bin/python --alphanl-repo ~/poker-ai \
        --log runs/alphanl_50bb.jsonl; do sleep 60; done' >> runs/alphanl_50bb.out 2>&1 &
     ```
   - Windows (PowerShell; the loop restarts after crashes; run `powercfg /change standby-timeout-ac 0` so the laptop doesn't sleep on AC):
     ```
     while ($true) { python eval\slumbot_client.py --agent alphanl --hands 5000 --workers 4 `
        --alphanl-python C:\Users\<you>\miniconda3\envs\alphanl\python.exe --alphanl-repo C:\path\poker-ai `
        --log runs\alphanl_50bb.jsonl; if ($LASTEXITCODE -eq 0) { break }; Start-Sleep 60 }
     ```
     Running it across reboots: create a Task Scheduler task "At log on" that runs this script. Resume makes repeat launches safe. Never run two processes on the same `--log` at once.
   - Expected throughput: about 4 workers × 0.2–1 hands/s, so 5000 hands is roughly 1–6 h and 50000 hands is a few nights.
5. Results (headline = `baseline_not_truncated`; truncated hands are systematically disadvantaged for the capped 50bb agent vs 200bb Slumbot, so they are reported separately as `truncated`; `all` is reference only): `python eval/slumbot_client.py --summarize runs/alphanl_50bb.jsonl`
