# R02 结果

状态：**PASS**。训练内核按未修改的官方裁判 `Holdem.player_action` 逐步转移。没有正式训练，GPU 小时为 0。

实现提交：`30cd4b6c0c79dc2527e0e5c97f9ae20d71f3c564`。pytest 记录在 `eval/r02/pytest_kernel.txt`（13 passed）。

## 检查

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| 合法/拒绝 | 0 差异 | `engine_diff_report.json`：边界 24593 次探测 + 回放 7785 个决策 + 随机轨迹 31636 个决策 |
| 下一行动者、底池、筹码、街、摊牌收益 | 0 差异 | 同上，`step_mismatches = 0`，`payoff_mismatches = 0` |
| 每手筹码守恒 | 通过 | `conservation_failures = 0` |
| 策略视图不含对手底牌或未发牌 | 通过 | `policy_view_leaks = 0` |
| clone 不互相污染 | 通过 | `clone_isolation_failures = 0`（回放 41、边界 239、随机 200） |
| 种子 | 已记录 | 边界 `20261031`，随机轨迹 `404` |
| 裁判文件 SHA | 未改 | `b70702fdcd28333f91087f1a3dc3ede1ca59737404622c64dc6dbf7e9206e544` |
| GPU | 0 | 只跑了 CPU 差分和 pytest |

## 语料

本机没有 Mac 上的 33 场 `botzone_replay_analysis_20261030`。整手回放用的是仓库里的 `tests/fixtures/botzone_real.jsonl.gz`：41 场、2050 手、7785 个决策。这是 `tests/test_botzone_referee_diff.py` 已经在用的夹具。

另外：边界状态 1003（种子 20261031），随机合法轨迹 10000 手 / 200 场 50 手比赛（种子 404）。`residual_mismatches = 0`。

## 限制

- 33 场 Mac 回放树不在这台机器上。父任务可以在那棵树上再跑 `eval/botzone_kernel_diff.py` 的回放。
- `legal.apply` 仍会在「正数加注但没有抬高本街最大投入」时结束本街。内核的转移不走 `apply`，走裁判。旧的 5 动作 id 只通过 `LegacyFiveActionBridge` 换成实际筹码响应后再 `step`。
