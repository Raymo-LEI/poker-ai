# R03 结果

状态：**PASS**。没有正式训练。GPU 小时 = 0。

实现提交：`e7feb6fce0edaf2429c2f56f953dc6a172e4cfc1`。

## 检查

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| 同一部署包的离散诊断一致 | 通过 | `attribution_summary.json` 的 `discrete_diagnostics_identical` |
| FP32 PyTorch 对 NumPy | 通过 | 最大绝对概率误差 `3.233e-7`，上限 `1e-4` |
| 无非法概率质量 | 通过 | 部署侧最大非法质量 0 |
| 每个反例有来源和证明等级 | 通过 | 29 个反例，`hard_dominated` 或 `research_pending` |
| 根因层级 / 未归因 | 通过 | 河牌坚果面对下注的 raw 弃牌归到 PPO；待研究局面列入 `unattributed` |
| 旧 V6 raw 弃牌 >1% | 基线缺陷，不是失败 | `baseline_defects` |
| 裁判文件 | 未改 | `b70702fdcd28333f91087f1a3dc3ede1ca59737404622c64dc6dbf7e9206e544` |
| Teacher 窗口 | 已参数化 | `teacher_window_manifest.json`，150..350，零到达不补无权平均 |

Teacher 快照和冻结学生 `policy_best.pt` 没有附带，这两层是 `MISSING`，没有编造标签。

pytest：`python3 -m pytest -q --override-ini='addopts=' --tb=line tests/test_r03_attribution.py` → 5 passed。
