# R04 结果

状态：**PASS**。没有正式训练。GPU 小时 = 0。

A5（`botzone_a5_v2`）和 A6（`botzone_a6_v2`）共用规则、公开特征、主干规模、数据方法和零训练预算。A6 只多一个 `quarter_pot`（id 5）。旧的 `botzone_5action_v1` 留作 V6 对照，哈希对不上就拒绝加载。

1000 个边界状态（种子 20261031）上，非法动作和超栈为 0，重复筹码响应没有被当成两次执行。公共动作按筹码响应上的概率质量比较，不按 id 对齐。

`quarterpot_effective_new_coverage` = **0.224**。另外 386 个局面里四分之一池合法化后等于旧尺寸，390 个局面里这一档不可用。不宣称已经获得小注能力。

裁判 SHA256 仍是 `b70702fdcd28333f91087f1a3dc3ede1ca59737404622c64dc6dbf7e9206e544`。

pytest：`python3 -m pytest -q --override-ini='addopts=' --tb=line tests/test_r04_action_schema.py` → 6 passed。
