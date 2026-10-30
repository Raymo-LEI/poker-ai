# A5 / A6 动作契约迁移

`botzone_5action_v1` 仍是冻结的 V6 对照：`fold`、`check_call`、`raise_min`（裁判最小正数额外筹码）、`raise_large`（`max(2 * safe_extra, pot)`）、`all_in`。它的实现留在 `alphanl.botzone.actions`。A5/A6 实验不读取这套 id 含义，也不改裁判。

`botzone_a5_v2` 与 `botzone_a6_v2` 共用同一套规则、公开特征、主干规模、数据方法和训练预算（本次预算是 0）。动作表只有一处差别：A6 在 id 5 增加 `quarter_pot`。

| id | A5 / A6 名称 | 筹码响应 |
| --- | --- | --- |
| 0 | fold | `-1` |
| 1 | check_call | `0` |
| 2 | half_pot | 未面对下注时是半池；面对下注时是待跟再加半池。再抬到裁判最小正数 |
| 3 | pot | 同样规则，比例是整池 |
| 4 | all_in | `-2` |
| 5 | quarter_pot | 仅 A6。比例是四分之一池 |

舍入是半入（`(pot * numer + denom // 2) // denom`）。正数额外筹码必须严格小于剩余筹码；不够则这一档不是独立尺寸，全下仍由 `-2` 表示。两个 id 算出同一筹码时，只保留较小的 id，不把同一次投入计成两个可执行结果。

## 拒绝兼容

任何旧标签、快照或 reservoir 进入 A5/A6 加载器时，必须带 `action_schema_id`、`action_schema_sha256` 和 `num_actions`。缺字段、id 不同、哈希不同或宽度不同都抛 `SchemaMismatch`，不默默当成五动作，也不把向量填宽。`botzone_5action_v1` 因此不能喂给 A6。A6 的六维头也不能装进宽度为 5 的实验头。

仓库里仍写死宽度 5 的 Teacher、采样、loss、mask、对手、评测、PPO、导出和护栏列在 `hardcoded_action_width_audit.json`。它们保持 `botzone_5action_v1` 对照，实验路径不改这些头。V6 权重没有被覆盖。

## QuarterPot

四分之一池在合法化之后经常落到半池、整池或最小加注上，这时它被去重，不另计一次执行。`quarterpot_effective_new_coverage` 只统计它变成 A5 没有的筹码额的局面比例。这个比例不是小注能力。
