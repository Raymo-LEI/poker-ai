# Botzone TexasHoldem2p 裁判差异报告（T-M14 + T-M14b）

对照的是未经修改的官方裁判 `tests/fixtures/botzone_judge.py`（`Holdem.player_action`）。规格复述只做二次交叉检查，文件头标了 `spec-based reference, NOT the real referee`。T-M14b 只改了最小加注：`src/alphanl/botzone/legal.py` 的 `safe_extra` 与 `round_raise` 记账。训练循环、PPO、网络、tokenizer、奖励、配置、动作空间宽度都没有改。

**第 2 项现在与裁判一致。** 最小额外筹码是 `2 * round_raise - own_round_bet`，至少为 1（`botzone_judge.py` 299-303）。`bound_b` 存的是 `2 * round_raise`。41 场实牌 2050 手整手重放，`our_rejected_real_actions = 0`，`engine_mismatches = 0`。原先被拒绝的 41 个加注（含 15 个翻前 150）现在都在 `legal_response` 里，并且 `apply` 能把整手走完。

第 5 项没有行动顺序错误。差分里 `we_legal_judge_illegal = 0`，`min_raise_mismatch = 0`：每个到达的局面上，我们的最小正数等于裁判的最小正数。`raise_min` / `raise_large` 一旦出现在掩码里，对应整数都被裁判接受。

| 项 | 代码位置 | 我们的行为 | 裁判行为 | 状态 | 最小复现 |
|----|----------|------------|----------|------|----------|
| 1. 响应语义 | `src/alphanl/botzone/legal.py:50-52`、`273-278`、`192-196` | `-1` 弃牌，`-2` 全下，`0` 过牌/跟注，正整数是这一下额外投入的筹码。非 `int`（含 `100.0`、`True`）直接 `ValueError`。非法响应也 `ValueError`，不改成弃牌。 | `201-203`、`269-272`：同样四个码；正整数在 `303-307` 做 `round_player_bet += bet`、`pot += bet`，不是 raise-to。`414-427`：`100.0` 当成整数，非整数（含 `True`、`150.5`）返回 `None`。`480-487`：解析失败或 `ValueError` 都改成弃牌。 | differs | 翻前小盲位，裁判接受 `150` 和 `150.0`，拒绝 `149` 和 `150.5`。我们 `apply` 接受整数 `150`，拒绝 `150.0`。我们实际发出去的是 Python `int`，这四个码的含义一致。 |
| 2. 最小加注 | `legal.py:142-160`、`242-245`、`293-296`；`actions.py:44-46` | 最小额外筹码 `= max(1, 2 * round_raise - own)`。`bound_b` 存 `2 * round_raise`：盲注之后 200，每条翻后街 100。每一次被接受的正数 `bet` 都做 `bound_b = max(bound_b, 2 * bet)`，包括新总额没有超过街最大的那一次。跟注和全下不改它。正数必须严格小于剩余筹码。 | `299-303`：合法条件是 `bet + own_round_bet >= 2 * round_raise`，所以最小额外筹码 `= 2 * round_raise - own`（小于 1 时最小正数是 1）。`303`：`round_raise = max(round_raise, bet)`，每次被接受的正数 `bet` 都更新。`291-298` 跟注不改。`285-290` 全下不改。盲注走加注分支（`333-334`），翻前走完是 100。`261` 翻后重置为 `big_blind // 2 = 50`。 | consistent | 见下面三张表。翻前小盲最小加注两边都是 150。 |
| 3. 筹码不够 | `legal.py:181-185` | `0` 只在剩余筹码**严格大于**待跟时给出。正整数只取 `[safe, 剩余)`，所以加注额严格小于剩余。不够时掩码里只剩 `-1` 和 `-2`。 | `293-294`：跟注 `chips <= inc` → `INSUFFICIENT_CHIPS`。`301-302`：加注 `chips <= bet` → 同样。正好花光必须发 `-2`，不能发 `0` 或等于剩余的正数。 | consistent | 待跟等于剩余时，我们的集合是 `{-1,-2}`，裁判对 `0` 和「等于剩余的正数」都抛 `INSUFFICIENT_CHIPS`。 |
| 4. 对手全下之后 | `legal.py:178-179` | `any(all_in)` 时只返回 `{-1,-2}`。双方全下后 `_end_betting_round`（`284-287`）结束下注，牌桌摊牌。 | `285-290`：全下把 `round_bet` 设为 `-2`。`291` 和 `299` 都要求 `round_bet >= 0`，否则 `309-310` `INVALID_BET`。双方 `round_player_bet` 都是 `-2` 时 `320-323` 进入 `_next_round`，`261-267` 在没有人还能说话时把剩余公共牌发完。 | consistent | 对手已 `-2` 后发送 `0` 或任意正数：裁判 `INVALID_BET`（随后当成弃牌）。我们根本不把这些放进掩码。 |
| 5. 座位与行动顺序 | 训练：`legal.py:102`（翻前小盲先）、`264-265`、`296`（每条新街 `actor = (dealer+1)%2`）。环境注释：`src/alphanl/transformer/env.py:13-14`。观测：`src/alphanl/transformer/botzone_table.py:111-115` 的 `current_player` 就是这同一个 `state.actor`。 | 庄家下大盲。非庄家下小盲，**翻前先动，翻后也先动**。观测转换没有第二条行动顺序。main 上没有单独的 Botzone JSON adapter；未合并的 `origin/pr/t-d2` 不在这条训练路径上。 | `332-334`：盲注后 `_next_player` 停在非庄家。`259-265`：新街先把 `round_idx` 设成庄家，再 `_next_player()`（`227-230`），仍是非庄家。`467`、`508`：`dealer = (initdata.dealer + hand) % 2`。 | consistent | `dealer=0` 时翻前和翻后第一个说话的都是座位 1。夹具 2050 手、每条街的第一次行动全部是非庄家（小盲）：翻前 2050、翻牌 830、转牌 658、河牌 496。重放 7785 个决策，行动者/街次不一致为 0。 |
| 6. 发牌与牌编码 | 编码：`src/alphanl/botzone/table.py:36-52`，`src/alphanl/transformer/botzone_table.py:41-52`。发牌：`table.py:7-16`、`28`、`294-308`。词表：`src/alphanl/transformer/schema.py:24` 的 `SUITS="CDHS"`，`tokenizer.py:65-66`。 | 牌号转换与裁判相同：`id % 4` 是花色（红桃 0、方块 1、黑桃 2、草花 3），`id // 4 + 2` 是点数。进 transformer 之前先换成 `H2` 这种字符串，词表下标是 CDHS 的花色主序，词表里不出现 0–51。发牌是小盲、大盲交替各两张，并且烧牌（公共牌下标 5,6,7,9,11）。 | `32-36`、`7-10`：编码同上。`327-331`：从牌组**末尾** `pop`，玩家 0 连续两张，然后玩家 1 连续两张。`235`、`238`、`241`：翻牌 3、转牌 1、河牌 1，不烧牌。 | differs | 编码往返 `0..51` 与 `Card.from_int` / `botzone_id_to_card` 一致。同一副牌组数组不会发出同样的底牌：我们烧牌，而且按小盲/大盲交替，不是「玩家 0 两张再玩家 1 两张」。随机洗牌时牌的集合分布不变。和裁判对拍要用请求里的牌，不要用我们的牌序去猜。 |
| 7. 请求格式 | 训练观测是 `BotzoneEnvTable.snapshot`（`botzone_table.py:158-181`），不解析裁判 JSON。全下筹码：`legal.py:228-229` 把剩余栈全部投入。 | 盲注不进动作序列。`my_chips` 在内部是盲注之后的剩余。全下在环境里知道具体筹码，因为 `-2` 表示投入当时的全部剩余。 | `373-388`：`num_players`、`dealer_id`、`my_id`、`my_chips`（盲注已扣）、`my_cards`、`public_cards`、`history`、`matchdata`。`335`：盲注记完后清空 history。`279-283`：弃牌记录没有 `round`。`313-317`：全下的 `action` 就是 `-2`，没有金额。跟注/过牌的 `action` 是 `0`，加注的 `action` 是本次额外筹码。 | consistent | 全下金额 = 行动前 `player_chips`（`287-288` 把这笔加入底池并把筹码清零）。我们发 `-2` 时做的是同一件事。main 没有从「没有金额的 -2」反推对手筹码的 JSON 解析器；环境自己握着栈。 |
| 8. 结算 | `legal.py:45`、`81-108` 每手 20000。`table.py:95-114` 平分，余数给座位号较小的一方。`src/alphanl/botzone/rule_eval.py:26`、`95-113`：50 手，累计筹码 / 100。 | 每手重置 20000。评分是累计赢的筹码除以大盲 100。平分用整数，奇数底池多 1 筹码给较小座位。 | `432-433`、`452`、`468`、`509`：每手 20000，默认 50 手。`396-398`：`score = 累计赢筹码 / big_blind`。`343-347`：`pot / 人数`（真除法）。`499`：`win_chips = 最终筹码 - 均值`。 | consistent | 41 场夹具：弃牌结算与裁判一致；有 5 张公共牌的摊牌，我们的 `hand_rank` 与裁判 `compare_full_cards` 胜负一致，平分结果一致。这些手的平分底池都是偶数，余数规则没有被用到。两人每条街投入相同，所以能摊牌的底池是偶数，余数分支在裁判可达的平分里到不了。 |

## 第 2 项的数字

`round_raise` 仍是「本街单次 `bet` 的最大值」（盲注算进去，`botzone_judge.py` 303、333-334）。「最大累计」是 `round_bet`。`bound_b` 存 `2 * round_raise`，所以 `safe_extra = max(1, bound_b - own)`。下表每一行差距都是 0。旧的下界 A 还导出（`lower_bound_a`），但不再参与合法性。

### (a) 翻前，盲注 50/100 之后

| 局面 | own | round_raise | round_bet | 裁判最小额外 | 我们的 safe | 差距 |
|------|-----|-------------|-----------|--------------|-------------|------|
| 小盲第一个说话 | 50 | 100 | 100 | 150 | 150 | 0 |
| 小盲跟注后，大盲可以加注 | 100 | 100 | 100 | 100 | 100 | 0 |
| 小盲加注 150 之后的大盲 | 100 | 150 | 200 | 200 | 200 | 0 |

复现：`dealer=0`，`new_hand` 后小盲发 `150`。裁判 `player_action(150)` 成功。我们 `150 in legal_response`，`safe_extra == 150`。`raise_min` 发出 150。

### (b) 翻后，每条街重置

| 局面 | own | round_raise | round_bet | 裁判最小额外 | 我们的 safe | 差距 |
|------|-----|-------------|-----------|--------------|-------------|------|
| 翻后第一注（`round_raise` 重置为 50，`bound_b` 为 100） | 0 | 50 | 0 | 100 | 100 | 0 |
| 面对 100 的下注 | 0 | 100 | 100 | 200 | 200 | 0 |
| 原下注者面对 200 的再加注 | 100 | 200 | 200 | 300 | 300 | 0 |

翻后第一注两边都是 100。再加注不再用 `2 * 最大累计`：自己已经投入的 100 要减掉，所以最小额外是 300，新总额 400。旧下界 A 在这一行仍是 400，只是不再当门槛。

### (c) 连续再加注

| 局面 | own | round_raise | round_bet | 裁判最小额外 | 我们的 safe | 差距 |
|------|-----|-------------|-----------|--------------|-------------|------|
| A 下注 100，B 再投入 300，轮到 A（新总额至少 600） | 100 | 300 | 300 | 500 | 500 | 0 |
| 翻前依次 `0, 100, 101, 102` 之后 | 201 | 102 | 302 | 3 | 3 | 0 |

主人的例子对上了：A 的新总额至少 `100 + 500 = 600`。`raise_min` 发出的额外筹码就是 500。

## 不够跟注的「加注」

裁判的加注分支不检查 `bet + own >= round_bet`。最小正数可以小于待跟。我们现在用同一条公式，不再要求 `safe > 待跟`。

最小复现（`dealer=0`，翻前）：小盲 `0`，大盲 `100`，小盲 `101`，大盲 `102`。此时小盲 own=201，`round_raise=102`，`round_bet=302`。`player_action(3)` 成功，新总额 204，仍小于 302。`player_action(2)` 是 `INVALID_BET`。我们 `safe_extra == 3`，`3` 在 `legal_response` 里，`2` 不在。

`apply` 仍只在新的累计**超过**街最大时才让对手重新行动（`legal.py:246-248`）。一笔不抬高街最大的正数会更新 `round_raise`（若它更大），但若对手此刻不需要行动，这条街会结束。5 档动作和 41 场实牌都没有走出这种下注：整手 `apply` 重放 `engine_mismatches = 0`。seed 0 的 5000 手探索也没有撞上「最小正数本身就小于待跟」的局面（`undercall_accepts = 0`）；脚本化的 `bet=3` 已经钉在测试里，两边的最小值都是 3。

## raise_min 和 raise_large

`actions.py` 的换算没有改。

- `raise_min` 就是 `safe_extra`（`actions.py:44-46`），也就是裁判的精确最小值。
- `raise_large` 仍是 `max(safe * 2, pot)`。`safe >= 1` 时 `safe * 2 > safe`，底池只可能把这一档抬得更高，所以它不会和 `raise_min` 变成同一次响应。`legal_action_ids` 不按响应整数去重：两个 id 各自在自己的整数属于 `legal_response` 时留在掩码里。这次不需要为「重复」加一条屏蔽。
- 最小加注变小只会让 `raise_large` 变小或不变，不会把它新推到剩余筹码之外。它若留在掩码里，就落在 `[safe, 剩余)` 里，裁判接受。

实测（seed 0）：我们自己的动作集 20000 手（随机合法、M7 规则 bot、未训练的 5 档策略轮流）加裁判可达的 5000 手探索，共 78663 个决策。

- `we_legal_judge_illegal = 0`
- `we_illegal_judge_legal_responses = 0`（其中最小加注这一类 `min_raise_mismatch = 0`）
- `mapped_raise_illegal = 0`
- `desyncs = 0`
- `conservative_decisions = 0`
- `gap_max = 0`
- 规格交叉检查：`action_amount` 与裁判不一致 0 次；`street_total` 不一致 34100 个决策。这是规格复述，不是我们的合法性。翻后 `round_bet=0` 时，把 `round_raise` 当成最大累计会得到 0，于是 99 会被规格放行，裁判拒绝（`2 * 50 = 100`）。

未训练策略用的是同一个 `PokerTransformer.act` 和同一张环境掩码，宽度收成 1 层 / `d_model=32`，只为了在 2 万手里采样。合法性由掩码和 `response_for_action` 决定，不由层数决定。

## 41 场整手重放

夹具 `tests/fixtures/botzone_real.jsonl.gz`：41 场、2050 手、7785 个决策。行动顺序不一致 0。裁判拒绝真实动作 0。映射动作非法 0。**我们拒绝的真实动作 0。** `apply` 与裁判的行动者、街次、筹码、底池、`round_raise` 不一致 0（`engine_mismatches = 0`）。

T-M14 记录过的 41 个加注现在全部通过。当时的 17 种形态（次数加总 41）是：

| 次数 | 街 | 真实额外筹码 | 旧 safe | 裁判最小（现在的 safe） | 待跟 |
|------|----|--------------|---------|--------------------------|------|
| 15 | 翻前 | 150 | 200 | 150 | 50 |
| 5 | 翻前 | 400 | 500 | 300 | 150 |
| 3 | 翻前 | 800 | 900 | 700 | 350 |
| 2 | 翻前 | 450 | 500 | 300 | 150 |
| 2 | 翻前 | 900 | 1100 | 650 | 300 |
| 2 | 翻前 | 820 | 920 | 720 | 360 |
| 2 | 翻前 | 410 | 500 | 300 | 150 |
| 1 | 翻前 | 100 | 200 | 100 | 0 |
| 1 | 翻前 | 3320 | 3520 | 2870 | 1310 |
| 1 | 翻前 | 1600 | 1800 | 1350 | 650 |
| 1 | 翻牌 | 14600 | 16040 | 12899 | 6319 |
| 1 | 翻前 | 1800 | 2300 | 1250 | 600 |
| 1 | 翻牌 | 4210 | 4720 | 2940 | 1380 |
| 1 | 翻牌 | 8420 | 10380 | 6060 | 2830 |
| 1 | 翻前 | 1700 | 1800 | 1150 | 450 |
| 1 | 翻前 | 870 | 1000 | 550 | 250 |
| 1 | 翻前 | 870 | 920 | 720 | 360 |

最常见的一条：翻前小盲开池加注 150（新总额 200）。`apply(150)` 现在成功。

还和裁判不同、但不是最小加注的只有入站浮点：`150.0` 裁判接受，我们 `apply` 拒绝。我们发出去的是 `int`。

## 已有检查点会看到什么

策略采样的是 5 个动作 id，不是原始筹码。id 的合法性只增不减：

- `raise_min` 的整数变小或不变，所以原来放得进剩余筹码的最小加注，现在仍然放得进。
- `raise_large = max(2 * safe, 底池)` 只会变小或不变，原来合法的大注仍然小于剩余筹码。
- 弃牌、跟注、全下的条件没变。对手全下之后仍然只有 `-1` 和 `-2`。

新出现的选项：翻前小盲的 `raise_min` 从 200 变成 150。完整的 `legal_response` 还包含 151 到 199，5 档头不直接采样这些整数。旧最小加注放不进筹码、新的最小值放得进的局面里，`raise_min`（有时还有 `raise_large`）会新出现在掩码里。

观测会跟着变。`tokenizer.encode` 的 `legal_mask` 是这 5 位。某一档从非法变成合法时，对应位从 0 变成 1。动作历史 token 用实际投入的 `size_bucket` 和底池比例，CLS 的数值通道带着底池、筹码和待跟。同一 id 现在投入更少（开池是 150 而不是 200），这些特征沿着手牌轨迹会换一批数。网络权重没有改，旧检查点仍然能载入，采样到的 id 在新环境里仍然合法。掩码和筹码特征的分布会移动。本 PR 保持草稿，等正在进行的训练评估结束后由主人决定是否合并。

## 测试计数

```
PYTHONPATH=src python3 eval/botzone_referee_diff.py --hands 20000 --explore-hands 5000 --seed 0
```

20000 手我们的动作集（random / rule / 未训练策略）+ 5000 手裁判探索。决策 78663。`we_legal_judge_illegal = 0`，`min_raise_mismatch = 0`，`we_illegal_judge_legal_responses = 0`，`mapped_raise_illegal = 0`，`desyncs = 0`。夹具重放：2050 手、41 场、7785 个决策，拒绝与引擎不一致都是 0。

```
python3 -m pytest -q --override-ini='addopts=' --tb=line
```

`172 passed in 51.70s`。M1/M2 的筹码守恒与零和（`tests/test_botzone_conservation.py`，1000 手）在这一次里面通过。

数字来源：`eval/botzone_referee_diff_summary.json`。

## 没有改的模块

`table.py` 在结算时原样拷贝 `bound_b`。盲注之后它仍是 200，翻后重置仍是 100，所以拷贝还对。`actions.py`、规则 bot、`botzone_table`、tokenizer 都只调用 `safe_extra` / `legal_action_ids`，没有自己的最小加注。`lower_bound_a` / `lower_bound_b` 仍从 `alphanl.botzone` 导出，前者不再当门槛。

差分脚本 `eval/botzone_referee_diff.py` 改了记账：探索和夹具重建 `HandState` 时，`bound_b` 必须在每一笔正数 `bet` 上更新，否则会把已经对齐的最小值测成不齐。它还加上了整手 `apply` 重放和 `min_raise_mismatch`。这是测量，不是训练路径。

## 还没对齐的产品选择

- 发牌顺序和烧牌与裁判不同。随机发牌不影响分布。
- 奇数底池的 1 筹码给较小座位。裁判可达的平分里底池是偶数，夹具里没有触发。
- 入站的整值浮点（`150.0`）。我们只接受 `int`。
