# R03 四层归因

状态：**PASS**。GPU 小时 = 0。裁判 SHA256 `b70702fdcd28333f91087f1a3dc3ede1ca59737404622c64dc6dbf7e9206e544`。V6 npz SHA256 `dfd57d307c6d4d9eb5d789ba145370792752ea85f5f43ba57b5dd787e3b4faaa`。

同一公开信息集上的四层：

1. Teacher 混合：`MISSING`。SuperStudent / 150..350 快照没有附在这台机器上，没有计算 own-reach，也没有用无权平均补标签。窗口在 `teacher_window_manifest.json`（after_exclusive=149，through=350）。
2. 冻结学生：`MISSING`。`policy_best.pt` 未附带。
3. PPO raw：V6 权重在部署护栏之前的掩码 softmax。
4. 最终部署：同一向量经过 V6 包里的 v10.2 护栏。护栏和 raw 分开记录。

PyTorch 与 NumPy 在同一套 FP32 权重、同一批输入上的最大概率绝对误差是 `3.23310777455843e-07`（上限 `0.0001`）。
部署概率的非法动作质量最大值是 `0.0`，raw 是 `0.0`。
同一部署包连跑两遍，离散诊断（合法动作、argmax、1e-8 概率刻度、证明等级）一致。

反例 29 个：硬证明河牌绝对坚果 8 个，待研究 21 个。每个文件在 `counterexample_suite/`，带 `source` 和 `proof_grade`。

河牌绝对坚果用官方 `compare_full_cards` 对所有对手底牌穷举，Hero 严格赢下每一手时，弃牌严格劣于跟注或过牌。转牌坚果、顶对、听牌、公共双对和超池再加注只记公平率线索，不写成零弃牌真理。

## 基线缺陷

下面这些旧 V6 raw 弃牌超过 1%。这是基线缺陷，不因此判 R03 失败。新模型的 1% 门槛留到 R08/R09/R13。过牌到河牌的同一坚果上 raw 弃牌是 0；超标出现在面对下注的信息集。有的花色置换上护栏把弃牌清掉，有的没有。raw 已经超过 1%，所以层级记在 PPO，不把护栏当成这批缺陷的来源。

- `river_ks_nut_flush_facing_bet`：raw 弃牌 0.015752，部署弃牌 0.000000
- `river_ks_nut_flush_facing_bet__suit1`：raw 弃牌 0.118990，部署弃牌 0.118990
- `river_ks_nut_flush_facing_bet__suit2`：raw 弃牌 0.071408，部署弃牌 0.071408
- `river_ks_nut_flush_facing_bet__suit3`：raw 弃牌 0.015718，部署弃牌 0.000000

## 尚未归因

这些局面没有可证明的支配动作，Teacher 和冻结学生又都缺失，所以不能把偏差归到蒸馏或 Teacher。

- `turn_nut_flush`
- `turn_nut_flush__suit1`
- `turn_nut_flush__suit2`
- `turn_nut_flush__suit3`
- `flop_74_top_pair_gutshot`
- `flop_74_top_pair_gutshot__suit1`
- `flop_74_top_pair_gutshot__suit2`
- `flop_74_top_pair_gutshot__suit3`
- `river_board_two_pair`
- `river_board_two_pair__suit1`
- `river_board_two_pair__suit2`
- `river_board_two_pair__suit3`
- `flop_nut_flush_draw`
- `flop_nut_flush_draw__suit1`
- `flop_nut_flush_draw__suit2`
- `flop_nut_flush_draw__suit3`
- `flop_facing_overbet_reraise`
- `flop_facing_overbet_reraise__suit1`
- `flop_facing_overbet_reraise__suit2`
- `flop_facing_overbet_reraise__suit3`
- `flop_85_top_pair_gutshot`

## 层级

硬证明局面上，raw 弃牌超过 1% 归到 PPO；raw 不超过、护栏把弃牌抬过 1% 归到护栏。部署发出的筹码响应若被裁判拒绝，归到编码。这次没有发现要先修内核才能继续的系统性规则错误。

