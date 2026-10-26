# Phase 1 / Phase 2：标准双人 200BB 与 SD-CFR

本次实现对照 `CFR.md` 的前两个 Phase。现有 `PokerTransformer`、PPO、PFSP、
PPO 配置和 checkpoint 格式保留；没有实现 Phase 3 的 50 手残差 exploiter，
也没有进行 200BB 长时间训练或声称已得到不可剥削的 GTO 权重。

## 规则与兼容性

新增 `alphanl.transformer.hunl.HunlTable`，固定每手重置双方 20,000 筹码，
SB=50、BB=100。固定座位 0 为 BB，座位 1 为 SB/button；翻前 SB 先行动，
翻牌、转牌、河牌 BB 先行动。收益以 BB 进入 CFR：筹码盈亏除以 100。

现有 `rlcard` 和 `botzone` 后端沿用旧规则。特别是仓库的 Botzone simple
裁判 fixture 在翻后也是 SB 先行动，与这次要求的标准 HUNL 不同。
新增路径按本次要求实现，不能据此宣称与旧 simple 裁判等价。
审计、下注公式、任意下注额和快速牌力评估的细节见
[hunl_environment.md](hunl_environment.md)。

保留原 5 个行动、275 词表、CLS/洞牌/街标记/行动顺序和 4D 数值特征。
`step_raise_to(amount)` 接收真实的当前街累计 raise-to，历史 token 取最近的
原有底池比例桶，数值通道保留真实下注比例。
`HunlTokenizer` 的比例通道为 `pot_fraction / 10`，不截断超过 10 倍池的下注；
原 PPO tokenizer 仍按旧规则截断，输出完全不变。
规则与特征版本分别是 `standard_hunl_200bb_v1`、`hunl_exact_fraction_v1`，
保存、恢复和部署时均需一致。

`phevaluator` 的 5/6/7 张牌 perfect-hash lookup 直接参与新牌桌的 PokerKit
结算，避免枚举 21 个五张组合；legacy 牌桌的 evaluator 不变。

## 模型包装与 Regret Matching

`alphanl.cfr.RegretNetwork` 独立复制现有 backbone，读取原 CLS 的 GELU MLP
输出，新增 `Linear(128, 5)` regret head（645 个参数）。原始 PPO 模型仍为
**592,390 参数**，`128 / 4 层 / 4 头 / FFN 256` 的默认结构没有改变。
策略头、价值头和原 state dict 的键名均保留；包装内不用的两个旧 head 冻结。
给包装层传入一个 PPO 模型，不会改动那个 PPO 对象。

```python
from alphanl.cfr import RegretNetwork
from alphanl.transformer.model import PokerTransformer

ppo_model = PokerTransformer()
regret_model = RegretNetwork(ppo_model, fallback="uniform")
raw_signed_regrets = regret_model(batch)
regret_model.eval()
probabilities = regret_model.strategy(batch)
```

只对合法行动的正预测 regret 做归一化；若所有合法 regret 都不大于零，
默认在合法行动间均匀分配，也可配置 `argmax` 回退。非法行动概率严格为零，
空合法集报错。head 零初始化，因此首轮均匀探索。

训练目标保留**有符号的瞬时 regret**，不能把负目标截成零。
这是 Deep CFR 的 advantage 回归及正 regret matching；它不实现表格 CFR+
在每轮累加后截断累计 regret 的更新。

## 遍历、训练和平均策略

双方各持有一个 regret network 和一个 reservoir，避免混淆两方的 regret。
每轮 t 采用同时更新：

1. 固定双方当前网络 `theta_(t-1)`，即本轮策略 `sigma_t`。
2. 分别以两位玩家为 traverser。其节点展开**所有合法抽象行动**，包括当前
   概率为零的行动；对手节点只按其当前策略抽一个行动。根节点预抽完整牌堆，
   分支共享该机会样本，但观察仅含自己的洞牌和已公开的公共牌。
3. 节点记录 `r(a) = v(a) - sum_b sigma_t(b) * v(b)`；非法槽为零。
   对手和机会的到达概率已包含在采样中，不再乘一次，也不乘 traverser
   自己的到达概率。不裁剪博弈树为虚构的终局收益。
4. Algorithm R reservoir 对历史 infoset 样本均匀保留。按原始迭代 t' 回归：
   `mean(t' * sum_legal((predicted_regret - sampled_regret)^2))`。
   默认每轮重新初始化网络并从累计 memory 拟合；`reinitialize=false` 为
   可选 warm-start 实验。训练和采样均关闭 dropout，但保留训练梯度。
5. 保存本轮实际使用的 **sigma_t / theta_(t-1)**，而非误标更新后的网络。
   再用拟合结果进入下一轮。

多棵树以协作式 DFS worker 交错推进，按玩家分组进行批量 Transformer 推理。
`traversal_batch_size` 控制同时推进的树数；这是单进程的向量化推理调度，
没有实现多进程 CPU 牌桌 worker。相同 24 个 infoset 的测试中，batch=4
将 forward 次数由 24 降为 6。实际吞吐仍取决于牌桌复制和树大小，
没有声称已完成 GPU 性能优化。

`SnapshotPolicy` 实现 SD-CFR trajectory sampling：每手开始按 t 权重抽取
一个历史策略，整手固定使用它。不能每个行动重新抽策略，也不能直接平均
各网络在当前 infoset 的概率而忽略自身到达概率。

```python
from alphanl.cfr import SnapshotPolicy
from alphanl.transformer.hunl import HunlTokenizer

policy = SnapshotPolicy("runs/sd_cfr_hunl/strategies", seed=7)
policy.begin_hand(player=1)  # SB；整手只调用一次
batch = HunlTokenizer().encode(table.snapshot(1))
action = policy.act(batch).item()
```

默认 `snapshot_every=1` 保存全部轮次。改为 K>1 时，仅对保留的轮次按 t
加权，属于稀疏近似。最新 regret 网络的策略不等于 SD-CFR 平均策略，
当前没有把快照蒸馏为单个部署模型。

算法依据：[Deep CFR](https://proceedings.mlr.press/v97/brown19b.html)、
[Single Deep CFR](https://arxiv.org/abs/1901.07621)。

## 运行

在仓库根目录、Python 3.10+ 环境中：

```bash
pip install -r requirements-cfr.txt
export PYTHONPATH=src

# 检查默认 0.59M 配置，不进行训练
python scripts/train_cfr.py --dry-run

# 真实 200BB 牌桌的短冒烟，仍完整遍历抽样子树
python scripts/train_cfr.py --iterations 1 --traversals 1 --train-steps 1 \
  --device cpu --out-dir runs/cfr_smoke

# 正式训练入口
python scripts/train_cfr.py --config configs/cfr/hunl_200bb.json

# 恢复完整训练状态；该文件包含本地可信的优化器/RNG/memory 元数据
python scripts/train_cfr.py --config configs/cfr/hunl_200bb.json \
  --resume runs/sd_cfr_hunl/train.pt

# 小游戏算法校验：生产遍历器 + 表格 regret oracle
python -m alphanl.cfr.kuhn --iterations 1000
```

默认 buffer 容量为每位玩家 2,000,000；RAM 路径按需分块分配。
将 JSON 的 `buffer_directory` 设为 `runs/cfr_memory` 即使用磁盘 memmap，
容量可以配置到 10,000,000。长度 64 的样本约 1,313 字节，每位玩家
2M 约 2.63GB、10M 约 13.13GB，双方合计翻倍；另需模型和训练激活内存。
磁盘模式按需映射页面，不要求所有样本驻留 RAM。批次采样允许重复。

输出包括：

- `metrics.csv`：双方 loss、节点数、终局数、样本数、推理批次和 memory 大小。
- `strategies/iteration_XXXXXXX.pt`：双方历史策略权重，不含 optimizer/memory。
- `strategies/index.json`：当前训练历史中有效的策略列表。恢复旧轮次时排除
  后来产生的未来策略；那些文件不被删除。
- `train.pt` 和 `checkpoint_data/`：恢复所需的优化器、RNG、memory 与完整
  历史策略。**备份/移动时两者需一起保留**，引用路径相对检查点目录。
  磁盘 memory 用不超过 8MB 的块复制到不可变的 generation，避免全量复制
  进内存；RAM memory 则直接嵌入 `train.pt`。

恢复到新输出目录会携带已有历史策略。恢复到原目录会还原对应的指标历史，
原指标另存到 `history/`。旧的 checkpoint generation 不自动清理，只有确认
不再需要其对应 `train.pt` 后才可删除。配置的 memory 容量、模型、抽样参数、
特征版本和规则版本不一致时拒绝恢复。

## 已验证的范围

自动测试覆盖标准 200BB 每条街的顺序、off-tree 真尺寸、隐藏信息隔离、
lookup 结算、合法掩码、精确有符号 regret、零概率行动展开、批量推理、
reservoir 均匀性与磁盘恢复、线性加权 MSE、神经 regret 拟合、完整训练续接、
历史策略恢复，以及原 PPO 模型与训练回归。

固定种子 1、1000 轮 Kuhn 的表格 sanity check 使用生产遍历器，
exploitability 从 0.45833 降至 0.01580，玩家 0 价值 -0.05501，接近
理论值 -1/18。最佳响应受信息集约束，不知道对手私牌。
另有 Transformer 回归 loss 下降测试和真实 HUNL 一轮采样/拟合测试。
这些不等同于神经版 Kuhn 或 200BB Hold'em 的收敛实验；随机 MCCFR 的
exploitability 也不保证逐轮单调下降。5 行动抽象、序列截断和神经近似
仍然限制最终策略质量，需长训后独立评估。
