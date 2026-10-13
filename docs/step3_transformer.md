# Step 3：下注历史上的小型 Transformer 编码器

本步只写训练代码和测试，**不训练**。权重、bb/100 和 Slumbot 对照都不在这一步。

## 做了什么

用一条 **PyTorch** 路径换掉 AlphaNLHoldem 的卷积编码器。保留的是算法形态，不是 TensorFlow 1.15 那份 IMPALA 图：

| 保留 | 换掉 |
|------|------|
| 5 个抽象动作上的概率策略（softmax） | `NlHoldemNet` 的牌面 / 历史卷积 |
| 独立的 value head | Ray RLlib 的 IMPALA / V-trace 训练循环 |
| 熵系数（`ppo.entropy_coef`，命令行 `--entropy-coef`） | 把整份上游 net 塞进 TF 1.15 |
| 非法动作在 softmax 之前被掩掉 | — |
| 自博弈 + 对手池 + checkpoint | — |

策略头仍是 5 维，和 rlcard 1.1.0 / `NlHoldemEnvWrapper` 一致：`fold`、`check_call`、`raise_half_pot`、`raise_pot`、`all_in`。历史 token 比这 5 个动作更细（check/call、bet/raise 分开，下注按底池比例分桶），这样以后接 pokerkit 的连续下注额时，编码器不用改词表的形状，只改谁来填 `ActionEvent`。

## 为什么用 PyTorch，而不是在 TF 1.15 里换编码器

上游训练脚本是 Ray 0.8.3 + `ImpalaTrainer` + `VTraceTFPolicy`（`third_party/AlphaNLHoldem/train_league.py`）。README 和 AlphaHoldem 论文写的是 PPO。本步按 README：PPO、概率策略头、熵正则，并且只换编码器。

TF 1.15 没有可维护的 Transformer 实现，也装不到当前的 Python / 笔记本驱动上。把新编码器焊回那张图，等于继续把 3070 和 Colab 绑在 2019 年的 CUDA 上。PyTorch 的 `TransformerEncoder` 在 RTX 3070 Laptop 和 Colab L4/T4 上是同一份代码。

代价：

- **不能**把 `c_1048.pkl` 加载进新网络。结构不同。
- Step 1 的评测环境（Python 3.7 + `requirements-alphanl.txt`）和 Step 3（Python 3.10+ + `requirements-transformer.txt`）分开。不要把两份依赖装进同一个环境。
- 新训练循环不经过 Ray league。对手池是进程内的 PFSP，见下文。

## 数据从哪来

决策时刻的输入是 `HandSnapshot`（`src/alphanl/transformer/schema.py`），和后端无关。

两条入口：

1. **现有 env。** `NlEnvTracker` 读 `NlHoldemEnvWrapper`（或长得一样的对象）的 `last_obs` 和 `history`。`history` 里只有 `[座位, 动作 id, 合法动作]`，没有筹码。`replay_events` 按 rlcard 1.1.0 的单挑下注规则重放（含 `int(pot/2)`、盲注 1/2、`in_chips` 累加），得到每次动作的底池比例。`tests/test_step3_rollout.py` 用真实 rlcard 对局核对了「重放的筹码增量 = 环境里 `all_chips` 的差」。
2. **以后的 pokerkit。** 不要重放 rlcard。调用 `make_action_event(...)`，传入真实的 `pot_before`、`chips_added`、`action_type`（`action_id` 可以是 `None`），拼成同样的 `HandSnapshot`。`SelfPlayEnv(..., backend="pokerkit")` 现在会明确报 `NotImplementedError`。

牌面字符串两种都认：rlcard 的 `SA`，pokerkit 的 `As`。

位置按**谁下了盲注**记 SB/BB，不按「按钮」。rlcard 翻牌后行动顺序和标准单挑不一致（上游 README 已写明）。模型学到的是这个环境的顺序。换 pokerkit 之后需要重训，不能把权重直接拿过去。

## Tokenizer

序列（CLS 固定在下标 0，策略和 value 都读它）：

```
[CLS] [HOLE] [洞牌] [洞牌]
[STREET_preflop] [动作] ...
[STREET_flop] [翻牌 3 张] [动作] ...
[STREET_turn] [转牌] [动作] ...
[STREET_river] [河牌] [动作] ...
```

每个动作 token 打包四件事：语义类型（fold / check / call / bet / raise / all-in）、SB 或 BB、是不是「我」、底池比例桶。另外 4 维连续特征保留未分桶的比例和筹码，避免分桶把 pokerkit 的细尺寸扔掉。

底池比例 = 本次投入筹码 / 行动前底池。桶：

| 桶 | 比例 |
|----|------|
| 0 | ≤ 0（check、fold） |
| 1 | (0, 0.25] |
| 2 | (0.25, 0.5] |
| 3 | (0.5, 0.75] |
| 4 | (0.75, 1] |
| 5 | (1, 1.5] |
| 6 | (1.5, 2] |
| 7 | (2, 4] |
| 8 | > 4 |

词表大约 275。右手边是 PAD。超长时先丢掉最早的动作，CLS、洞牌和该街的街标记尽量留下。默认最长 64（笔记本）或 96（Colab）。50bb、5 个抽象动作的单挑很少超过这个长度。

非法动作不进词表。它们只出现在 `legal_mask` 里：对应 logit 被写成一个很大的负数，softmax 之后概率为 0。采样不会走出这个掩码。

## 网络和 PPO

`PokerTransformer`：token / 类型 / 位置嵌入，加上数值特征的线性投影，然后是 Pre-LN 的 Transformer encoder。CLS 经过一层 GELU 全连接，再分成：

- `policy_head` → 5 维 logits（已掩码）
- `value_head` → 一个标量

策略头用 gain 0.01 的正交初始化，开局接近在合法动作上均匀分布，熵正则才有东西可拉。这和上游 `normc_initializer(0.01)` 的用意一样。

损失（`src/alphanl/transformer/ppo.py`）：

```
ratio = exp(new_logp - old_logp)
policy = -mean(min(ratio * A, clip(ratio, 1-ε, 1+ε) * A))
value  = 0.5 * mean(max((V-R)^2, (V_clip-R)^2))
loss   = policy + value_coef * value - entropy_coef * entropy
```

熵是掩码之后的分布算的，非法动作不贡献 `0 * (-inf)`。`entropy_coef` 越大，越把合法动作往均匀推。这是防止策略塌成永远弃牌的那个旋钮。

其他默认值和上游对齐的部分：`gamma = 1`（一手牌才结算）、学习率 `3e-4`。奖励是**大盲为单位**的筹码盈亏（rlcard 大盲 = 2 筹码，起始 100 筹码 = 50bb），不是原始筹码。Advantage 在每次更新前按整个 rollout 标准化。

没有做 AlphaHoldem 的 Trinal-Clip。见文末未决问题。

## 自博弈和对手池

`SelfPlayEnv` 直接调用 rlcard 1.1.0，不经过 TensorFlow。每个环境随机给英雄换边。英雄每步一个决策；对手在自己的座位上连续行动。对手每手从三处抽一个：

- `random_prob`：合法动作均匀随机
- `selfplay_prob`：当前网络（随机采样，不是贪心）
- 其余：池子里的冻结 checkpoint，按 PFSP，权重 `(1 - 胜率)^2`。胜率是赢 1 / 平 0.5 / 输 0 的平均，不是 bb。胜率封顶 0.95，避免把一个一直在赢的对手完全抽不到。

池子只保留最近 `max_size` 个 `state_dict`（CPU 副本）。多个环境共用一个 shadow 网络，每次对手行动前再加载该手的权重。模型大约 0.6M–5M 参数，这次拷贝相对 8GB 可以忽略。

英雄没轮到说话就结束的牌（例如英雄在大盲、对手开局弃牌）记入对手池和被动奖励，但不进 PPO，因为没有英雄动作。

## 显存和两份配置

| | `configs/transformer/laptop_8gb.json` | `configs/transformer/colab_l4.json` |
|--|--|--|
| 参数量 | 592,390 | 4,904,454 |
| d_model / 层 / 头 / FFN | 128 / 4 / 4 / 256 | 256 / 6 / 8 / 1024 |
| 序列长度 | 64 | 96 |
| 并行环境 × rollout | 8 × 128 = 1024 个英雄决策 | 32 × 256 = 8192 |
| PPO minibatch / epochs | 256 / 4 | 1024 / 4 |
| 熵系数 | 0.01 | 0.01 |
| 对手池 | 8 | 16 |
| rollout 设备 | CPU | CPU |
| 更新设备 | `auto`（有 CUDA 就用 GPU） | 同左 |
| AMP | 关 | 开 |
| 对随机对手的贪心评估 | 每 25 次更新，20 手 | 每 20 次更新，50 手 |

Rollout 放在 CPU 上：这个网络很小，batch 只有「环境个数」那么大，GPU kernel 启动比 CPU 前向还贵。PPO 的 minibatch 才上 GPU。rlcard 是纯 Python，吞吐会先卡在环境，而不是 3070 的 8GB。

显存是估算，**没有在 3070 / L4 / T4 上量过**：

- 笔记本：256 × 4 头 × 64² × 4 字节的注意力图大约 16MB/层，4 层加上反向，激活远小于 1GB。参数和 Adam 状态大约 10MB。
- Colab：1024 × 8 头 × 96² 大约 300MB/层，6 层加上反向大约 2GB 量级。T4 16GB 和 L4 24GB 都放得下；AMP 再把激活砍一截。不要把 Colab 配置拿到 8GB 笔记本上跑。

## 怎么跑

Step 3 用单独的环境（Python 3.10 或更新）。不要和 Step 1 的 TF 1.15 混装。

```bash
pip install -r requirements-transformer.txt
# GPU 机器请换成和驱动匹配的 CUDA 版 torch，不要留 CPU wheel。

export PYTHONPATH=src
python3 scripts/train_transformer.py --dry-run
python3 scripts/train_transformer.py --config configs/transformer/laptop_8gb.json
python3 scripts/train_transformer.py --config configs/transformer/colab_l4.json --entropy-coef 0.02

# 从 checkpoint 接着训
python3 scripts/train_transformer.py --config configs/transformer/laptop_8gb.json --resume
```

输出目录（默认 `runs/transformer_laptop/` 或 `runs/transformer_colab/`）：

| 文件 | 内容 |
|------|------|
| `metrics.csv` | 每次更新一行：policy/value/entropy、KL、clip 比例、平均 bb、池大小、对随机对手的贪心 bb |
| `tb/` | 同一批标量的 TensorBoard。没装 `tensorboard` 时只写 CSV |
| `train.checkpoint_dir/update_XXXXXXX.pt` | 模型、优化器、GradScaler、对手池、更新步数、配置、全部 RNG 状态和牌桌状态；只保留最近 `keep_last`（默认 3）个，跑完时最后一步也会存。详见 `docs/training.md` |

`--updates` 可以先改成很小的数做冒烟。`pytest` 里有一次「1 个环境、2 步 rollout、1 次 PPO 更新」的接线测试，会写出 CSV 和 checkpoint。那不是训练结果。

测试：

```bash
export PYTHONPATH=src
python3 -m pytest -q
```

覆盖：token 形状、街标记、截断、`As`/`SA`、rlcard 筹码重放、非法动作掩码和采样、padding 不进 CLS、GAE、熵系数只改变熵那一项、随机数据上几次 PPO 更新、权重会动、保存/加载后 logits 一致、PFSP 更常抽到输的对手、两份 JSON 配置、pokerkit 后端被拒绝、以及装了 rlcard 时的一整手牌。CPU 上大约几秒。

## 和 Step 1 的边界

- `src/alphanl/__init__.py` **不**导入 `transformer`。Step 1 评测不需要 PyTorch。
- 没有改 `third_party/AlphaNLHoldem` 里的网络和训练脚本。
- `NlEnvTracker` 可以在 TF 环境里读现有 wrapper，但新训练器自己调用 rlcard，这样 PyTorch 环境不必安装 Ray / TF 1.15。

## 未做 / 开放问题

1. **没有训练。** 没有 checkpoint 可以打随机机器人或 Slumbot。熵系数 0.01 相对上游 IMPALA 的 `entropy_coeff = 0.1` 更小；0.1 在 5 个动作上很容易把策略拉平。哪一个不会塌成 always-fold，要扫，本步没有扫。
2. **标准 PPO clip，不是 Trinal-Clip。** AlphaHoldem 论文还有额外的 clip 边界。要不要加，是下一步的算法选择，不是编码器的事。
3. **策略头锁在 5 个动作上。** pokerkit 可以先把 fold / check-call / 0.5 pot / 1 pot / all-in 映射回这 5 个动作，历史里用真实比例。若要更细的下注，policy head 和 `legal_mask` 的宽度要一起改。
4. **rlcard 的底池和翻牌后行动顺序是错的。** 和 Step 1 一样。换 pokerkit 之后这套权重不保证还能用。
5. **对手池是 PFSP，不是上游的 K-best Ray league，也没有单独的 exploiter。**
6. **没有从 `c_1048.pkl` 做行为克隆。** 热启动要另做数据管线。
7. **价值头预测的是 bb 回报（大约 ±50），没有 PopArt / 回报归一化。** 开局 value loss 会比 policy loss 大，梯度有 `max_grad_norm = 0.5` 截断。
8. **GPU 路径没在 3070 或 Colab 上跑过。** `device=auto` 在有 CUDA 时把 PPO 更新放到 GPU；单测和本环境的冒烟都是 CPU。
9. **评估只对随机合法对手，贪心出牌，手数很少，只用于看有没有塌。** 不是 Step 1 那种带置信区间的 mbb/hand，也没有接 RuleBot（那个机器人在 TF 环境里）。
