# Stage 1：Equity 输入、1.49M / 8M、二次权重与靶向外采样

2026-10-28，基于已合并 PR49 的 `b6000fb`。本 PR 准备训练代码与配置，**尚未向训练服务器上传本 PR，未启动新的正式训练，CUDA 峰值与新策略棋力未验证**。

## 已完成模型的实际状态与败因

PR49 是 T150→T200 的 1200/2400 预算对照及 checkpoint 留存，不包含 Equity 特征。服务器当时使用直接部署的 `c72eb74`，因此“没有 merge PR49”本身不使预算对照失效。保留的两档 σ200 均为旧 `hunl_exact_fraction_v1`、0.59M，配置中没有 `equity_features`。现有结果不能用于评价 Equity 特征是否有效。

原同轮次全历史线性平均策略，各对 Equity 打 2500 对换座牌：

- 1200：−8872.8 mbb/手，95% CI [−10483.8, −7261.8]；SB −4125.6、BB −13620.0。SB 首行动弃牌 55.04%，直接盲注损失折算全部 5000 手为 −137.6 mbb/手。
- 2400：−8475.1 mbb/手，95% CI [−10001.7, −6948.5]；SB −4270.0、BB −12680.2。SB 首行动弃牌 56.40%，对应 −141.0 mbb/手。

此前清理删除了这两档的 σ1…σ199、完整恢复归档和逐牌对日志；目前保留 σ200、最终 train.pt 及汇总报告。训练 checkpoint 中的 Reservoir/当前权重不能还原历史策略。**因此无法重放原 2500 对混合策略对局，也无法对这段历史实施真实的后半程 burn-in；本 PR 明确保留此缺口。**

无需重训，另对保留的两档 **单个 σ200** 各重新打 2500 对 / 5000 手，评测 seed=1567594708，Equity 每决策仍为 64 次 Monte Carlo。结果见 [结构化诊断](benchmarks/cfr_fullupgrade_20261028/loss_summary.json)：

- 1200：−7290.7，CI [−8819.2, −5762.2]；SB −2775.8、BB −11805.6。SB 首行动弃牌 71.24%，直接贡献 −178.1。按最后一次下注街归类，翻前/翻牌/转牌/河牌分别贡献 −339.5 / −3535.8 / −1764.0 / −1651.4 mbb/全部手。
- 2400：−8686.0，CI [−10276.8, −7095.2]；SB −4725.6、BB −12646.4。SB 首行动弃牌 56.32%，贡献 −140.8。四街分别贡献 −662.6 / −3993.2 / −2078.6 / −1951.6。
- “最后加注者为模型的摊牌手”净贡献分别为 −6248.0 / −7446.4。其中模型最后主动动作是 **all-in** 的子集为 399 / 434 手，净贡献 **−6520.0 / −7920.0**，相当于总净亏损约 89% / 91%。其中 BB 子集贡献 −5440.0 / −5840.0。
- “对手最后加注、模型曾跟注至少半底池、最终摊牌”这一互斥类别仅贡献 −100.8 / −163.4。更早的跟注也可能与后续全下同时发生；不能把这个类别当作所有跟注决策的反事实成本。
- 模型为最后加注者并输掉摊牌的手中，471 / 518 手在最后一次翻后主动下注时，对实际对手底牌的事后估计 equity <35%。这是弱牌下注代理指标，采用独立 128 次 runout；不能把所有主动下注失败都称为诈唬，也不等于针对对手范围的 equity。

这些账目支持优先排查 **BB 主动全下被跟注的线路**。直接弃盲损失远小于全部净亏损；弃牌的机会成本未被测量。结算街归因把整手收益归给最后下注街，不是逐动作反事实 EV。单快照与原混合策略的差值没有匹配完整策略随机数，不能作为 burn-in 效果。

## 满配配置与权重

`configs/cfr/stage1_149m_8m_equity_league_2400.json`：d=192，4 层，6 头，FFN=512，单个 regret 网络实际可训练参数 **1,491,013**。T=300 从头训练，每席每轮 512 根树，每轮共享拟合 2400 步，minibatch=1024。

- `equity_features=true`：在现有四通道布局内加入随机范围 equity、有效跟注赔率和两者差值。翻前使用带哈希的 169 类 MC 查表；翻后每可见局面 32 次确定性采样，不读取对手暗牌或未来牌。参数/张量形状保留，新的 numeric version 阻止混用旧数据。
- 沿用历史配置的 `buffer_capacity` 单位：共享模式实际容量是该值两倍，因此这里填 **4,000,000，实际为一池 8,000,000**。Algorithm R 的保留概率不变，每个训练 batch 在 CPU/GPU 两条路径均严格 512 BB + 512 SB；池中两席留存数量不强制相等。
- 学习率最大 2e-4，每个 CFR 拟合阶段前 120 步 warmup，再 cosine 至 2e-5；每轮重新开始此确定性 schedule，Adam 状态保留。checkpoint 在完整拟合边界保存，无额外 scheduler 状态需要恢复。
- `iteration_weight_power` 支持 1 / 1.5 / 2 等正指数；新配置使用 2。Huber/MSE 的合法动作损失乘 `(t_i/T)^2`，T 为当前固定轮次。相对权重与 `t_i²` 相同，归一因子固定于整次拟合，不按随机 minibatch 均值归一。旧默认仍为未归一的线性 `t_i`。训练探针使用同一权重。
- 这是 **Polynomial 样本 loss weighting**；没有实现 DCFR 的累计正/负遗憾递推折扣，也不承诺洗掉某个特定 Fold 目标一定改善棋力。

`train_cfr.py --dry-run` 输出有效容量、numeric version、实际参数量及内存账目。长度 64 的 8M 池，CPU 紧凑数据为 10,512,000,000 bytes；GPU 数组及席位索引上界为 11,600,000,000 bytes。后者不包含模型、优化器、激活、CUDA allocator 和 worker 内存，不能作为峰值显存测量。

## 15% 固定对手根树

`scripted_opponent_prob=.15` 在每个根树开始时做一次 Bernoulli 抽样，选中后在整棵树固定非遍历方的身份。池内权重为 Equity 75%、Nit 25%；总体期望 85% 自博弈、11.25% Equity、3.75% Nit。实际比例随轮次波动，日志记录每席根数、各脚本身份根数、脚本动作及其产生的 regret 样本数。

遍历方始终使用当前网络并枚举全部合法动作，包括策略概率为零的动作；脚本方只接收自己的 `HandSnapshot`。脚本方保持与评测相同的原始规则动作集，遍历方沿用训练 guard；额外记录脚本走到 guard 外动作的次数。单进程与 spawned workers 均接入，根抽样和脚本 RNG 种子来自可恢复的遍历 RNG。

混合目标会改变纯自博弈 CFR 的训练分布和目标；这里没有重要性修正将其还原成纯自博弈估计，不宣称 GTO / Nash 收敛保证。15% 根数也不等于 15% 留存样本，因为各树访问的信息集数量不同。

## 启动、监控与保留

代码部署后，先检查有效配置，再在独立目录做 8M 固定显存分配、原尺寸模型及 1024 batch 的短 smoke：

```bash
PYTHONPATH=src /root/miniconda3/bin/python scripts/train_cfr.py \
  --config configs/cfr/stage1_149m_8m_equity_league_2400.json --dry-run

PYTHONPATH=src /root/miniconda3/bin/python scripts/train_cfr.py \
  --config configs/cfr/stage1_149m_8m_equity_league_2400.json \
  --iterations 1 --traversals 16 --train-steps 12 --warmup-steps 2 --workers 4 \
  --out-dir /root/autodl-tmp/cfr_fullupgrade_cuda_smoke_s1
```

该 smoke 并未填满主机 Reservoir，不能代替满池 RAM/保存 checkpoint 测量；GPU staging 按完整 8M 容量分配，指标中的 allocator 峰值可以覆盖这次短拟合。确认资源及损失后，在另一个全新目录启动正式训练：

```bash

PYTHONPATH=src /root/miniconda3/bin/python scripts/run_cfr_fullupgrade.py \
  --config configs/cfr/stage1_149m_8m_equity_league_2400.json \
  --out-dir /root/autodl-tmp/cfr_stage1_149m_8m_equity_league_2400_s1
```

启动器拒绝复用已有输出目录及占用中的 GPU。新 run 保存实际 config、有效 dry-run 配置、源码哈希、PID、train.log、status.json；每 5 秒保存 cgroup RAM、nvidia-smi 显存/利用率和磁盘空间，以及实测高水位。超过 90 GB RAM / 24 GB VRAM 会停止自己启动的进程组并标记失败，不自动换配置重跑。训练每轮另记 CUDA allocator 峰值。监控属于周期采样，不能捕捉全部瞬时峰值。

保留 **全部 σ1…σT**。最新完整恢复点每 10 轮保存；为控制 50 GB 数据盘空间，本配置另在 T100/T200/T300 保留完整压缩归档，区别于 PR49 对照实验的每十轮历史归档策略。旧实验数据不在本启动器清理范围。空间预测超限时保留已发布 checkpoint 并失败。

训练完成自动对全历史线性策略与后半程二次策略各打 2500 对相同换座牌，输出原始逐手轨迹、各自 CI 及牌对配对差值 CI。后半程严格取 `floor(T/2)+1 … T`，每手固定一个快照。该干预与原始全历史平均策略分别报告。

手工诊断入口：

```bash
PYTHONPATH=src python scripts/diagnose_cfr_equity.py \
  --strategies /path/to/strategies --through 300 --selection full \
  --pairs 2500 --out /path/to/new-full-diagnosis
PYTHONPATH=src python scripts/diagnose_cfr_equity.py \
  --strategies /path/to/strategies --through 300 --selection burnin_quadratic \
  --pairs 2500 --out /path/to/new-burnin-diagnosis
```

历史不完整时 full / burn-in 直接报错；单快照必须明确选择 `--selection latest`。同种子原始牌对日志供配对比较，绝不重训补造历史。

## 验证边界

本地 CPU 检查覆盖解析遗憾目标、1/1.5/2 次权重与梯度、每轮学习率端点、脚本仅替换非遍历方、零概率动作枚举、根混合 RNG 恢复、真实多进程特征传递与统计、位置均衡、单进程混合策略逐权重精确续训、旧格式拒绝混载、checkpoint/蒸馏回读，以及 CPU 启动器→训练→全历史/burn-in 配对评测全流程。极小 CPU 流程使用 2 对牌，仅验证入口，不证明正式容量稳定或棋力。

本机 Python 3.12 / PyTorch 2.14.1 CPU：全仓回归 **468 passed, 6 skipped**；最终启动 manifest 元数据调整后，满配与特征定向检查 **25 passed**。`git diff --check` 通过。CUDA 相关检查仍跳过，CPU dry-run 的 `cuda_available=false`；8M 内存账目属于 schema 计算，未做实际满池 RAM/VRAM 分配。

既有 PPO 目标和模型结构保持原行为；现有 residual collector 仅补充从冻结归档读取 tokenizer 配置，使其不会把新特征模型当成旧输入。新对手混合只发生在 MCCFR 路径。

仍待：4090 原容量 staging + 1024 batch 的实测峰值、完整 2400 步拟合稳定性、长程混合比例及真实对战结果。旧 1200/2400 历史 burn-in 已因历史文件缺失而不可验证。
