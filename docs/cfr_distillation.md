# SD-CFR 平均策略蒸馏准备

For the exact-mixture, opponent-augmented fidelity pipeline with playing-strength selection and conditional acceptance, see [Distillation fidelity](distillation_fidelity.md). The legacy workflow below remains unchanged.

本工具把已冻结的统一位置条件 SD-CFR 策略归档压缩为单个
`PolicyNetwork`，产物是离线候选模型。它不恢复或修改 CFR 训练，不自动接入
Stage 2，也不自动打包、上传 Botzone。长训耗时、棋力与部署延迟尚待实测。

## 为什么不能直接平均 25 个网络的输出

快照 `iteration_t.pt` 表示本轮采样使用的 `sigma_t = theta_(t-1)`。
最新训练网络、神经网络参数平均、逐状态概率算术平均，都不能直接替代
SD-CFR 的平均策略。

对保留的历史集合 S，线性权重的平均行为策略是：

```text
bar_sigma_i(I,a) = sum[t in S] t * reach_i(sigma_t,I) * sigma_t(I,a)
                   --------------------------------------------------
                   sum[t in S] t * reach_i(sigma_t,I)
```

`reach_i` 只含自己的动作到达概率。不能重复乘对手和机会到达概率。
分母为零的信息集没有被教师到达，不能凭此数据声称已学会那里。

实现采用轨迹采样，而非显式枚举到达概率：每手按 t 权重选择一个快照，
整手保持不变；hero 按该快照行动，只记录 hero 实际到达的决策及其软概率。
对手在受同一 guard 限制的合法集合内均匀随机，独立于 hero 抽到的快照。
因此同一个完整信息集的样本里，快照出现频率已正比于 `t * reach_i`。
软标签交叉熵的总体最优解就是上式；**训练时不再额外乘 t**。
每个 epoch 不能改用学生行动，不能每步重新抽快照，也不能在任意 replay
状态上直接查询所有快照再按 t 算术平均。

这里只在完整信息集意义上有上述等价关系。有限轨迹、截断历史、共享神经
网络容量及优化误差仍会影响结果；不能据此宣称候选模型达到纳什均衡。
依据见 [SD-CFR 第 5 节](https://arxiv.org/abs/1901.07621) 和
[Deep CFR](https://proceedings.mlr.press/v97/brown19b.html)。

## 冻结教师与快照选择

默认保留 index 中的全部历史。准备阶段逐个检查快照的版本、轮次、权重、
位置编码、模型结构、regret fallback、数值单位、guard 语义及权重有限性，
记录 SHA-256。每个数据 split 采样前后再次校验，归档被替换则失败。
按已冻结的明确轮次列表采样，不会随训练新增 index 自动扩大教师。
输出目录必须不存在，避免覆盖实验。建议从只读归档副本运行。

当前单学生路径仅支持 `separate_player_models=false` 且
`position_conditioning=true` 的统一网络。旧双网络归档明确拒绝，不能静默
丢掉 BB/SB 差别。全街 SPR guard、旧 preflop guard 和位置编码均从教师
配置恢复，不强行套用最新默认值。

```bash
export PYTHONPATH=src

# 第 350 轮文件尚不存在时会明确失败；不要把较短前缀误认为 350 轮模型。
# 只做归档检查和 manifest，不采样、不训练。
python scripts/distill_cfr.py \
  --strategies runs/sd_cfr_hunl/strategies --through 350 \
  --out-dir runs/distill_full_prepare --prepare-only

# 全历史教师候选；显式选择设备，不推断已有 4090。
python scripts/distill_cfr.py \
  --strategies runs/sd_cfr_hunl/strategies --through 350 \
  --out-dir runs/distill_full_candidate --device cuda \
  --train-hands 10000 --validation-hands 2000 \
  --steps 4000 --batch-size 256 --seed 17

# 对照候选：丢弃 <=100 的轮次，取 110,120,...,350，恰好 25 个。
# 100,110,...,350 若两端都包含则是 26 个。稀疏/尾部教师会标记 approximation。
python scripts/distill_cfr.py \
  --strategies runs/sd_cfr_hunl/strategies --through 350 \
  --after 100 --stride 10 --out-dir runs/distill_tail_candidate --device cuda \
  --train-hands 10000 --validation-hands 2000 --steps 4000 --seed 17
```

这些是起步实验参数，不是收敛或时间保证。先用小规模 CPU 冒烟，再测真实
4090 的教师采样、训练耗时与峰值内存。此实现优先验证语义，教师逐决策
推理；CUDA 并不保证比 CPU 更快。数据在 RAM 中，`--max-decisions` 默认
每个 split 200,000 条，超限失败，不静默截断手牌。应按本机内存调整。
默认长度 64 的主张量约 1.9KB/决策；同时有逐行 Tensor 开销、拼接临时
副本及双方 split，实际峰值明显更高。这里尚未实现流式分片采样。

## 产物和验证指标

- `teacher.json`：教师轮次、选择条件、每份快照哈希、规则/特征/guard 语义。
- `dataset.pt`：train/validation 两组完整手牌划分的数据，含 hand_id、位置、
  street、来源轮次、合法掩码和软标签。手牌 ID 范围不交叉，牌堆种子采用
  `(seed << 64) + hand_id`，避免压缩成 32 位种子时的碰撞。
  源教师的策略随机流也按 split 区分；固定 seed 可复现。
- `policy.pt`：独立 `sd_cfr_distilled_policy_v1` 格式；单 backbone 和策略头，
  不含 regret head、value head、optimizer 或 reservoir。默认约 0.59M
  参数，具体数目写进报告。加载无需访问教师归档。
- `report.json`：参数、手牌数、快照抽样分布、位置/街覆盖、训练前后留出集
  交叉熵、component KL、非法概率质量与本机 CPU warm forward 延迟。

`sampled_component_kl_nats` 是对**被抽中快照的软标签**计算的平均 KL。
即使学生恰好等于平均策略，这个值也可能非零；它不是精确条件平均策略
的 KL，更不是 exploitability。交叉熵用于同一个冻结教师、同一留出集的
前后对比。两个教师各自的留出分布和目标不同，不可仅比较它们的报告数字
就认定丢弃前 100 轮更好。

保存后重新加载学生，再测单线程 CPU、batch=1 的 warm forward+softmax
p50/p99。计时不含冷启动、模型读取、tokenization 和 Botzone 协议处理。
这不能证明 Botzone 的端到端 5ms；目标容器需独立测量。

```python
import torch
from alphanl.cfr.actions import CFRActionGuard
from alphanl.cfr.distillation import load_student
from alphanl.transformer.hunl import HunlTokenizer

policy, metadata = load_student("runs/distill_full_candidate/policy.pt")
guard = CFRActionGuard.from_config(policy.config)
tokenizer = HunlTokenizer(policy.config.model.max_seq_len, position_conditioning=True)
batch = tokenizer.encode(guard.apply(snapshot))  # 自己的洞牌与公开信息
with torch.inference_mode():
    probabilities = policy.strategy(batch)     # 每次决策只运行一个 encoder
    action = torch.multinomial(probabilities, 1).item()
```

此加载接口是后续 Stage 2/部署接入的边界，不应把独立格式伪装成现有 PPO
checkpoint。Stage 2 的 `FrozenBase` 当前仍使用原快照教师。旧 Botzone simple
裁判和标准 HUNL 的翻后行动顺序有差别，现有 NumPy PPO 打包入口也不自动
支持这个格式，需单独实现并验证适配。

## 正式替换教师前的验收

1. 冻结完整历史、尾部稀疏教师及数据种子。保留完整历史快照，不清理前
   100 轮文件；尾部截断只作为实验条件。
2. 先做 CPU 小样本端到端冒烟和目标 GPU 吞吐实测，再扩采样量。检查
   BB/SB、每街覆盖；缺失或稀疏街需要增加数据。小样本或 loss 下降不算棋力。
3. 在**同一完整历史教师的独立留出数据**上评估所有学生。可用
   `evaluate_student(student, full_validation)`；记录相同切片的交叉熵与
   非法质量。与未训练学生相比应改善；未出现的切片不能按 0 误差处理。
4. 用独立牌堆做学生 vs 完整教师、规则对手和其他基线的座位互换配对评测，
   报告 mbb/hand 及以牌堆对为单位的 95% CI，事先约定可接受的棋力损失。
   必须加入强对手/偏离轨迹、off-tree 下注测试；当前均匀抽象对手覆盖
   不保证学生访问的新状态安全。真实德扑 exploitability 仍未被计算。
5. 为 Stage 2 保留随时可退回的快照基座；验证冻结特征、prior 与 PPO 比率。
   部署侧验证动作翻译、位置/数值编码、guard、规则、无隐藏信息泄漏、
   单核完整协议 p50/p99 和冷启动资源。达到约定标准后才能替换/提交。

准备 PR 的自动测试覆盖完整/稀疏归档选择、损坏/混合快照拒绝、归档替换
检测、到达概率反例、真实 HUNL 轨迹和固定手内快照、位置/guard、软标签
掩码与梯度、拟合、便携导出/重载、单 encoder 调用和 CLI 端到端冒烟。
测试使用小型随机快照，不代表真实 350 轮模型的蒸馏或棋力结果。
