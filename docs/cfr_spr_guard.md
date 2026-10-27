# 全街 SPR All-in 门控

T=100 的翻牌圈大比例 max-bet 暴露了旧 guard 只覆盖翻前的缺口。本次采用方案 A；五维输出和 index 4 的真实 All-in 语义保持不变，不对 regret target 截零，也不额外截断 RM 概率。

## 规则

- `spr_action_guard=true`，`guard_max_pot_ratio=3.0`，`guard_unlock_raises=2`。
- 有效投入为 `min(me_stack, opp_stack + to_call)`；比值除以当前底池（包含对手本次下注）。双方投入相同时等于通常的 `min(stacks)/pot`。待跟注额避免错误地把已下注的对手判断为浅筹码。
- 比值 **严格大于 3** 且本街主动下注/加注少于两次时，从合法掩码移除 index 4。边界 3 倍允许全下。只统计本街 `bet/raise/all_in` 且 `chips_added > to_call_before`；跟注全下不计作加注。
- 翻前、翻牌、转牌、河牌相同；每条街重新计数。本街达到两次主动加注仍可在高 SPR 下全下，这是显式树规则，不能将所有高 SPR 全下都称为泄漏。
- Fold、Check/Call、Half Pot、Pot 沿用引擎合法性；对手可用完整 NL 动作。原来的 20BB/15BB 绝对门槛不再绕过 SPR 门控。

单进程遍历、并行 worker 和平均策略评测使用同一 guard，观察中的 `legal_mask` 与分支枚举一致。全负/全零遗憾回退也必须给被屏蔽动作零概率。审计按位置/街记录决策、门控次数、高 SPR 全下，以及“主动加注不足两次的高 SPR 全下”；后者是本次待消除的指标。

## 归档与恢复

旧 checkpoint/快照缺失 `spr_action_guard` 时按 false 解析，保留原先的 preflop guard 和翻后动作树。混合归档要求 guard 配置一致。恢复训练时若改动作树会拒绝；应新开 run 重新生成样本和平均策略，不把旧树 reservoir 当成新树的训练数据。

本 PR 同时带入 T=100 已使用、尚未提交到 main 的共享位置条件网络、统一 reservoir、目标缩放/Huber 与兼容性支持，才能读取真实的 `models['shared']` 归档。架构说明见 [Unified SD-CFR](cfr_unified.md)。

## 冻结 T=100 对照

```bash
python scripts/eval_cfr_readiness.py --strategies /path/to/strategies --hero-iteration 100 --out runs/spr_baseline --opponents nit,equity --stages duplicate,health --pairs 1000 --health-hands 4000 --skip-ladder --cache-size 100
python scripts/eval_cfr_readiness.py --strategies /path/to/strategies --hero-iteration 100 --out runs/spr_guarded --opponents nit,equity --stages duplicate,health --pairs 1000 --health-hands 4000 --milestones 100 --cache-size 100 --spr-action-guard
```

相同任务使用相同 seed；结果目录独立，原归档和原评测不被覆盖。英雄使用显式门控，历史对手保留归档配置，自博弈健康检查双方均应用门控。报告中记录干预参数；这是修改掩码后的历史网络混合，不能冒充原训练策略或新树重训结果。

开启 ladder 时显式包含首轮平均策略、上一归档轮次的平均策略及当前单快照。每手按 t 权重抽取一次快照并整手固定。Nit/Equity 按发牌对均值计算 95% CI；正均值且 CI 跨零不能宣称已证实转正。

## Slumbot

```bash
python scripts/eval_cfr_slumbot.py --strategies /path/to/strategies --iteration 100 --out runs/spr_slumbot --hands 100 --spr-action-guard
```

此独立入口是标准 200BB CFR，没有 50BB cap。通过引擎重放真实公开下注，验证底池、剩余筹码、行动顺序和可见牌一致。未知牌仅填充引擎，不进入策略观察；对手 off-tree 尺寸保留真实数值。审计、快照/源码哈希和原始结果逐手落盘，API token 不记录。输出目录必须全新，失败写入 report，不无限重试。100 手仅为连通与适配诊断；正式外部棋力证据仍需至少 5000 手及置信区间，不能用小样本代替。

## 2026-10-27 冻结 T=100 实测

本机最终 PR checkout 完整回归 **395 passed、6 skipped**（CUDA 不可用）。独立服务器上的 SPR、Slumbot 观察等价、共享模型及并行测试全部通过，覆盖 CUDA/BF16 与恢复。

相同种子、每对手 1000 对发牌（2000 手），完整 sigma_1..sigma_100：

- Nit：原版 **−5553**，95% CI [−7347, −3759]；门控 **−70.75**，CI [−225.54, 84.04] mbb/hand。尚未证实转正。
- Equity：原版 **−13182.5**，CI [−16354, −10011]；门控 **−11160**，CI [−13725, −8595] mbb/hand。仍显著亏损，不能把消除早期深筹码全下等同于棋力验收通过。
- 门控对首轮策略：**+4784**，CI [2075, 7493]；对上一轮平均策略 sigma_1..sigma_99：**+1359.25**，CI [−1167, 3885]；对单快照 sigma_100：**−249.5**，CI [−2679, 2180] mbb/hand。各 1000 对；后两项不能断言胜出。

各 4000 手自博弈，Flop 的真实 index 4 使用率：BB **447/1394=32.07% → 63/1483=4.25%**；SB **207/1073=19.29% → 33/1170=2.82%**。早期高 SPR 全下次数 BB **435→0**、SB **204→0**；其余全下仍遵守低 SPR/本街两次主动加注解锁规则。该种子不同于原面板，因此原始比例不必等于面板的 34.3%/21.2%。

原始完整报告与快照/源码哈希：[原策略](benchmarks/cfr_spr_t100_20261027/baseline.json)、[门控消融](benchmarks/cfr_spr_t100_20261027/guarded.json)。远端原始逐手日志保留于 `/root/autodl-tmp/cfr_spr_ablation_20261027/{baseline,guarded}/`。服务器评测源码为独立部署快照，实际哈希见报告；PR 同时保留最新 main 的输出锁、依赖版本和原始日志哈希校验。此阶段没有重训，下一步需要新树的新 run 验证 Equity 的剩余亏损。

[Slumbot 100 手诊断](benchmarks/cfr_spr_t100_20261027/slumbot.json) 已完成，均值 **+615**、95% CI **[−7368, +8598] mbb/hand**，非法选择、掩码泄漏、非有限遗憾和各街早期高 SPR 全下均为 0。小样本区间很宽，只确认外部 200BB 适配与完整采集跑通，不证明正盈利。
