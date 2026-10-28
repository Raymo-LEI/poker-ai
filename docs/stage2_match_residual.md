# Stage 2：50 手上下文残差 PPO 与 Stage 1 蒸馏交接

本次交付是代码、CPU 回归和合成快照端到端 smoke。没有访问训练服务器、加载最终 Stage 1 权重、启动 GPU 训练或改动现有 Stage 1 目录。代码基于包含 PR #50 的主线；旧 PPO/旧 `exploiter` API 保留。

## 入口与输入契约

- `alphanl.stage2.env.BotzoneMatchEnv`：50 手、每手 20,000 筹码、盲注 50/100、英雄交替 BB/SB，每场只采样一次对手身份。只在 50 手后置 `done`；手间回报连续进入 GAE，单位为英雄净 BB / 200。缺少英雄决策的被动手也计入比分和回报。
- `observe_poker()` 直接使用 Stage 1 的 `HunlState.observation()`，共享 `HunlTokenizer`、Equity/赔率构造器、五维动作和 `CFRActionGuard`。输入采用参考 checkpoint 中的 config，不能凭模型宽度猜测特征版本。统一位置条件是强制要求；错误文件/旧双网络没有静默 uniform 回退。
- `PolicyNetwork` 保留一个共享 `PokerTransformer`；192/6/4/512 的学生参数量为 1,491,013。128 的旧尺寸也按 checkpoint 动态读取。
- `ResidualExploiterPolicy` 冻结整份学生参考，永久关闭其 dropout，在一次 backbone 前向中获取隐藏表示与参考 logits；MLP 上下文编码、融合层、策略修正和 value head 可训练。
- 修正公式是 `ref_logits + alpha * tanh(delta)`；策略头最终层权重和 bias 全零。关闭推理安全干预时，初始动作概率与蒸馏参考逐元素相等。这里的等价对象是学生参考，学生蒸馏误差需要独立验收，不能据此声称恢复了精确 GTO。
- 训练缓存冻结隐藏状态和参考 logits，PPO epoch 不重算 backbone。损失采用最小化约定：`policy_loss + value_coef * value_loss - entropy_coef * entropy + beta * KL(final || ref)`。前十手 beta 较强，再线性降至末局系数。KL 是正则约束，不保证未经训练即可识别强敌或 5–10 手内稳定获利。

## MatchCtx 的 12 维顺序

固定顺序：`hand_progress, bankroll_diff, vpip, pfr, three_bet, fold_to_preflop_raise, flop_aggression, turn_aggression, river_aggression, fold_to_postflop_bet, all_in_frequency, showdown_win`。顺序写入 Stage 2 checkpoint，加载时严格校验。

`hand_progress` 是已完成手数 / 50（当前手零基索引）；`bankroll_diff = 2 * hero_net_bb / 200`，表示零和比赛中英雄与对手累计筹码的差。它按 200BB 缩放而不截断到 [-1,1]；英雄净赢 10BB 时双方差 20BB，输入为 0.1。

其余十项是带 Beta 先验的成功次数 / 真实机会数。VPIP/PFR 按手统计；3-bet 只统计首次面对一个 open 的机会；翻前弃牌率只统计首次面对翻前加注；翻后弃牌率按面对下注的决策；各街 aggression frequency 以全部对手自愿决策为分母；全下频率包含 all-in call，但 all-in call 不算 raise；摊牌平分计 0.5。先验均值、强度及完整计数器见 `context.py`，可 JSON 往返保存。统计器不读取对手底牌或未来公共牌。

## Stage 1 完成后的接手顺序

以下命令是交接示例，本轮没有执行真实模型训练。先确认最终归档已经发布。若实际只完成到 T=320，将 `--end-iter` 改为 320；不存在的结束轮次会报错。

```bash
export PYTHONPATH=src
export OMP_NUM_THREADS=1
python scripts/distill_cfr_policy.py \
  --strategies /absolute/stage1_final/strategies \
  --out-dir runs/stage1_distillation_prepare \
  --burn-in 100 --end-iter 350 --weight-power 2 --prepare-only
```

`--burn-in 100` 明确定义为包含 100，即 `[100,350]`。若要丢弃第 100 轮本身，传 101。支持 `--weight-power 1`（t）或 2（t²），默认为 1；不会悄悄依据训练损失的权重改变平均策略。与旧参数 `--after`（排除端点）、`--through` 不能混用同一边界。

审核 teacher manifest 后用另一个全新输出目录做正式蒸馏：

```bash
python scripts/distill_cfr_policy.py \
  --strategies /absolute/stage1_final/strategies \
  --out-dir runs/stage1_distillation \
  --burn-in 100 --end-iter 350 --weight-power 2 \
  --train-hands 10000 --validation-hands 2000 \
  --steps 4000 --batch-size 256 --device cuda
```

脚本按 t 或 t² 每手选择一个 teacher，整手保持不变，收集英雄真正到达的信息集，训练概率软标签交叉熵。这样保留 own-reach 权重，不能用 Reservoir 上无条件平均每个 teacher 的节点输出替代。没有再次对同一标签乘 t。Teacher 的哈希、规则/数值/模型契约全部验证；训练和验证按整手划分、牌种子不重叠。过滤或稀疏归档的 manifest 会标为近似。

产物包括 `teacher.json`、`dataset.pt`、`report.json`、独立的 `stage1_distilled_ref.pt`。报告的 sampled-component KL 不是平均策略 exploitability；必须用真实冻结的最终归档做独立对战、位置分组、CPU 部署验收再接受参考模型。

编辑 `configs/stage2/botzone_50hand.json` 中 `reference`、`strategies` 为实际产物路径，另存正式配置。模型结构和 Equity/Guard 参数直接来自参考文件；历史池的完整契约必须与参考一致。

```bash
python scripts/train_stage2.py --config /absolute/stage2_config.json \
  --out-dir runs/stage2_candidate --device cuda --updates 10000
```

默认配置只有两次 CPU 更新（合计 40 场完整比赛），必须显式选择 CUDA 和正式预算；导入模块、运行测试或打开监控均不会启动训练。50 手回报使用 gamma=lambda=1，默认优化累计净筹码；比赛胜负另行记录，不将它伪称为胜负概率优化。

对手池现已升级为固定 35% 防守配额与 65% 短板优先规则池；每次 PPO 更新收集 20 场完整比赛。四层比例、整局参数抖动、历史权重缓存、自身历史、独立评测与 -20 mbb/hand 防线的完整定义见 [MatchPool 升级说明](stage2_matchpool.md)。PPO checkpoint 保存参考、残差、优化器、PFSP/RNG；只在完整比赛边界保存。续训使用新输出目录：

```bash
python scripts/train_stage2.py --config /absolute/stage2_config.json \
  --resume runs/stage2_candidate/train.pt --out-dir runs/stage2_continued \
  --device cuda --updates 10000
```

`metrics.jsonl` 直接兼容原监控台，专用 Stage 2 导航展示参考 KL/系数、50 手净 BB、PFSP 与对手 HUD；训练回报不会填入独立评测胜率栏目。

## 单文件导出与裁判适配边界

```bash
python scripts/export_stage2_agent.py \
  --checkpoint runs/stage2_candidate/train.pt \
  --output runs/stage2_candidate/agent.py --decisions 300
```

导出的 `.py` 嵌入基座、残差、Tokenizer/Guard、先验和统计器，只依赖 NumPy，不依赖仓库、Torch、PokerKit、phevaluator 或外部权重文件。Equity 使用同一确定性采样算法和预翻牌表；运行时用纯 Python 牌力比较器替代 phevaluator，测试验证胜负/平分与 Equity 数值一致。尾部 padding 只在前向计算中省略，返回的观测张量不变。初始化 BLAS 的开销计入加载。

裁判输入为 `requests` 数组，输出整数 `response` 和 JSON `data`。进程重启后 `data` 恢复 MatchCtx，按 hand/history 长度去重；重复请求保持幂等。跨手比分读裁判的 `total_win_chips`。可选 `--safety` 对高 SPR、低概率的 all-in 概率转移到合法 call；这是部署干预，打开后需要单独评测，默认关闭。

**重要规则差异：**仓库真实 TexasHoldem2p 裁判并非 Stage 1 标准 HUNL：各街顺序与最小加注规则不同。训练环境按本任务的强对齐要求复用 `HunlTable`；导出把裁判 BB/SB 映射到标准座位，保留实际筹码和可见事件，再按裁判规则生成、检查响应。半池/满池动作如不满足裁判最小加注，会调整到裁判最小值。这是显式部署适配，不能宣称与标准 HUNL 动作树完全等价。

裁判发给下一手的 request 不保证包含上一手终结动作或摊牌标记。部署统计器立即记录自己的已知动作；无法观测的对手终结动作不会伪造。只对可以从自身终结 call 确定的摊牌更新摊牌胜率。训练环境能观测完整公共终结轨迹，因此部署 HUD 中部分机会可能被截尾；正式上线前必须在目标协议评测其分布差异。这一限制不涉及读取隐藏底牌。

`agent.acceptance.json` 单独报告加载与单线程决策（投影、上下文、Equity、编码、前向和采样）延迟，任何测量决策超过 15ms 或加载超过 1s 都令脚本非零退出。它是本机 gate；Botzone CPU/NumPy 版本、源码大小、内存和上传限制仍需在真实平台验收，不由本机 smoke 推断。

## 本轮 CPU 验证

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python scripts/smoke_stage2.py --out-dir /absolute/new_stage2_smoke
```

该脚本仅用合成未训练 teacher，真实执行 1.49M 蒸馏（12 手训练、4 手验证、2 次更新）、两个 20 场配额批次的 PPO 更新和单文件导出。另有单元测试在真实裁判 fixture 上执行两场 50 手部署流程，校验合法响应、筹码守恒和跨手上下文。

关键测试：两种 Equity 配置、双方座位、四条街的 Stage 1/Stage 2 observation bitwise equality；0.59M/1.49M 零残差等价；训练后基座逐参数不变；KL 方向、mask 梯度及系数；PPO 反传、完整比赛 reward/done；PFSP/RNG；checkpoint；32/128/192 宽度的 NumPy 观测与 logits 一致性；监控日志解析与浏览器显示。

本机满配 synthetic smoke 的 300 次最终测量：中位 1.470ms、P99 2.398ms、最大 6.492ms；加载 92.83ms，未导入 Torch。之前未裁剪 padding 的 300 次测量出现 19.742ms 尖峰并正确失败；后续优化后重新通过。所有数值仅说明本机代码管线可运行，不是最终 Stage 1 棋力、蒸馏保真度、GPU 性能或线上部署结论。

机器环境和可机器读取的摘要见 `docs/benchmarks/stage2_cpu_smoke_20261028.json`。
