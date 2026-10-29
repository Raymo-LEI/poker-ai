# Botzone v10：C200 深筹码护栏与交付说明

v10 在 Botzone NumPy 推理包装器中加入可配置部署护栏，C200 及其冻结基座的网络权重保持不变。训练器、训练对手池、远端运行目录均未修改。它是针对 50 手短赛的保守策略干预，不是对手范围求解器，也不保证不再输掉 200BB。

## 默认规则与配置

默认使用用户最后要求的规则，1BB = 100 筹码：

- 在合法动作的原始策略分布上，将概率严格小于 `0.10` 的动作置零，再归一化。恰好 `0.10` 保留。若用户把阈值调得过高导致全空，保留原分布最大概率的合法动作。
- Action 4 是主动加注、额外投入严格超过 `30BB` 且严格超过决策时底池 `2.0` 倍时，将其概率质量转到 Action 3 满池下注。所有街次生效，不设强牌豁免。Botzone 若要求更大的最小加注，使用实际合法金额检查；满池/半池都超限时退到合法跟注/过牌或弃牌。跟注熔断的优先级更高。
- 实际待跟投入 `min(to_call, me_stack) > 50BB` 且超过对手这次下注前底池 `1.2` 倍时，只有随机手牌 equity 至少 `0.82` 才可能继续。翻后还要求真实两对以上、且有私牌贡献，公牌两对/三条加一个踢脚不算。其余情况将 Call、所有 Raise 和 All-in 一起屏蔽，强制 Fold。翻前无成牌分类，使用 equity 门槛。
- `pot_before_bet` 优先取当前街最后一次对手主动下注事件中的精确底池，不能把包含对手下注的当前底池当作下注前底池。无历史才退到 `pot - to_call`，并在 debug 标明。
- 随机手牌 equity 使用可见牌、128 次固定 seed 采样，不读取对手私牌或未发公牌，不消耗策略/牌堆 RNG。`0.82` 不是“对手价值范围胜率”，真实两对也不必然满足它。
- 早先的 `>40BB` 跟注置信度锁保留为 **可选**：`big_call_enabled=true` 时，原始 Call 概率 `<0.35` 或 equity `<pot_odds+0.05` 会直接弃牌。默认关闭，避免把这一额外限制叠加到最新两条规则。置信度始终使用剪枝前概率。
- 权重缺失、格式/配置错误等异常进入 fallback 时，不允许默认跟注 `>35BB` 的下注；错误仍明确写入 stderr/debug。

护栏参数嵌在 `stage2_model.npz` 的 `meta_json.deployment_guard` 中。旧权重没有该字段时默认关闭新护栏，保持原有部署策略；新的源码 ZIP 可以加载旧 12 维和新 14 维 MatchCtx 权重。原 `--safety` 仅在未开启 v10 时生效。

## 打包、上传与调参

导出 C200：

```bash
python tools/pack_botzone_stage2.py \
  --zip delivery/botzone_source.zip \
  --checkpoint /path/to/update_0000200.pt \
  --weights-out delivery/stage2_model.npz --v10-guard
```

也可用 `--guard-config guard_config.json` 导出自定义参数。完整交付外层 ZIP 包含源码 ZIP、实际 C200 NPZ、配置、无 Torch 调参工具、说明、校验清单和原始同牌日志。**外层交付 ZIP 不能直接当作 Botzone 源码上传**：

1. 解压完整包，在 Botzone TexasHoldem2p 新建/编辑 Bot，选择 python3、JSON 交互。
2. 上传内层 `botzone_source.zip` 为源码。
3. 将 `stage2_model.npz` 以此名称上传到 Botzone 用户存储空间，运行时读取 `data/stage2_model.npz`。
4. 调试日志应出现 `ok stage2_model.npz guard=v10`；先调试再加入排名。未在用户账号上代为上传或挑战天梯。

只需 NumPy 即可修改护栏配置，网络参数不变：

```bash
python tools/configure_botzone_guard.py --zip delivery/botzone_source.zip \
  --weights delivery/stage2_model.npz --config guard_config.json \
  --out configured.npz
```

上传输出时仍命名为 `stage2_model.npz`。完整包内此工具叫 `configure_guard.py`。将配置的 `enabled` 设为 `false` 可关闭新护栏；本地也可用 `POKER_STAGE2_GUARD_CONFIG=/absolute/config.json` 临时覆盖。直接修改交付包中的 JSON 不会自动改变 Botzone 上 NPZ 内的配置，必须重写并上传 NPZ。

## 固定 C200 同牌对照：2026-10-29

C200 checkpoint SHA256：`c89f9f3016cb15742d093bd2f18a6e5a3d394f80f084d0498104dc428a6703aa`。`seed=20261030`，每个对手 1,000 对发牌、换位后 2,000 手，40 个完整 50 手赛段；95% CI 使用 20 个同牌 50 手赛段对作为独立簇，正态近似。对手使用固定参数，equity_samples=64。三组共 6,000 手/版本；未保护版与默认版合计 12,000 手。原始未保护版结果精确复现旧矩阵的三项基线。

默认护栏（最新两条 + 10% 剪枝）的结果，单位均为 mbb/手：

- **Equity**：未保护版 `-304.0` → v10 `-10.25`，v10 CI `[-1434.53, 1414.03]`；配对收益差 `+293.75`，CI `[-1413.94, 2001.44]`。单手亏损 ≥190BB：`59 → 45`；50 手赛段获胜 `20/40 → 17/40`。
- **TAG_Pro**：`-562.75 → -836.75`，v10 CI `[-1528.95, -144.55]`；配对收益差 `-274.0`，CI `[-1254.96, 706.96]`。单手亏损 ≥190BB：`18 → 13`；赛段获胜 `17/40 → 13/40`。
- **Trapper_CR**：`-1297.5 → -1153.5`，v10 CI `[-1794.10, -512.90]`；配对收益差 `+144.0`，CI `[-725.89, 1013.89]`。单手亏损 ≥190BB：`25 → 15`；赛段获胜 `15/40 → 12/40`。

三类合计单手亏损 ≥190BB 从 `102/6000` 降至 `73/6000`，这是描述性频次。默认版共有 369 次主动全下概率重映射。三固定脚本对照中超池跟注熔断触发为 0；该分支由专门牌例和独立裁判比赛验证。不能把这次结果解释为熔断本身创造了收益，也不能声称提高了 50 手胜率：三个对手的赛段获胜数都下降，收益差 CI 全跨 0。

额外叠加 40BB 锁的独立对照保存在 `plus_40bb_lock.json`：Equity `-1260.5`、TAG `-709.75`、Trapper `-873.5`；≥190BB 亏损合计降至 `28/6000`，但相对未保护版的三项配对收益差 CI 也全跨 0。它降低了观测到的损失尾部，代价可能是弃掉有利跟注，因此未选为默认。

复现默认对照：

```bash
python scripts/eval_botzone_guard.py --checkpoint /path/to/update_0000200.pt \
  --out-dir /new/output/directory --pairs 1000 --seed 20261030
```

此对照使用标准 HUNL 引擎，**不是线上 Botzone 天梯战绩**。每个候选、每个对手共用同一发牌计划，策略/对手 RNG 每赛段重置，双方拥有独立公开 HUD；日志包含逐手净收益、合法性审计和护栏触发次数。

## 牌例与协议验证

- 用用户给出的 `T♦4♥ / Q♦3♥2♠T♥6♦` 构造自洽的可见历史测试，不假装拥有原始完整回放。河牌面对 624 下注、当前底池 1224：即使强制模型偏向全下，包装器输出裁判允许的最小标准加注 1248，而非 19700 全下；将对手动作改为 19700 全下，即使强制模型偏向 Call，也输出 `-1` 弃牌。测试不读取对手底牌。
- 阈值边界、剪枝前概率、所有继续动作同时屏蔽、公牌成牌误判、配置关闭和最小加注回退均有回归测试。
- 本机全库复测 `579 passed, 8 skipped`；其后新增的配置修改无损测试单独 `1 passed`。最初受沙箱限制的共享内存/localhost 测试在本机授权环境复测后通过，未将环境报错伪装成通过。
- 实际 C200 导出包在 macOS 本地通过真实裁判 fixture 的两场完整 50 手比赛，每步新进程；141 次决策，0 非法动作、0 模型回退，最大进程耗时约 0.080 秒，最大 data 2590 bytes。只作为协议验收，不拿随机对手比分证明强度。
- 本次未复验 Python 3.6/NumPy 1.19 或 Linux 单核/256MB 限额；本地 `--portable` 明确关闭 Linux 资源限制。远端上传验收包被自动审批拒绝后，没有继续上传或改动训练服务器。历史平台验收不能替代此版本的目标平台验证。

相关证据见 [默认对照](benchmarks/botzone_v10_20261029/default.json)、[额外 40BB 锁](benchmarks/botzone_v10_20261029/plus_40bb_lock.json) 和 [本地裁判验收](benchmarks/botzone_v10_20261029/judge_acceptance_local.json)。完整交付压缩包另含原始 JSONL 逐手日志和 SHA256 清单。

本包只包含旧实验 C200。只读检查显示 `stage2_v3_pro_hunter` 已执行到 300，但第 300 次更新未通过 retention gate，`retained.pt` 仍为新实验 update 200；它不能直接标为“v3 最终验收冠军”混入本包。
