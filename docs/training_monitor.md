# Overwatch 训练监控台

整套源码与双击启动入口均已纳入仓库；通用部署及独立 ZIP 打包步骤见 [monitor/README.md](../monitor/README.md)。下文含历史本机安装记录，实际部署以通用入口为准。

## Overwatch 2.0 工作台

现已在原有 A～E 训练面板上加入分页面的运行总览、实验库与对比、评测变体、对局与漏损、模型与归档、运行与证据时间线。正式入口仍是 <http://127.0.0.1:6009/#overview>。

- 总览区分训练日志轮次、独立评测轮次、阶段文件时间和资源采样时间。阶段与资源记录超过 90 秒标记过期；进程存活没有独立证据时保持未知。ETA 只使用最近完整 `iteration_s` 的四分位耗时区间，不含额外评测。
- 有效配置优先使用 `launch_manifest.json/effective_config`，展示实际容量、可训练参数、权重和对手根树配置；不会加载模型核验参数。设备标签分别标明训练设备与评测运行环境。
- 实验库可搜索、按类型筛选、选择 2～4 个实验。对照提供配置差异、诊断曲线、相同记录时间预算下的完整日志点、逐对手收益和 Markdown 报告导出。累计时间不是 GPU 账单时间，缺失计时后不继续推算。
- `workbench.evaluations` 保留报告输入身份，full/burn-in/单快照、seed、预算与 guard 干预可独立选择；根目录发布副本去重。归档内的首轮/上一快照对手与固定规则对手分开，避免把各实验自己的历史网络视为同一个对手。
- 归档页只检查文件、引用代次目录、压缩 manifest 和 `objects/` 的存在性及大小，不读取大权重、不解压、不宣称实际恢复测试通过。压缩档案适配 `checkpoint_archives/`。
- 回放只接收服务器根据已知报告发现的 trace ID，不接收任意文件路径。按需索引 `hands.jsonl` 前 64 MiB / 25,000 手，单记录最多 1 MiB；分页最多 100 手，内存保留最近四份索引。缺少动作轨迹时明确提示，不启动补采。底牌、合法行动和未记录的数值不补造。
- 亏损桶同时给出桶内均值和按全部已索引手数计算的贡献；它们是描述性账目，不是反事实动作 EV。

新增只读 API：`/api/catalog`、`/api/health`、`/api/hands`、`/api/hand`；`/api/run` 增加 `workbench` 字段。新适配器位于 `scripts/monitor_workbench.py`，无 Torch、GPU 调用、子进程或文件写入；前端扩展为 `workbench.js` / `workbench.css`。

本机安装副本同步时还需包含上述三个新文件。SSH、旧曲线、gzip、ETag 和断线页面可用性保持现有机制。控制台不启动训练、评测或归档恢复任务。

本地 HTML 面板，直接读取 SD-CFR / benchmark / PPO 现有日志。Python 标准库即可启动；前端不访问 CDN、不安装 npm 依赖，也不加载 checkpoint。监控进程与训练进程独立，不修改训练循环，不增加 GPU–CPU 同步。

## 启动

在 `poker-ai` 根目录：

```bash
python3 scripts/monitor_training.py
```

打开 <http://127.0.0.1:6008>。默认监控本仓库 `runs/` 的直接子目录，每 5 秒刷新，可切换实验、暂停刷新、选择曲线范围、悬停查看数据、展开配置和导出当前实验 JSON。使用浏览器实际时区显示日志时间。

也可以双击仓库根目录的 `打开训练监控.command`。终端关闭或按 Ctrl+C 后监控服务停止；训练不受影响。

指定实际训练输出目录，支持一个实验目录或多个实验的父目录；可重复指定：

```bash
python3 scripts/monitor_training.py --runs-dir /path/to/runs --port 6008
python3 scripts/monitor_training.py --runs-dir runs --runs-dir /path/to/another-run
```

服务器训练时，在服务器仓库运行同一命令，然后在自己电脑另开终端转发：

```bash
ssh -N -L 6008:127.0.0.1:6008 用户名@服务器地址
```

仍在自己电脑访问 <http://127.0.0.1:6008>。服务器和本地必须各自有相应进程运行。默认只监听服务器回环地址，无需额外开放公网端口。此 HTML 服务使用 6008；已有 TensorBoard 可继续使用 6006。

### 本机已接通的 4090 服务器

2026-10-27 已使用现有 SSH 别名 `seetacloud-poker` 接入服务器，地址分开保留：

- <http://127.0.0.1:6008>：本机 `runs/`，包含本机评测结果。
- <http://127.0.0.1:6009>：本机常驻网页入口，经独立 SSH 隧道读取 4090 服务器日志，页头显示“4090 服务器已连接”。

服务器监控程序独立放在 `/root/.cache/poker-overwatch/20261027-dashboard`，只监听远端 `127.0.0.1:6008`，读取 `/root/autodl-tmp` 和 `/root/poker-ai/runs` 的直接子目录。没有修改训练代码、训练配置或重启训练进程。会读取实际 run 的 `ops/run_config.json`；CLI 覆盖的训练目标以训练进程参数为准，配置文件里的初始 iterations 可能仍为 50。

本机入口由现有 LaunchAgent `com.runyuanlei.overwatch-tunnel` 启动 `scripts/monitor_remote.py`：Python 常驻监听 `127.0.0.1:6009` 并提供网页；SSH 只监听内部端口 `127.0.0.1:6012`，转发到服务器 `127.0.0.1:6008`。SSH 断开时网页仍返回 200，API 返回明确的 503 重连提示，前端保留已有数据并自动重试。隧道按 1–15 秒退避，稳定运行一分钟后重置退避；仅使用已有 `~/.ssh/config` 的身份配置，不包含密钥内容。

已安装副本位于 `~/Library/Application Support/Overwatch/`，包含三份监控 Python 源码与五个前端文件，避免依赖项目目录一直可用。修改仓库监控代码后需要同步此副本；启动项配置和日志位于 `~/Library/LaunchAgents/com.runyuanlei.overwatch-tunnel.plist` 与 `~/Library/Logs/overwatch-tunnel.launchd.*.log`。原配置备份在 `~/Library/Logs/overwatch-gateway-before-*.plist`，2.0 代码备份在各监控安装目录的 `v2_*before_*`。本机网关访问回环端口时明确绕过系统代理。

仓库中的 `连接服务器监控.command` 直接运行当前 checkout 的网关；默认使用已有 SSH 别名 `seetacloud-poker`，不依赖个人 LaunchAgent。已有网关占用端口时会报错，需先退出旧监控或另选端口。服务器重启后需先重新启动远端日志服务：

```bash
ssh seetacloud-poker
nohup /root/miniconda3/bin/python \
  /root/.cache/poker-overwatch/20261027-dashboard/scripts/monitor_training.py \
  --runs-dir /root/autodl-tmp --runs-dir /root/poker-ai/runs \
  --source-label '4090 服务器' --port 6008 \
  > /root/.cache/poker-overwatch/20261027-dashboard/monitor.log 2>&1 < /dev/null &
```

先确认原监控服务未运行，再执行启动命令；端口已占用时不会替换已有进程。页面“已连接”表示日志服务连通，训练是否存活仍须检查训练进程。

## 当前可见的数据

- `metrics.csv`：CFR 玩家 Loss、reservoir、采样计数；PPO policy/value loss、KL、entropy、步数与已有评测。
- `performance.csv`：双方采样/拟合耗时与吞吐；采样总吞吐为双方样本总数除以双方采样耗时总和。
- `benchmark.json`：基准的迭代记录、设备和配置；CSV / JSONL 同轮数据覆盖报告中的同名字段。
- `checkpoint_performance.jsonl`：保存耗时；`train.pt` 仅取文件大小和修改时间，绝不反序列化。
- `strategies/index.json`：展开配置可检查策略归档清单。
- `report.json`：若格式为 `sd_cfr_readiness_v1`，读取目标轮次对战胜率、已有置信区间、Random 对手位置胜率，以及自博弈 health 行动统计。可直接将评测输出目录作为 `--runs-dir`。标准 HUNL 50 手比赛单独标注，不冒充 Botzone simple 比赛。
- `evaluations/readiness_periodic/full_budget_t*/report.json`：读取同一实验的历史正式评测，按报告里的 `hero_iteration` 合并；根目录报告是最新发布副本，同轮不会重复计数。只接受数字轮次的固定目录，不混入 guard 干预评测，不递归扫描其他文件。最多保留 200 份报告、累计读取 32 MiB；历史文件变化、新增或删除都会使缓存失效。
- `metrics.jsonl` / `eval.jsonl`：下述统一指标接口。

每个日志最多读末尾 8 MiB、最近 5,000 条记录；文件未变时复用解析缓存。训练写到一半的最后一行暂不展示，后续完整写入后自动出现。损坏行跳过并提示，`NaN` / `Inf` 显示缺失。重写或截断 CSV 后刷新可见更新后的记录。只扫描指定目录及其直接子目录，不递归扫描 checkpoint 数据。

B 面板的胜率曲线显示全部已读取评测，不随 A 面板“最近 100 轮”裁剪。“评测轮次”可切换 B 的胜率快照和 C 的动作统计，默认跟随最新评测。选择历史轮次后自动刷新会保留选择；该轮没有采集的字段显示缺失，不借用最新一轮的数据。训练总览仍展示最新训练和评测摘要。

### SSH 链路较慢时

服务对浏览器支持 gzip 压缩与实验数据 ETag：数据未变化时返回 304，复用页面内存中的数据，不重复传输全部历史。每次刷新完成后等待 5 秒再发起下一次；失败时按 10、20、40、60 秒退避。切换实验会取消旧请求，后台标签暂停轮询，回到前台立即刷新。单个请求的 45 秒时限包括响应体传输，超时后仍保留上次成功的数据；页面显示最后成功读取时间，连接成功不代表训练仍在运行。

如果列表和实验数据都很慢，而服务器内部接口正常，应检查 SSH 链路。本次修复前实测新实验未压缩响应约 970 KB，经隧道需 28–35 秒，超过旧前端 12 秒限制。服务改为紧凑 JSON、压缩和条件请求，避免固定高频重传进一步占用链路。

**“日志服务已连接”只表示页面连上监控服务。** 页面显示日志最后写入时间，不根据修改时间猜测训练是否存活。Loss 下降不等于棋力提高；现有冒烟、性能基准不能当作正式策略成绩。

## 胜率与 HUD 指标接口

### 归一化拟合诊断与样本不对称

标准 HUNL 的 seat 0 固定为 BB，seat 1 固定为 SB/button，所以此路径的 P0=BB、P1=SB。之前面板声称 P0/P1 不对应固定位置的说明已更正。外采样遍历枚举 traverser 自己的全部合法行动，只抽取对手一个行动；对手早弃牌可截断当前 traverser 后续的信息集。双方 root 数相等不意味着 regret 信息集数相等，buffer 计数也不是实际对局数。算法定义见 [MCCFR 原论文](https://www.cs.cmu.edu/~kwaugh/publications/nips09b.pdf)。

原 Loss 为 `mean(t_i * sum_legal(error_i²))`，`t_i` 是 reservoir 中每条样本产生时的轮次。并不是将无权重 Loss 统一乘当前 T；目标尺度、样本组成与拟合情况都会影响曲线，也不保证单调增长。

新训练代码对每人每轮使用最多 256 条独立 RNG 抽取的保留样本，在拟合前后测同一个批次，记录：

- `p{0,1}_probe_loss_ratio`：同批次的 end/start；低于 1 表示本轮拟合降低了此批次误差。
- `p{0,1}_probe_r2`：`1 - SSE/SST`，以按迭代权重计算的每个行动的目标均值作为常数基线。非法行动排除；零方差目标的 R² 显示缺失，不伪造 0。
- `p{0,1}_probe_normalized_mse`：`sum(t_i * sum_legal(error_i²)) / sum(t_i)`，消除整体权重尺度。它仍受目标方差影响，不保证跨轮可比。
- `p{0,1}_probe_unweighted_mse`：同批次无迭代权重的误差。

探针不消耗训练 sampler 的 RNG，不增加优化器更新，也不在 minibatch 中取新增日志标量。原训练 Loss 的累计改为设备端累计后在拟合结束读回。原有非有限数/梯度安全检查保持不变。保留训练样本可能已被训练使用，**这些指标不是独立验证集成绩，也不是棋力**。

已完成的历史训练没有保存实际 start_loss，不能事后伪造 End/Start 历史。独立脚本 `scripts/collect_monitor_diagnostics.py` 可从用户自有可信 checkpoint 计算 R² 和 `p{0,1}_probe_end_zero_ratio`（相对零预测的误差比），写入 `diagnostics.jsonl`。End/Zero 在页面独立选项中展示，与实际观测的 End/Start 区分。

```bash
python3 scripts/collect_monitor_diagnostics.py --run-dir /path/to/run \
  --iteration 100 --probe-samples 2048 --health-hands 4000 --pairs 1000
```

此脚本只做推理，先测 checkpoint，再评估 `sigma_1..sigma_T` 的线性加权平均策略，追加真实 `eval.jsonl`；不启动或恢复训练。每个对手 `pairs` 组独立牌对，每组交换身份两手，95% CI 按牌对均值计算。当前支持 random/call/maniac/overbet/nit/equity。完整结果与原始对局日志保存在 `run/monitor_diagnostics/t*/`。需要为每次运行保留独立输出目录；不要并发调用此脚本向同一个实验写日志。

本次服务器使用的隔离诊断副本位于 `/root/.cache/poker-overwatch/20261027-diagnostics/scripts/collect_monitor_diagnostics.py`，其 `src/` 随副本一起保存。T=100 完整结果为正式 run 下的 `monitor_diagnostics/t0000100_1793065135/report.json`。后续训练的两处核心修改已在核对原文件 SHA-256 后安装，原文件备份保留于 `/root/.cache/poker-overwatch/before_fit_probe_1793065841/`。未由此诊断脚本启动续训。

自博弈 HUD 按实际 seat 网络区分 P0/BB 与 P1/SB，展示各街五行动频率及决策数。全街行动比例以该玩家该街决策数为分母；“SB 开局推 200BB”只以 SB 首次行动机会为分母，“BB 面对开局加注 Fold”只以首次面对 SB 加注的机会为分母，两种口径不能混用。配对收益和自博弈 HUD 来自不同轨迹，分别标注 T 和预算。

### HUD 空白与新统一网络

`—` 或“未采集”表示日志没有对应测量，并不表示 0。原采集器只记录 SB 开局 VPIP、部分行动计数和熵，没有 PFR、BB VPIP、BB 3-Bet 或 raw regret 正值个数；这四项现已接通。

- VPIP：该座位已发底牌的手数中，至少一次主动投入翻前筹码的比例。盲注不计；BB walk 仍计入已发手数；免费 check 不计 VPIP；一手多次跟注/加注只计一次。
- SB PFR：SB 已发手数中，至少一次翻前加注的比例，包括 limp 后的再加注，一手只计一次。
- BB 3-Bet：面对 SB 首次翻前加注、且抽象合法行动允许再加注的机会中，BB 实际再加注的比例。面对对手 all-in、自己只能跟注或弃牌时不计机会；all-in call 不计为加注。分母为零显示“无机会”，不能填 0%。
- 正 Regret 行动数：每次决策中 raw regret 大于 0 的合法行动数，再对决策求均值；使用 guard 之后的合法集合。不能根据 softmax 后的正概率推断，也不能从熵推算。页面同时展示决策数。

共享网络中 P0/P1 是 BB/SB 座位，不是两个独立网络。guard 开启时，0% SB 深筹码开局 all-in 部分由动作约束决定，页面会明确提示。CFR 页面隐藏 PPO 专属 rollout entropy；免费 check 弃牌率只有存在非强制、且 fold/check 同时合法的机会才展示，避免把规则禁止的行动伪装成策略成绩。

已有冻结候选可以仅补采 HUD：

```bash
python3 scripts/collect_monitor_diagnostics.py --run-dir /path/to/run \
  --iteration 100 --health-hands 4000 --seed 20261027 --health-only
```

只读取 `sigma_1..sigma_T` 策略档案，不重新训练、不计算 checkpoint 探针、不改既有对手胜率；新报告保存在 `monitor_hud/`。若同轮、同预算已有 HUD，会在写入前核对旧指标完全一致；重放不一致则拒绝写入。应沿用原健康评测的 seed 和预算。旧报告与原轨迹保留。

面板可以展示指标，但不会自行启动额外评测、计算未提供的置信区间，或伪造不存在的数据。未来评测器可在实验目录的 `eval.jsonl` 末尾追加一行 JSON，以换行结束。`iteration`、`update`、`step` 任一数值字段均可作为横轴；PPO 的 update 从 0 开始。既支持平铺的 `/` 标签，也支持嵌套字典或外层 `metrics`。

下例仅为格式说明，数值不是训练结果，不要写入正式实验冒充测量：

```json
{"iteration":50,"eval/winrate_vs_random_mbb":125,"eval/winrate_vs_random_mbb_ci_low":-30,"eval/winrate_vs_random_mbb_ci_high":280,"hud/preflop_vpip_sb":0.62}
```

支持 Overwatch.md 中的胜率名称：`random`、`calling_station`、`aggressive`、`slumbot`、`iter_1`、`prev_checkpoint`。完整均值标签为 `eval/winrate_vs_<opponent>_mbb`，95% CI 边界为同标签加 `_ci_low` 和 `_ci_high`。单位均为 **mbb/hand**。已有 PPO `eval_random_bb` 自动乘 1,000 转换为 mbb/hand；训练回报不转换成对战胜率。

其他标签：

- `eval/positional_sb_mbb`、`eval/positional_bb_mbb`。
- `eval/botzone_50hand_match_win_rate`：0–1 比例；仅接收真实 Botzone 比赛统计。评测报告的标准 HUNL 50 手比赛使用 `eval/hunl_50hand_match_win_rate_vs_random`。
- `hud/preflop_vpip_sb`、`hud/preflop_pfr_sb`、`hud/preflop_vpip_bb`、`hud/preflop_3bet_bb`：0–1 比例。
- `hud/action_freq_{preflop,flop,turn,river}/{fold,call,half_pot,pot,all_in}`：0–1 比例。同街需提供完整五项且总和约为 1 才显示分布；缺失行动不自动填 0。翻前 all-in > 10% 显示告警。
- `hud/policy_entropy_mean`：nats；`hud/positive_regret_actions_mean`：正 regret 合法行动数。
- `train/explained_variance_p0`、`train/explained_variance_p1`：文档中提到的可选诊断；现有 CFR 未输出时显示缺失。

胜率与 HUD 显示各自最近一次有数据的轮次，不从其他指标前向填充。要提供一致的评测快照，应在同一个评测轮次记录完整指标。文档附件在 Panel C 第一项处截断，因此第四区按现有日志实现为运行性能。

## 验证

```bash
python3 -m pytest tests/test_training_monitor.py -q
node --check tools/training_monitor/app.js
```

测试覆盖日志合并、PPO 单位、CSV / JSONL 断行、非有限数、尾部读取、续训重写、文件越界与只读行为。

## Stage 2：50 手上下文残差 PPO

保留 Claude 的 `neural` / `paper` 主题，新增 `#stage2` 导航；选择 Stage 1 时显示待接入提示。
Stage 2 输出目录放在现有 `--runs-dir` 根目录下一层（服务器默认 `/root/autodl-tmp/<run>`），
写入 `config.json`（`stage: stage2`）和 `metrics.jsonl` 后自动被发现，无需加载 Torch 或权重。

- `update` 是 PPO 更新；最新 trainer 每次更新默认采集 20 场完整 50 手比赛。`timesteps` 是英雄决策数。
- `base_kl` / `beta_kl` / `kl_penalty` 映射到 `stage2/*`，分别显示参考 KL、beta 均值及实际惩罚。
- `stage2/hero_net_bb` 是本次更新的比赛净 BB 均值；最近 100 个更新按 `stage2/sampled/*` 实际场数加权。按对手汇总使用 trainer 已记录的 `stage2/window/*` 滚动比赛窗口，读取最多 2 MiB / 5000 行尾部。
  窗口受文件大小限制，原始机会数和独立评测 CI 不由训练日志推算。
- `stage2/opponent` 保留文本身份；`stage2/pfsp/*` 为更新对手池后下一场采样概率，非实际抽样频数。
- `stage2/context/*` 为最后一场结束时的对手公开行动统计，含 Beta 先验，不是所有比赛累计 HUD。
- 总览展示 reference、strategies、残差 alpha、KL 调度、PPO 和 PFSP 配置。
  `updates` 是本次追加预算；没有 resume 起点元数据时不画虚假的全程进度百分比或 ETA。
- `seconds` 在当前 trainer 保存 checkpoint 前记录，包括采样、PPO 更新和当轮周期评测。
- 训练收益保持在训练页面；独立评测读取 `stage2/eval/*` 或 `stage2_evaluations/update_*.json`（`stage2_duplicate_matches_v1`），显示 Equity 收益/比赛胜率、冻结参考对战与 retention gate。保留真实零值和 FAIL；后续训练更新不会覆盖最近一次独立评测。

本次接入只更新监控代码，不负责启动 Stage 2、生成蒸馏参考或更改训练日志格式。

Stage 2 训练器本身已持久化 `metrics.jsonl` 与 `config.json`，控制台只读这些文件。
完整动作轨迹由独立 PR 的 writer 生成在 `evaluations/<source>/{report.json,hands.jsonl}`。
轨迹报告和 `stage2_duplicate_matches_v1` 棋力报告分别适配；有轨迹不代表有独立棋力结果。
前端、后端 helper 与网关须作为一套安装：`scripts/monitor_training.py`、
`monitor_workbench.py`、`monitor_remote.py` 和 `tools/training_monitor/` 全目录。
没有增加 Python 三方依赖，也没有将控制台接入训练的 import 路径。
