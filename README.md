**中文** · [English](README.en.md)

# poker-ai

双人无限注德州扑克（heads-up no-limit Texas Hold'em）的研究与工程仓库。当前代码覆盖标准 200BB 的 SD-CFR、Transformer PPO、50 手残差 Stage 2，以及接到 Botzone 裁判上的规则机器人、NumPy 推理包和 Route 1 动作契约。

**项目状态：训练、规则差分、断点、导出和 CPU 冒烟已经接上。仓库没有附带经长时间训练并在独立对战里验收过的 200BB 策略，也没有 Botzone 排名或 GTO Wizard 排行榜成绩。** 测试通过只说明对应契约和接线成立。各轮结果以 `eval/r02`–`eval/r06`、`eval/r08` 的 `RESULT.md` 为准；`eval/r07` 不在本树。

## 三条牌桌

三条路径的盲注、行动顺序和动作宽各自成套。checkpoint、特征版本和指标都按自己的牌桌解释。

| 路径 | 牌桌 | 动作 |
| --- | --- | --- |
| SD-CFR，以及读它的 Stage 2 参考 | 标准 200BB：双方 20,000，盲注 50/100，每手重置 | 五个抽象动作。Stage 2 的特征版本和网络宽度跟参考 checkpoint 走 |
| Transformer PPO | `rlcard`、`pokerkit`，或独立的 Botzone 后端 | 五档。后端之间的 checkpoint 互不加载 |
| Route 1（R02 起） | 未修改的官方 Botzone 裁判 `Holdem.player_action` | `botzone_a5_v2` 为 fold / check-call / half-pot / pot / all-in。`botzone_a6_v2` 在末尾追加 `quarter_pot`。旧 `botzone_5action_v1` 只作 V6 对照，schema 哈希不符就拒绝加载 |

标准 HUNL、旧 `rlcard` 环境和仓库里早期的 Botzone simple 夹具也彼此不同。R02 之后的内核差分对着官方裁判，见 [R02 结果](eval/r02/RESULT.md)。

## Stage 1：标准 200BB SD-CFR

[PokerKit](https://github.com/uoftcprg/pokerkit) 牌桌，五个抽象行动，regret matching，external sampling，双方独立 reservoir，迭代加权回归，历史策略快照和断点恢复。采样在单进程里交错推进多棵树，一棵结束就补入新根。规则见 [HUNL 环境](docs/hunl_environment.md)，算法见 [SD-CFR 说明](docs/sd_cfr.md)。

已经接上、并有专门说明的部分：

- 全街深筹码 all-in 护栏，以及共享的位置条件训练：[SPR guard](docs/cfr_spr_guard.md)、[统一 SD-CFR](docs/cfr_unified.md)。
- T=150→200 的预算对照、共同训练探针和同轮次评测：[Stage 1 分阶段选拔](docs/cfr_stage1_selection.md)。
- 可选满配：equity/赔率输入、1.49M 模型、单侧 8M reservoir、多项式 regret 回归、15% 固定对手根混合。CPU 接线和对冻结 T200 的损失诊断在仓库里；这份配置尚未在 CUDA 上部署或验收，见 [满配说明](docs/cfr_fullupgrade.md)。
- RTX 4090 配置显式打开 CUDA BF16、fused Adam 和 TF32。测量命令见 [服务器效率](docs/server_4090_efficiency.md)，吞吐分阶段见 [P0–P3](docs/cfr_performance_rollout.md)。仓库内没有把某次 4090 跑分写成已发布的策略质量结论。
- 外部采样在写入 regret 时，若 buffer 提供 `add_strategy`，会同时记下当前策略，权重为 `iteration * traverser_reach`。并行 worker 把策略和这份权重跟 regret 放进同一共享槽。主进程在 buffer 提供 `strategy_sink` 时收下，供之后的策略拟合复用同一次遍历。见 `src/alphanl/cfr/traversal.py` 与 `src/alphanl/cfr/parallel.py`。

配对对战、50 手比赛和自博弈健康检查走 `scripts/eval_cfr_readiness.py`。`scripts/eval_cfr_periodic.py` 在新里程碑到达后冻结策略再评测，支持跳过已完成任务和中断续跑。配置与 systemd 示例见 [离线及定期评测](docs/cfr_evaluation.md)。

## 平均策略蒸馏

离线压缩见 [蒸馏准备](docs/cfr_distillation.md)：冻结归档校验、保留 own-reach 的轨迹标签、单个策略学生、留出诊断和可携带 checkpoint。对局强度、Stage 2 和 Botzone 验收是后面的门。

[精确混合的 fidelity 蒸馏](docs/distillation_fidelity.md)在此之上加了窗口内后验标签、按街和底池加权的前向 KL、开发集选 checkpoint，以及六对手重复牌面的强度门槛。CPU 和合成测试只固定契约与算术。未达到门槛时 Stage 2 保持暂停；通过之后也只有同时给出 `--stage2-template` 和 `--stage2-out` 才会另开一份新的 Stage 2。

## Transformer PPO

下注历史 token 编码器、策略头和价值头、自博弈及 PFSP。支持 `rlcard`、`pokerkit` 和 Botzone 规则后端，另有卷积控制组、行为克隆，以及旧 [AlphaNLHoldem](https://github.com/bupticybee/AlphaNLHoldem) 基线适配。见 [Transformer 路径](docs/step3_transformer.md)、[控制组](docs/bc_and_baseline.md)和 [Botzone 里程碑 M1–M12](docs/botzone_transformer_milestones.md)。

配置在 `configs/transformer/`。续训见 [训练说明](docs/training.md)。Windows 上进程被杀掉之后，`tools/train_autoresume.ps1` 只从写完的 PPO checkpoint 拉起，见 [自动续训](docs/train_autoresume.md)。

`src/alphanl/exploiter/` 是更早的 50 手残差原型，只做到 CPU 接线和训练冒烟。现行的 50 手适应在 Stage 2，见下一节。原型说明在 [exploiter](docs/exploiter.md)。

## Stage 2：50 手残差 PPO

冻结的 Stage 1 学生作参考，残差策略在 50 手比赛里用公开行动统计做 PPO。共享输入契约、PFSP、NumPy 单文件导出和 CPU smoke 已在仓库中；这些入口不会自行启动 GPU 训练。见 [Stage 2 交接](docs/stage2_match_residual.md)。

其后的可选层各自有开关和文档：

- 四层 MatchPool、整局风格抖动、固定 35% 防守配额：[MatchPool](docs/stage2_matchpool.md)。
- TAG / check-raise 探针、冻结策略的 Slumbot 评测、跨代重复牌面：[Pro Gauntlet](docs/pro_gauntlet.md) 与 [2026-10-29 报告](docs/pro_gauntlet_report_20261029.md)。训练池人选的外部评估见 [开源对手笔记](docs/open_source_stage2_opponents.md)。
- TAG/Trapper 训练族、trap HUD、C200 纯权重热启动和残差输出混合：[Pro Hunter](docs/stage2_pro_hunter.md)。
- `configs/stage2/botzone_native.json`：softmax 前的主动 all-in 掩码，以及三名天梯复刻对手。开关关闭时，Pro Hunter 权重和 v10 护栏保持原样，见 [Botzone-native](docs/stage2_botzone_native.md)。
- 部署护栏写在 `stage2_model.npz` 的 `meta_json.deployment_guard`。旧权重没有该字段时护栏关闭。见 [v10 护栏](docs/botzone_v10_guard.md)。

NumPy 运行时在交出动作概率前调用 `alphanl.stage2.wizard.fuse`。1.49M 蓝图权重只读。没有叶值头 `vleaf.*` 时，函数原样返回蓝图概率（`wiz=0`）。有叶值头时，翻前仍返回蓝图；翻后只在底池至少 6BB，或英雄正面对下注时搜索。搜索使用 8 个范围桶、2 条发牌和 4 次 CFR，再与蓝图混合。异常退回蓝图（`wiz=err`）。搜索按五个动作 id 进行，并随 Stage 2 源码包导出。契约测试是 `tests/test_wizard_solver.py`。这里没有单独的强度报告。

## Botzone 交付包

| 包 | 用途 | 说明 |
| --- | --- | --- |
| `botzone_bots/rulebot.zip` | 规则机器人 | [上传说明](docs/botzone_rulebot_upload.md) |
| `botzone_bots/model.zip` | 无 Torch 的 NumPy 前向 | 开发用随机初值验证过结构。仓库不附带可上场的训练权重。[上传说明](docs/botzone_model_upload.md) |
| `botzone_bots/stage2.zip` | Stage 2 前向，Python 3.6 + NumPy。权重放用户存储，文件名 `stage2_model.npz` | 上传步骤，以及早前一次单核、256MB、完整 50 手的记录，见 [上传说明](docs/botzone_stage2_upload.md)。当前包含蓝图子博弈钩子；没有叶值头时动作仍来自蓝图。那次记录写在 v10 和这个钩子之前 |

打包入口是 `tools/pack_botzone_rulebot.py`、`tools/pack_botzone_model.py`、`tools/pack_botzone_stage2.py`。v10 护栏在 Stage 2 打包时用 `--v10-guard` 或 `--guard-config` 写入。外层交付 ZIP 要先拆开：内层源码 ZIP 上传为代码，`stage2_model.npz` 上传到用户存储。

## Route 1：裁判内核与 A5/A6 契约

这一路把训练和评测收到官方裁判与显式动作 schema 上。R02–R06 与 R08 的状态取自各自的 `RESULT.md`。

| 轮次 | 状态 | 仓库里有什么 |
| --- | --- | --- |
| [R02](eval/r02/RESULT.md) | PASS，GPU 小时 0 | 可分支内核按未改的官方裁判逐步转移。合法拒绝、下一行动者、底池、筹码、街、摊牌收益的差分记录为 0 |
| [R03](eval/r03/RESULT.md) | PASS，GPU 小时 0 | 同一 V6 部署包上的四层归因和反例。Teacher 快照与冻结学生 `policy_best.pt` 未附带，这两层标为 MISSING |
| [R04](eval/r04/RESULT.md) | PASS，GPU 小时 0 | A5/A6 共用规则、公开特征和零训练预算。A6 只多 `quarter_pot`。边界测量给出 `quarterpot_effective_new_coverage = 0.224`，结果文件写明这不代表已经获得小注能力 |
| [R05](eval/r05/RESULT.md) | INCOMPLETE | CPU 上量过内核对局、导入和拷贝耗时。GPU 探针未跑，预算提案未批准，`total_gpu_hours_cap` 仍空 |
| [R06](eval/r06/RESULT.md) | INCOMPLETE | Kuhn 正确性和 checkpoint 恢复在 CPU 上跑过。A5/A6 的 GPU pilot 未启动。在 CPU 宿主上，`eval/r06/launch_pilot.py --execute` 只做一步有限反传，不启动 CUDA |
| R07 | 本树未收录 | 监控可以读运行目录里的 `learning_curve_live.json`。`eval/r07` 与其 manifest 不在仓库中 |
| [R08](eval/r08/RESULT.md) | NOT_RUN | Teacher 仍是 PENDING_TEACHER。`distill_gate.py` 打印命令后以退出码 2 停下。蒸馏、SD-CFR 和 PPO 都未从这里启动 |

GTO Wizard 对照是离线夹具，外加一个需要密钥的 live smoke。离线命令不访问网络，也不读 teacher 目录：

```bash
python3 scripts/eval_gto_wizard_offline.py \
  --fixture eval/gto_wizard/fixtures/spots.json \
  --out-dir runs/gto_wizard_fixture
```

`kl` / `l1` 是夹具分布之间的距离。筹码和 AIVAT 在离线报告里为空。live 入口是 `scripts/eval_gto_wizard_live.py`：密钥放在环境变量 `GTOWIZARD_API_KEY`，只请求 `/hands` 与 `/hands/{id}/act`，不提交排行榜。上游客户端仓库没有许可证，本仓库不内嵌其源码。见 [GTO Wizard 说明](docs/gto_wizard_eval.md) 和 `eval/gto_wizard/NOTICE.md`。

## Overwatch 训练控制台

只读已有日志，不加载模型。Python 3.10+ 标准库即可，无需训练依赖或 npm。

```bash
python3 scripts/monitor_training.py --runs-dir /absolute/path/to/runs --open-browser
```

本地页面默认 `http://127.0.0.1:6008`。`--runs-dir` 可重复，每个值是一次 run 或其直接父目录；发现不会递归进更深的子目录。远程查看、`--source-label`、`learning_curve_live.json`，以及 `?scope=all`（未标注的 run 会显示出来，同时仍保持非 formal）见 [monitor 入口](monitor/README.md)。工作台页面和 Stage 2 日志字段见 [控制台文档](docs/training_monitor.md)。

训练进程用 `--experiment-kind` 自报身份，取值为 `formal`、`development`、`smoke`、`benchmark`、`evaluation`。控制台按日志里的这个字段筛选，不会把缺标签的目录改写成 formal。

macOS 可双击仓库根目录的 `打开训练监控.command` 与 `连接服务器监控.command`。

## 快速开始

命令在仓库根目录执行，Python 3.10+。PyTorch 的 CPU 或 CUDA 轮子按本机安装。`requirements-cfr.txt` 含当前 CFR 路径（它包含 `requirements-transformer.txt`）。

```bash
git clone https://github.com/Raymo-LEI/poker-ai.git
cd poker-ai
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-cfr.txt
export PYTHONPATH=src

# 只检查配置和模型规模
python scripts/train_cfr.py --dry-run

# CPU 短冒烟：真实牌桌采样、一次拟合、一份 checkpoint
python scripts/train_cfr.py --config configs/cfr/hunl_200bb.json \
  --device cpu --iterations 1 --traversals 1 --train-steps 1 \
  --out-dir runs/cfr_smoke
```

完整训练是 `python scripts/train_cfr.py --config configs/cfr/hunl_200bb.json`。输出含 `metrics.csv`、`performance.csv`、`strategies/`、`train.pt` 和 `checkpoint_data/`。移动恢复点时，`train.pt` 与 `checkpoint_data/` 放在一起。已有目录续训用 `--resume`。

4090 起点是 [hunl_4090.json](configs/cfr/hunl_4090.json)，多 worker 起点是 [hunl_4090_parallel.json](configs/cfr/hunl_4090_parallel.json)。

同一环境可以检查 PPO 配置和早期 exploiter 原型：

```bash
python scripts/train_transformer.py --config configs/transformer/laptop_8gb.json --dry-run
PYTHONPATH=src python scripts/train_exploiter.py --dry-run
```

`train_exploiter.py` 默认寻找 `runs/cfr_s2_ref/strategies`。该目录不存在时，脚本会报告自己正在使用未训练的均匀策略替身。

Stage 2 没有 dry-run。`scripts/train_stage2.py` 的 `--out-dir` 必填，默认配置是 `configs/stage2/botzone_50hand.json`，并且会去读配置里的参考 checkpoint 和策略目录。接口冒烟是另一条命令：它在新目录里写未训练的合成 teacher，做两次配额块，再导出，目录必须事先不存在。

```bash
PYTHONPATH=src python scripts/smoke_stage2.py --out-dir runs/stage2_smoke
```

这次冒烟写明自己只验证接口和 CPU 执行。它的输出不能当作 Stage 1 或 Stage 2 的强度。

旧 AlphaNLHoldem 使用独立的 Python 3.7 / TensorFlow 1.15 环境，依赖文件是 `requirements-alphanl.txt`。需要上游代码或 `c_1048` 权重时：

```bash
git submodule update --init --recursive
```

50BB 截断的 Slumbot 客户端说明在 [eval/SLUMBOT_CLIENT.md](eval/SLUMBOT_CLIENT.md)。它和上面的 200BB SD-CFR 是两套栈。

## 验证与目录

`pytest.ini` 已把 `pythonpath` 设为 `src`。

```bash
python -m pytest -q
PYTHONPATH=src python -m alphanl.cfr.kuhn --iterations 1000
```

Kuhn 是表格型 external-sampling CFR 的玩具博弈，用来核对采样和可利用度下降。它的数字不是 Hold'em 收敛结果。R06 记录过 1000 次迭代、种子 1 的一次 CPU 运行。

- `src/alphanl/cfr/`：SD-CFR、reservoir、并行采样、历史策略、蒸馏、评测与 checkpoint。
- `src/alphanl/transformer/`：牌桌适配、观测编码、PPO、PFSP 和模型。
- `src/alphanl/botzone/`：官方裁判内核、合法响应、A5/A6 schema、规则机器人。
- `src/alphanl/stage2/`：50 手环境、残差策略、对手池、导出、部署护栏和蓝图子博弈搜索。
- `src/alphanl/exploiter/`：更早的 50 手残差原型。
- `scripts/`：训练、评测、基准、监控和冒烟入口。
- `configs/cfr/`、`configs/transformer/`、`configs/stage2/`：对应路径的配置。
- `eval/`：Slumbot 客户端、GTO Wizard 脚手架，以及 R02–R06、R08 的结果和清单。
- `botzone_bots/`、`tools/`：可上传的入口和打包工具。
- `docs/`：规则、算法、训练和部署说明。[调研笔记](docs/reference_survey.md)记录早期基线。
- `README.en.md`：这份说明的英文版。页首链接可以切过去。
- `monitor/`：控制台的独立说明。前端源码在 `tools/training_monitor/`。

## 许可与第三方代码

仓库根目录目前没有 `LICENSE` 文件。`third_party/AlphaNLHoldem` 是独立的 Git 子模块。使用其代码以及其他依赖时，分别查看对应上游项目的许可。GTO Wizard 公开客户端同样没有随仓库附带许可证，因此没有被内嵌。
