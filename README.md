# poker-ai

双人无限注德州扑克（heads-up no-limit Texas Hold'em）AI 的研究与工程仓库。当前包含标准 200BB 牌桌与 SD-CFR 训练路径、基于下注历史的 Transformer PPO 路径、50 手对手适应原型，以及 Botzone 规则机器人和模型推理入口。

**项目状态：代码和本地校验已覆盖主要训练、规则与推理链路；仓库尚未发布经长时间训练和独立对战验证的 200BB 策略。** 测试通过或入口能运行，不代表已达到特定胜率、可利用度或 Botzone 排名。

## 已实现的内容

- **标准 200BB SD-CFR：** [PokerKit](https://github.com/uoftcprg/pokerkit) 牌桌，双方各 20,000 筹码、盲注 50/100；五个抽象行动、regret matching、external sampling、双方独立 reservoir、迭代加权回归、历史策略快照与断点恢复。采样在单进程中交错推进多棵树，树结束后立即补入新根。细节见 [SD-CFR 说明](docs/sd_cfr.md)和[规则说明](docs/hunl_environment.md)。
- **Transformer PPO：** 下注历史 token 编码器、策略与价值头、自博弈及 PFSP 对手池；支持 `rlcard`、`pokerkit` 和独立的 Botzone 规则后端。另有卷积控制组、行为克隆和旧 [AlphaNLHoldem](https://github.com/bupticybee/AlphaNLHoldem) 基线适配。不同后端的行动顺序与特征语义不能混用，见 [Transformer 路径](docs/step3_transformer.md)和[控制组说明](docs/bc_and_baseline.md)。
- **50 手 residual exploiter 原型：** 冻结 SD-CFR 历史策略作为基础策略，收集对手的公开行动统计，用残差策略做短局 PPO 更新。目前是 CPU 接线和训练冒烟，未完成长期训练、质量评估或部署。见 [exploiter 说明](docs/exploiter.md)。
- **Botzone 入口：** 有规则机器人与 NumPy 模型前向的打包代码。模型入口已用开发用随机初值验证结构；仓库不附带可用于正式对战的训练权重。正式使用需另行导出真实训练权重并核对目标比赛规则。见 [规则机器人](docs/botzone_rulebot_upload.md)和[模型入口](docs/botzone_model_upload.md)。Stage 2 的前向另有 Botzone 包 `botzone_bots/stage2.zip`（Python 3.6 + NumPy、权重放用户存储空间），已用真实裁判在单核、256MB 下跑完整比赛，见 [Stage 2 Botzone 入口](docs/botzone_stage2_upload.md)。

标准 HUNL、旧 `rlcard` 环境和 Botzone simple 裁判并非同一套规则；不同深度或后端的对战指标也不能直接横比。

## Overwatch 训练控制台

完整 monitor（前端、只读日志后端、SSH 重连网关和启动脚本）随本仓库维护，
沿用神经／纸本主题。仅需 Python 3.10+，无需安装训练依赖或 npm：

```bash
python3 scripts/monitor_training.py --runs-dir /absolute/path/to/runs --open-browser
```

本地页面默认 `http://127.0.0.1:6008`。远程查看和独立 ZIP 打包见
[monitor 入口](monitor/README.md)，日志与 Stage 2 接口见 [控制台文档](docs/training_monitor.md)。

## 快速开始：SD-CFR

以下命令在仓库根目录运行，需要 Python 3.10+。PyTorch 的 CPU 或 CUDA 版本请按本机环境安装；`requirements-cfr.txt` 包含当前 CFR 路径依赖。

```bash
git clone https://github.com/Raymo-LEI/poker-ai.git
cd poker-ai
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-cfr.txt
export PYTHONPATH=src

# 只检查配置和模型规模，不启动训练
python scripts/train_cfr.py --dry-run

# CPU 短冒烟：真实牌桌采样、一次拟合和 checkpoint
python scripts/train_cfr.py --config configs/cfr/hunl_200bb.json \
  --device cpu --iterations 1 --traversals 1 --train-steps 1 \
  --out-dir runs/cfr_smoke
```

完整训练入口为 `python scripts/train_cfr.py --config configs/cfr/hunl_200bb.json`。训练输出包括 `metrics.csv`、`performance.csv`、`strategies/`、`train.pt` 和 `checkpoint_data/`；移动或备份恢复点时，`train.pt` 与 `checkpoint_data/` 要放在一起。已有输出目录续训使用 `--resume`，不要直接重跑覆盖。

训练预算的同起点对照见 [Stage 1 分阶段选拔](docs/cfr_stage1_selection.md)：默认 T=150→200 的 1200/2400 步分支、固定共同训练探针、同轮次/同算时六对手评测，以及待选的 1.49M 模型配置。完整起点缺失时先重建；不会用历史策略快照冒充可恢复样本池。

RTX 4090 服务器可从 [4090 配置](configs/cfr/hunl_4090.json)开始，它显式启用 CUDA BF16、fused Adam 和 TF32；测量命令与尚待验证的性能假设见 [服务器效率与基准说明](docs/server_4090_efficiency.md)。目前没有仓库内的 4090 实测吞吐或策略质量结论。

SD-CFR 可使用 `scripts/eval_cfr_readiness.py` 做配对对战、50 手比赛和自博弈健康检查。
`scripts/eval_cfr_periodic.py` 根据新到达的里程碑冻结策略并评测，支持完成任务跳过、
中断续跑和定时器调用。配置与 systemd 示例见 [离线及定期评测](docs/cfr_evaluation.md)。

## 其他入口

同一 Python 3.10+ 环境可检查 PPO 配置和 residual exploiter 原型：

```bash
python scripts/train_transformer.py --config configs/transformer/laptop_8gb.json --dry-run
PYTHONPATH=src python scripts/train_exploiter.py --dry-run
```

PPO 训练配置在 `configs/transformer/`，运行细节见 [训练与续训](docs/training.md)。`train_exploiter.py` 默认寻找 `runs/cfr_s2_ref/strategies`；该目录不存在时会明确报告未训练的均匀策略替身，不能把其输出当作 SD-CFR 成绩。

旧 AlphaNLHoldem 的 TensorFlow 1.15 / Python 3.7 环境使用独立的 `requirements-alphanl.txt`，与上述 PyTorch 环境分开。需要上游代码或 `c_1048` 权重时，先运行 `git submodule update --init --recursive`。

## 验证与目录

```bash
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python -m alphanl.cfr.kuhn --iterations 1000
```

- `src/alphanl/cfr/`：SD-CFR、reservoir、历史策略与 checkpoint。
- `src/alphanl/transformer/`：牌桌适配、观测编码、PPO、PFSP 和模型。
- `src/alphanl/exploiter/`：50 手上下文与残差策略。
- `scripts/`：训练、基准、行为克隆和冒烟入口。
- `eval/`、`botzone_bots/`、`tools/`：评测、Botzone 入口和导出工具。
- `docs/`：规则、算法、训练和部署说明；[调研笔记](docs/reference_survey.md)记录早期基线与参考资料。

## 许可与第三方代码

仓库根目录目前没有 `LICENSE` 文件。`third_party/AlphaNLHoldem` 是独立的 Git 子模块；使用其代码以及其他依赖时，请分别查看相应上游项目的许可文件。

Deep-stack CFR action abstraction and evaluation: [all-street SPR guard](docs/cfr_spr_guard.md). The supporting shared position-conditioned training path is documented in [Unified SD-CFR](docs/cfr_unified.md).

Offline average-strategy compression is available through [SD-CFR distillation preparation](docs/cfr_distillation.md): frozen archive validation, own-reach-preserving trajectory labels, a single policy student, held-out diagnostics, and a portable checkpoint. Playing-strength and Stage 2/Botzone acceptance remain separate gates.

The opt-in [Stage 1 full-capacity profile](docs/cfr_fullupgrade.md) combines equity/odds inputs, a 1.49M model, one 8M reservoir, polynomial regret regression and a 15% fixed-opponent root mixture. CPU wiring and frozen T200 loss diagnostics are available; this profile has not yet been deployed or validated on CUDA.

## Stage 2：50 手跨手上下文残差 PPO

共享 Stage 1 输入契约、平均策略蒸馏、冻结参考残差 PPO、PFSP、NumPy 单文件导出与 CPU smoke 已提供；不会自动启动 GPU 训练。入口、测试及真实裁判适配边界见 [Stage 2 交接文档](docs/stage2_match_residual.md)。四层 MatchPool、整局风格抖动、固定 35% 防守配额和独立评测见 [MatchPool 升级说明](docs/stage2_matchpool.md)。

Independent [Pro Gauntlet evaluation](docs/pro_gauntlet.md) adds TAG/check-raise
style probes, frozen-policy Slumbot evaluation and cross-generation duplicate
tournaments. See the [implementation and 100,000-hand report](docs/pro_gauntlet_report_20261029.md)
for the published matrix, uncertainty and provisional selection, and the
[external-opponent assessment](docs/open_source_stage2_opponents.md) for future
training-pool candidates. No training pool or deployment is changed automatically.

The opt-in [Stage 2 v3 Pro Hunter](docs/stage2_pro_hunter.md) adds jittered
TAG/Trapper training families, an opportunity-based trap HUD, C200 weight-only
warm starts and residual-output blend evaluation. Its report includes the
expanded same-deal matrix and the completed 500-hand C200 Slumbot test.
