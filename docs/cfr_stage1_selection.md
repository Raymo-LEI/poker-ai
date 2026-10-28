# Stage 1 分阶段选拔：训练预算分支与 1.49M 配置

本次训练步数对照固定 **T=150→200**。2026-10-28 接管时，v2 只有 T=250 的完整 `train.pt`，没有 T=150 的 Reservoir/RNG/Adam 状态。历史 sigma_t 只用于策略评测，不能重建这些状态。找不到 T=150 完整状态时，先按基线 seed=1 配置从头重建到 T=150，再冻结新起点；明确标为新基线轨迹，不冒充历史 v2 的逐位复现。

## 可恢复分支

`SDCFRTrainer.load_checkpoint(path, branch_train_steps=True)` 和 CLI `--branch-train-steps` 显式允许更改 `train_steps`。普通 resume 保持严格校验。模型、编码、游戏规则、容量、loss、学习率等仍不能通过该选项改变。目标必须是独立的新输出目录；不能覆盖源 run 或已有实验。

分支恢复网络、Adam、整个 RAM/memmap Reservoir（含 seen 和 RNG）、遍历 RNG、worker RNG、GPU 抽样 RNG、Torch CPU/CUDA RNG，以及历史策略和日志。为了保留 worker RNG，正式实验保持与 checkpoint 相同的 16 workers。新 `ops/branch.json` 记录来源及有效配置。大训练 checkpoint 使用 pickle protocol 4，策略快照继续使用 weights-only 可读取的格式。

已实测 v2 的 T=250 策略文件在服务器 PyTorch 2.8 中无法以 weights_only=True 读取（protocol 4 / unsupported operand 149）。显式分支会从可信 checkpoint 引用的策略文件读取权重，校验版本和轮次后，仅在新目录重新编码为 protocol 2。原文件不改写；SnapshotPolicy 的普通安全读取方式不放宽。

示例：

```bash
PYTHONPATH=src python scripts/train_cfr.py \
  --config /path/to/source_config.json --resume /path/to/train.pt \
  --branch-train-steps --train-steps 2400 --iterations 200 \
  --out-dir /path/to/new_branch
```

## 一次编排

没有历史 T=150 完整状态时，先重建基线，一次只启动一个 trainer：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=src python scripts/run_cfr_stage1.py run \
  --bootstrap-config configs/cfr/stage1_059m_4m_1200.json \
  --study /root/autodl-tmp/cfr_stage1_budget_20261028 \
  --run-root /root/autodl-tmp --source-iteration 150 --target 200 \
  --pairs 2500 --eval-workers 6 --probe-rows 4096 --seed 20261028
```

已有 T=150 完整状态时，将 `--bootstrap-config` 换为 `--checkpoint /path/to/t150/train.pt`，两者互斥。脚本校验完整状态的轮次，拒绝用 T=250 等其他轮次代替 T=150。

Study 目录必须不存在。编排过程以硬链接固定完整源文件和 checkpoint_data。文件原子替换不会改写该冻结副本；原 run 保留不动。两个训练臂和新建基线位于 run-root 的直接子目录，标准监控可发现。每 10 轮和终点保存完整 checkpoint，阶段状态为 `status.json`，失败写 traceback 并停止，不静默换配置。

### 从 T=100 起每 10 轮保留完整恢复点

三个 Stage1 配置都显式设置 `checkpoint_archive_from=100` 和 `checkpoint_archive_every=10`，后续 1.49M 正式训练也沿用。`train.pt` 仍是最新恢复点；另外保留 `checkpoint_archives/iteration_0000100.tar.gz`、110、120 等不可覆盖的压缩归档。每份包含模型、Adam、完整 Reservoir、RNG、日志和全部策略引用；历史网络按 SHA-256 在同目录 `objects/` 只存一份，避免每个恢复点重复存整段历史。RAM 与 memmap 均支持，旧 checkpoint_data 被回收后仍可恢复。

本次重建 100…150 共 6 点，两分支 160…200 各 5 点，共 16 点。原始满池快照总量超过当前 50GB 数据盘，不能直接无限复制。归档采用 gzip level 1；每次记录实测压缩体积、剩余归档数、满池增长估计和 50% 余量。压缩过程中检查空闲磁盘，额外保留下一次原始 checkpoint 写入空间和至少 2GiB 空间；若剩余归档的预计空间不足，保留已发布的恢复点并明确停止，不删除旧实验或默默漏存。预测不是压缩率保证，必须以实际归档报告为准。

**移动恢复点时须同时保留归档、同名 `.json` 和整个 `objects/` 目录。** 恢复工具先校验指纹和路径，再在新目录展开，不执行 pickle：

```bash
python scripts/restore_cfr_checkpoint.py \
  --archive /path/to/run/checkpoint_archives/iteration_0000150.tar.gz \
  --out-dir /path/to/restored_t150
```

随后可将 `/path/to/restored_t150/train.pt` 传给分支脚本的 `--checkpoint`。归档的实际体积和空间预估写入 `checkpoint_archives.jsonl` 与每份 JSON 清单。

训练顺序为新基线 1200 步到 T=150（如果需要重建）、冻结完整起点、1200 分支到 T=200、2400 分支到 T=200、CPU 六类对手评测。正式模式还会检查 T=150 的池是否确实满容量；未满则停止，不会仅根据轮次标成满池。`arm1200.json` 和 `arm2400.json` 记录逐轮累计时间和资源采样。RAM 报告采用 cgroup 而非宿主机的 free；每轮边界检查 90,000,000,000 字节 RAM 和 24,000,000,000 字节 VRAM 上限。边界采样不能替代拟合期间峰值监测；CUDA allocator 的峰值另存，扩 8M 前仍须实测。

### 共同固定探针

源池每席抽 4096 行，使用独立 NumPy RNG，保存一次并校验 SHA-256。两臂每轮在相同的行、目标、样本轮次权重上，使用 FP32 测 `train/fixed_probe_r2_p{0,1}` 和 `train/fixed_probe_normalized_mse_p{0,1}`。归一 MSE 为 `sum(t_i * sum_legal(error_i²)) / sum(t_i)`。

**这是固定的保留训练样本诊断，不是独立验证集。** 不把它标为 held-out，不改变原训练抽样 RNG，也不替换原每轮重新抽取的 End/Start 探针。监控下拉新增“跨分支共同探针”两项，卡片显示探针来源轮次和分支训练计时。

### 相同轮次与相同算时

同轮比较两臂 T=200 的全历史线性平均策略。计时从恢复结束后的第一轮开始，累计 `run_iteration + 固定探针 + 周期 checkpoint` 的墙钟耗时；不含前期数据加载、排队、共同前缀重建和对手评测。

1200 臂到终点的耗时是共同预算。2400 臂保存不超过此预算的最后一个完整轮次策略前缀，记录实际用时及剩余预算；不用插值伪造精确时刻的网络。该前缀是评测归档，明确标记不能恢复训练。原始继承前缀仍参与完整平均策略，因此这是“共同训练历史后的分支干预”，不是全程 2400 步的新实验。

每个有效策略前缀对 random/call/maniac/overbet/nit/equity 各打 2500 对镜像牌。牌种子只依赖 study seed 和对手名称，不依赖候选、迭代或比较模式。保存逐牌对日志、mbb/hand、以牌对为独立单位的 95% CI，以及 2400 相对 1200 的配对差值 CI。相同策略前缀复用同一评测，避免重复消耗。源代码和评测快照有哈希，评测结束重新检查归档未变化。

结果写 `evaluation_report.json`，并将真实六对手收益发布到各臂 `eval.jsonl`。`training_budget_winner` 暂不自动赋值：预算优劣需结合逐对手短板和同算时结果，不用未约定的任意加权分数宣布冠军。

只读监控入口：`python scripts/monitor_training.py --runs-dir /root/autodl-tmp --port 6008`。选择两个独立分支查看共同探针 R²、归一 MSE、分支训练时间和已完成对战评测。监控不加载训练 checkpoint，不启动评测；服务器通过既有 SSH 隧道访问。前端无 CDN 依赖。

## Config 3 准备

`configs/cfr/stage1_149m_4m_1200.json` 与 `stage1_149m_4m_2400.json` 是两个备选模板，不会同时启动。架构固定 d=192、4L、6H、FFN=512、max_seq_len=64，统一池实际 4M，初始 lr=3e-4。骨干实际 1,491,206 个可训练参数；包含 regret head 并冻结 PPO 双头后可训练参数为 1,491,013；存储参数共 1,492,171。小模型对应 592,390 / 592,261 / 593,035。

阶段一结论出来后确定步数，Config3 从头使用训练 seed=1，不能载入 0.59M 权重假装同架构恢复。manifest 中的单核 CPU 延迟是 batch=1、满序列长度的合成输入 smoke，仅计 FP32 model.strategy，不含编码、快照 I/O 和进程启动；不能冒充 Botzone 端到端验收。正式门槛还要在真实决策分布和目标部署环境验证。

8M 池、同种子完整模型/容量训练、burn-in=50、四候选 cross-play 和 Stage2 最终红线属于后续晋级阶段。本轮不在这些结果缺失时声称已选出 Stage2 底座；后续 burn-in 实验须保留幸存快照的绝对轮次 t 权重。

## 验证

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m pytest tests/test_cfr_stage1.py tests/test_cfr_trainer.py \
  tests/test_cfr_unified.py tests/test_cfr_buffer_checkpoint.py \
  tests/test_cfr_diagnostics.py tests/test_training_monitor.py -q -o addopts=''
node --check tools/training_monitor/app.js
```
