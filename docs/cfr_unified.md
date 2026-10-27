# Unified Position-Conditioned SD-CFR

默认架构为一个位置条件 regret 网络、一个全局 Algorithm-R reservoir 和一个 Adam 优化器。`buffer_capacity` 仍表示旧版单人容量；共享池容量自动乘二，默认 4,000,000 条。每条额外存储 uint8 `hero_seat`。P0=BB，P1=SB/Button。

`HunlTokenizer` 在永不截断的 `[HOLE]` token 的四维数值通道写入 `[is_BB, is_SB, 0, 0]`，开局无动作历史也有显式位置；原有 CLS 底池、剩余筹码与跟注额度均保留。旧归档按其旧配置使用不含此特征的编码。

每轮双方各采集 `traversals_per_player` 棵根树，双方均查询共享网络；训练前双方策略冻结。样本统一入池，保留概率不按座位改变。每个训练 batch **精确各半** 从 BB/SB 层内有放回均匀抽样并打乱，CPU 与 GPU 路径一致。若任一座位没有保留样本，本轮延迟拟合并记录 `fit_deferred_missing_seat`，不会把缺失位置伪造为另一位置。每次采样阶段结束后构建座位索引，训练时不逐批扫描全池。

默认 warm-start，Adam 状态跨轮保留。默认训练 1200 steps × 1024 batch；独立验收 runner 使用 1500 steps × 1024 batch。CUDA 启用 BF16 autocast、fused Adam、TF32，regret head、目标和 Loss 保持 FP32；CPU 自动退回普通 FP32/Adam。GPU 训练池按完整容量分配以稳定 CUDA allocator 保留显存，并只从已写入的有效前缀采样；4M 池会从第一次拟合起预留约 5.8GB 数据与索引空间。

遍历的终局效用原本就以 **BB** 为单位，训练时再除以 `regret_target_scale=100`，200BB 目标成为 2.0，不会重复按筹码除大盲。默认目标为 `E[t × sum_legal SmoothL1(pred, target/100, beta=1)]`。regret matching 对正比例缩放不变。Huber 改变了拟合目标，对外采样噪声更稳健，但这仍需棋力验证，不能由拟合 Loss 推导均衡保证。旧版 MSE 可用 `--loss mse --target-scale 1` 对照。

共享模式每轮只做一次网络更新，策略归档每轮只包含一份 `models['shared']`。保留原 `iteration_0000001.pt` 文件命名与线性加权按手采样的平均策略；sigma_t 是本轮遍历实际使用的 theta_(t-1)，因此快照在本轮训练前冻结。训练 checkpoint 为 v3、策略快照为 v2；仍可读取旧版双网络 v2/v1。新旧架构不能直接混合续训。`--separate-player-models` 保留独立网络/池消融，其他新训练设置相同；完整旧行为还需相应旧编码、单位、Loss、重置和 guard 配置。旧 JSON 配合 `--resume` 自动按旧语义解析。

默认启用全街 SPR 门控：有效 All-in 投入超过当前底池 3 倍且本街主动下注/加注少于两次时，移除 index 4。它不改变底层 NL 规则，也不限制评测对手。旧归档缺失 `spr_action_guard` 时保留原动作树，原翻前 guard 仅作兼容。完整定义、显式对照与新训练命令见 [SPR 门控](cfr_spr_guard.md)。结构约束产生的低全下率不能宣称为模型自主学会合理策略。

全负遗憾默认用单位温度 softmax + 5% 合法动作均匀探索；零预测严格均匀，不会固定选择尾索引。`uniform`、`argmax` 保留作消融。

监控记录共享 Loss、共享拟合耗时及吞吐，以及两席独立 R²、真实 End/Start、样本/Root、入池比例与训练抽样数。Loss 是缩放后的 Huber；固定批次诊断仍以缩放后的平方误差计算加权 R² 与 End/Start，比例对目标共同缩放不变。探针是保留训练数据，非泛化验证集；不可在训练探针上特意额外拟合。每席 `fit_acceptance` 如实检查 R² > 0.50 且 End/Start < 0.30，未通过就显示未达标。Warm-start 已拟合较好时单轮 End/Start 未必持续下降至 0.30，因此要与绝对 R²、独立对战一起解读。

新训练与冻结策略的评测：

```bash
PYTHONPATH=src python scripts/train_cfr.py --config configs/cfr/hunl_4090_parallel.json --out-dir runs/cfr_spr_new
python scripts/eval_cfr_readiness.py --strategies runs/cfr_spr_new/strategies --out runs/cfr_spr_new_eval --hero-iteration 100
```

评测保留完整线性加权按手混合、交换座位发牌与按牌对计算的置信区间。旧模型的门控消融不代表新动作树已完成训练。
