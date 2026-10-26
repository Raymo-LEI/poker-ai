# RTX 4090 服务器效率审计

2026-10-27 后续实施、已验证范围与服务器对比命令见 [CFR P0–P3 实施与验证](cfr_performance_rollout.md)。下文保留审计时点的判断。

审计日期：2026-10-27。初始审计基线：`9816517`；本 PR 基于最新 `main`：`b425faf`。参考附件：`4090.md`。
附件中的命令式文字按待评审方案处理，没有当作修改仓库的授权。

硬件按截图：RTX 4090 24GB × 1、25 vCPU Xeon Platinum 8481C、90GB RAM、
30GB 系统盘、50GB SSD 数据盘；镜像为 PyTorch 2.8.0、Python 3.12、CUDA 12.8。
截图不能证明 25 个独占物理核、NVMe、实际可用空间或 PCIe 带宽，需要服务器实测。

## 结论

第一优先级是 CPU 游戏树状态复制与采样调度，其次是多进程 CPU + 单 GPU 聚批，
再做 BF16、fused optimizer、同步点与数据搬运优化。暂不按附件一次性预留 20M
显存样本、16K–32K batch，也不把 GPU 利用率作为唯一目标。

判断标准应是：完整流程每小时产生多少有效 regret 样本，以及相同算时下策略质量。
GPU 可以在 CPU 遍历期间利用率较低；为了拉高利用率而增加无收益训练量并不省钱。
按截图单价 ¥2.08/小时，可用 `2.08 × 1e6 / 每小时有效样本数` 比较每百万样本成本；
样本成本还需配合相同规则、抽象和评测协议下的策略成绩解释。

本轮没有连接服务器，没有得到 CUDA 吞吐或 GPU 峰值内存数据。本 PR 增加了可选 BF16、fused Adam、TF32，树结束后立即补入新根，以及 `performance.csv` 计时；旧配置默认仍为 FP32、普通 Adam、TF32 关闭。

## 1. 代码现状与证据

### 1.1 采样没有多进程，实际推理批次远小于树数上限

`src/alphanl/cfr/traversal.py` 是单进程协作 DFS；`batch_size` 是同时推进的根树数。
每轮请求按玩家拆成两组，分别调用各自的 regret network。原先整波等待已改为树结束即补根，
降低异构树深度造成的尾部空洞，但不改变遍历语义或多进程状态。
默认 `traversals_per_player=128`、`traversal_batch_size=32`；只把后者改成 1408，
也不会得到 1408 条实际推理。`scripts/train_cfr.py` 的 `--threads` 默认还为 1。
把这个线程数改成 25 只影响 torch 线程池，不能让 Python/PokerKit 遍历使用 25 个进程。

每次 GPU 推理后，`strategy(...).double().cpu().numpy()` 把结果取回 CPU；
还把已经传入 GPU 的 legal mask 取回 CPU 验证。必要的决策回传应聚批，
legal mask 则可保留 CPU 原件，减少冗余传输。

### 1.2 分支复制成本很高

`src/alphanl/cfr/traversal.py:50` 每个 child 调用 `table.clone()`，
`src/alphanl/transformer/hunl.py:106` 用 `copy.deepcopy(self)` 复制整张牌桌。
树分支包含 PokerKit 状态、牌堆、历史、RNG 等 Python 对象。

本地定位实验（不是服务器基准）：macOS ARM CPU、torch 2.14.1、单线程；
完整默认 regret 模型，提前预热 evaluator 和 forward；真实 HUNL 根 seed 1000–1015，
traverser=1、16 棵树、遍历 RNG=201、cProfile 开启。

- 1569 个节点，679 个终局，297 条 regret 样本，计时 3.065 秒。
- `HunlTable.clone` 调用 1553 次，累计 2.014 秒，占约 65.7%。
- 890 条推理输入，260 次推理，平均 batch 仅 3.42。
- `RegretNetwork.strategy` 累计约 0.363 秒，占约 11.9%。
- `observation` 累计约 0.271 秒；reservoir.add 累计约 0.015 秒。

累计调用耗时包含嵌套，不能相加；cProfile 会放大 Python 开销，
且一个种子/策略的小样本不能外推完整长跑或 4090 加速比。
这些数据足以支持先检查状态复制和批次尾部，不能证明服务器上瓶颈占比相同。

### 1.3 CFR 增加了 opt-in AMP 与 fused optimizer，尚无 compile

`src/alphanl/cfr/trainer.py` 支持可选 BF16 autocast 和 `Adam(fused=True)`；默认仍使用普通 FP32 Adam。`train_player` 每步从 NumPy 样本池取样，
逐字段 `.to(device)`，没有复用 pinned staging、异步预取或 GPU reservoir。
loss 验证、梯度裁剪的非有限值检查和 `float(loss.detach())` 都可能造成 GPU 主机同步。
`iteration_weighted_mse`、`regret_matching` 内还有 Python 张量条件判断。

本地 `torch.compile(..., backend='eager', fullgraph=True)` 探针：
纯 `RegretNetwork.forward` 可捕获；`strategy` 和 loss 报 data-dependent branching。
这验证的是本地 2.14.1 的 Dynamo 捕获行为，不是 2.8 CUDA/Inductor 编译成功或性能。
编译应先限定纯 forward，不能直接承诺整条采样/训练循环 fullgraph 成功。

模型默认每轮重建并重新初始化，优化 compile 时还要测重建、缓存与重复预热成本。
可以探索保持模块对象、原地重新初始化参数并重置 optimizer，但必须保持原初始化与 RNG 语义。
不能为了省时间直接把 `reinitialize=false` 当成等价性能优化。

### 1.4 PPO 已有部分优化，不能重复计算收益

`src/alphanl/transformer/env.py:92` 已按 snapshot id 缓存对手模型，
`:147` 已按对手模型分组批量推理。缓存驻留设备由 rollout_device 决定；
旧配置默认 CPU，改为 GPU 后要重新测小批次开销。
PPO 训练的整份 flat rollout 在 `train.py:726` 附近一次搬到训练设备，
并非每个 minibatch 都从 CPU 重传。

PPO 的 AMP 当前默认 CUDA FP16 + GradScaler；CFR 没有相应 AMP 路径。
PPO 每个 minibatch 将多项指标转 Python float，首 minibatch 还为梯度日志额外求两次梯度。
可以降低日志/梯度探针频率并聚合指标，但这是 PPO 路径的后续优化。

附件所述 50 手 MatchCtx 残差 exploiter 尚未实现，现有 PPO/PFSP 不能直接当成该阶段。

## 2. 附件需要修正的预算与假设

### 2.1 样本池必须乘双方，并包含完整字段

当前 `src/alphanl/cfr/reservoir.py:61` 的每条长度 64 样本为 1313 字节：
token uint16=128、type uint8=64、numeric FP32=1024、padding bool=64、
legal bool=5、regrets FP32=20、iteration int64=8。

当前实现双方各一个独立 reservoir，配置容量是每位玩家容量。

- 每人 2M：双方约 5.252GB RAM，当前默认值合理。
- 每人 5M：双方约 13.13GB RAM。
- 每人 10M：双方约 26.26GB RAM。
- 每人 20M：双方约 52.52GB RAM；RAM checkpoint 再全量复制可能超过 90GB。

附件 schema 漏了当前模型必需的 type_ids、pad_mask；其 13.38GB 是单份 20M 的
不完整紧凑 schema。若保留完整字段，numeric 改 FP16、iteration 改 32 位，每条 797 字节：
每人 20M 的双方存储是 31.88GB（约 29.69GiB），尚无训练激活，24GB GPU 放不下。
若“20M”指双方合计，则约 15.94GB，仍必须实测激活、graph、gather、dtype 转换与临时分配。
pad_mask 可由 token 的 PAD_ID 推导，但应明确实现和验算；type_ids 不能直接删去。

以上 GB 均为十进制字节估算。实际可用 VRAM 用 nvidia-smi 与 torch 的 allocated/reserved
共同确认；预留显存不应固定照搬附件比例。FP16 存储数值特征会改变精度，需独立误差和策略评估。

100M 条历史样本按当前 schema 约 131.3GB，即使 797 字节方案也约 79.7GB，
无法同时轻松塞进附件声称的 60GB 系统 RAM 预算并留出 worker、checkpoint 和 OS 余量。

### 2.2 PCIe 是否第一瓶颈需要测，不能凭小模型参数量判断

当前还原后的输入、目标与权重约 2160 字节/样本，batch=256 约 0.55MB。
很多小传输、NumPy gather、Python 调度和强制同步可能比带宽本身更重要。
没有 DataLoader 并不自动获得 GPU 吞吐；本仓库 CFR 本来就没有使用 DataLoader。

先比较两个方案：

1. CPU RAM 为权威全历史，复用两组小型 pinned batch，单独 copy stream + event 管理生命周期。
2. 训练玩家切换时，把该玩家完整保留池一次搬入 GPU，此阶段在 GPU 均匀抽样。

默认每人 2M 的原始 FP32 compact storage 约 2.626GB，先试一位玩家即可。
这样不必先把双方扩到各 20M，也不必牺牲数值精度。
GPU 版本需显式实现 schema 的 CUDA dtype、索引/转换、插入与保存恢复。
不能用仅含近期样本的“热缓存”偷偷替代全历史均匀采样。

Pinned memory 复用和 copy/compute 重叠需验证；在热路径临时 `.pin_memory()`
可能反而变慢，见 [PyTorch 官方教程](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)。

### 2.3 16K–32K batch、4–8 秒训练没有当前证据

只看参数大小无法计算训练激活。batch=16384、seq=64、d=128 时，单个 BF16 hidden
就约 268MB，4 层还会保留多份 attention/FFN/反向相关张量，部分残差和 normalization 为 FP32。
3.5GB 激活预算没有证明；更大的 1.49M 模型需要重新测。

按附件自己的 800,000 samples/s，batch=16384 × 1000–2000 步需要
每位玩家 20.48–40.96 秒，双方 40.96–81.92 秒，不含遍历和保存，无法同时得到 4–8 秒。

默认训练抽样预算是每人每轮 `512 × 256 = 131072`；附件建议是它的 125–250 倍。
更大 batch 不等于免费降低 Monte Carlo 方差，数据重复率、optimizer 更新数和训练质量都会改变。
例如 `1024 × 128` 虽匹配抽样总数，却只有四分之一 optimizer steps，仍非算法等价配置。

### 2.4 已可使用 SDPA，不等于已经选到 FlashAttention

当前使用 Pre-LN `nn.TransformerEncoderLayer`，没有手写 attention 矩阵。
PyTorch 2.8 的该层 fallback 调用 `need_weights=False` 的 MHA，具备 SDPA 路径；
推理也可能进入另一种 encoder fused fastpath。实际选择取决于 dtype、mask、grad 和运行模式。
先用 profiler 看 `_scaled_dot_product_flash_attention`、高效 SDPA 或 encoder fastpath，
不以“手动改写 attention”作为第一项工作。

来源：[PyTorch 2.8 MHA](https://docs.pytorch.org/docs/2.8/generated/torch.nn.MultiheadAttention.html)、
[2.8 Transformer 源码](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/transformer.py)。

### 2.5 历史策略驻留不是基础训练的首要瓶颈

默认 backbone 592390 参数；regret wrapper 总计 593035 参数。
192/4层/6头/FFN512 backbone 为 1491206，wrapper 总计 1492171。
只看权重，64 个 BF16 默认模型约 76MB；保存双方则约 152MB，不能固定认作 1GB 需求。
FP32 模型与额外元数据另计，vmap 激活还会随模型数 × 输入数扩大。

基础 CFR 遍历只使用双方当前网络，不逐节点读取历史快照。
`SnapshotPolicy.begin_hand` 确实每手读盘加载，可在历史策略评测时加 CPU/GPU 缓存；
PPO 已有按对手缓存。`vmap` 应是后续可选优化，先按快照 id 分组、分块处理。

SD-CFR 应保留所有历史策略的正确轨迹采样；只留最后 64 个属于策略近似。
不能忽略自身 reach，直接平均各 snapshot 在同一 infoset 的行动概率来声称精确平均策略。
依据：[SD-CFR 原论文](https://arxiv.org/abs/1901.07621)。

## 3. 推荐实施顺序

### P0：计时和 checkpoint，先让长跑可判断、可持续

分别记录 root 构造、collect、观测打包/IPC、推理队列等待、GPU 推理、reservoir 插入/采样、
fit、checkpoint，以及包含全部工作在内的总 wall time。
同时记录实际 batch 分布、infosets/s、训练 samples/s、CPU RSS、GPU allocated/reserved 峰值、磁盘增量。
GPU 计时需正确同步或 CUDA event；微基准预热与编译冷启动分开报告。

磁盘模式下 `cfr/checkpoint.py` 每次复制保留池到不可变 generation，旧代不自动清理。
每人 2M 满池时每代约 5.252GB，8 代 + 活动池已约 47.27GB；
若 100 代都满池，checkpoint 数据单独约 525GB。
RAM 模式不同：样本嵌入覆盖写入的 train.pt，不会每代在 checkpoint_data 再留一份。
但存盘时有 RAM 全量副本、磁盘旧文件与临时文件峰值。

需要新增安全保留策略：新 checkpoint 成功发布且验证引用后，只清理无人引用的旧 generation。
策略历史继续保留；同文件系统当前 hardlink 已有收益，不应盲目删除共享归档。
按实际增长给磁盘留余量；一开始使用 RAM 池和本地数据盘，避免系统盘存大样本。

### P1：减少复制和单进程尾部损失

- 已完成：树结束后立即补入待办根，减少整波等待；仍按玩家批量推理。
- 重用 tokenizer/观测 staging，避免每个 infoset 大量小 Tensor 分配。
- 先 profile deepcopy 的子对象，再实现经过验证的轻量 clone、不可变共享或 apply/undo。
  固定完整机会样本的牌堆可共享不可变数据，分支可变筹码、动作和状态必须独立。
- 若 Python/PokerKit 经上述优化仍明显占主导，再考虑 C++/Rust 状态机。
  新引擎用 PokerKit 做差分 oracle；仓库已经接入 phevaluator，不能再把同一查表加速算一遍。

### P2：CPU 多进程 + 单 GPU 服务

用 spawn，worker 仅进行 CPU 状态推进、观察与 regret 回溯；导入 torch 本身不等于创建 CUDA context，
关键是 worker 不分配 CUDA Tensor、不初始化 GPU。只有一个进程持有 GPU 网络与训练器。

worker 起步扫描 8/12/16/20/22；每 worker 并发树 16/32/64，
`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`、torch threads=1。
25 vCPU 可能有 SMT/配额限制，不应预设 `N-3` 最优。

使用 Python `multiprocessing.shared_memory` 的连续槽传输入和小输出，队列只传元数据；
附件的 `torch.multiprocessing.shared_memory` 不是正确模块路径。
第一版用有明确所有权的槽和同步协议，不必先写 lock-free ring。

GPU 服务按 `(player, policy_version, shape_bucket)` 聚批，目标 batch 从 256/512 起测，
到达容量或约 0.5–2ms 超时即处理；阈值需实测，避免为了凑 2048 阻塞所有树。
总在途树数是上限，按玩家拆组、树深度与 CPU 生成速度都会降低真实 batch。
128 个根的采样预算也不足以同时跑满 22×64 棵树，增加根数会改变每轮采样工作量。

请求带 worker/tree/request id 和策略版本；合并端是每位玩家 reservoir 的唯一写入者，
不能把等容量 worker 局部 reservoir 随意拼起来当全局均匀样本。
批量 reservoir 替换若产生重复槽，要明确顺序/优先级并保持均匀保留语义。

维持当前“同轮双方策略冻结、双方采样完成后才训练”的屏障。
每树/worker 独立可恢复 RNG；调度变化后验收统计正确性，不强求旧单进程随机调用顺序逐位一致。

### P3：GPU 训练加速

- BF16 autocast；参数主副本、regret head、targets、加权 loss 保留 FP32。
  不直接整模型 `.bfloat16()`，现有 head 强制 FP32 输入，需要匹配权重精度。
- 优先同算法的 `Adam(fused=True)`；直接换带 weight_decay 的 AdamW 会改变目标。
  来源：[PyTorch 2.8 Adam](https://docs.pytorch.org/docs/2.8/generated/torch.optim.Adam.html)。
- CUDA 上启用适合 2.8 的 TF32 配置并做精度对照；BF16 与 FP16 路径分开验证。
- 先 compile 纯 forward，输入 batch 按固定尺寸桶；清理 tensor→Python 条件、float 指标同步。
  验证保留在 CPU 数据边界或聚合检查，不静默吞掉非法 mask/NaN。
- `reduce-overhead` 的 CUDA graph 有适用条件，也会增加内存，不能预先宣称一定捕获或加速。
  来源：[PyTorch 2.8 compile](https://docs.pytorch.org/docs/2.8/generated/torch.compile.html)。
- 统计真实序列长度；可尝试裁去纯右 padding 并按 16/32/64 桶处理，
  保持 token 顺序、position、mask 和最大截断语义，不能直接缩短 max_seq_len 丢历史。

### P4：GPU reservoir 与容量扩展

先每人 2M RAM，比较复用 pinned 双缓冲和当前玩家完整 GPU 池；
训练 batch 扫描 512/1024/2048/4096，8192 以上需峰值显存与质量证据。
记录转入 GPU 的整池成本，不能只报 GPU 抽样循环吞吐。

只有观测到样本容量/覆盖限制或采样搬运主导，再扩到每人 5M/10M。
CPU 主池、GPU 副本、保存恢复共用明确格式；压缩与大 batch 分开做实验。

## 4. 服务器验证方式

新增 `scripts/benchmark_cfr.py` 测量现有训练器。它使用全新空目录和 RAM 样本池，
不会恢复旧运行或打开配置里已有的 memmap。第一轮含冷启动，只在末尾计时保存一次 checkpoint。
fit 时间包含重初始化、CPU 取样/H2D、forward/backward/optimizer，不冒称纯 GPU kernel 吞吐。

在服务器仓库根目录，先用现有镜像依赖，确认设备：

```bash
nvidia-smi
lscpu
df -h
python -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name())'
```

先检查配置解析（本地无 CUDA 时也可运行）：

```bash
PYTHONPATH=src python scripts/train_cfr.py --config configs/cfr/hunl_4090.json --dry-run
```

然后运行（out-dir 必须是新的空目录）：

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python scripts/benchmark_cfr.py --config configs/cfr/hunl_4090.json --device cuda \
  --iterations 2 --traversals 16 --tree-batch 32 \
  --train-steps 32 --batch-size 256 \
  --out-dir runs/benchmark_4090_baseline
```

结果在 `benchmark.json`，训练器另写 `performance.csv`：分别有双方 collect/fit 时间、总时间、checkpoint 时间、
实际平均推理 batch、两个不同阶段的 samples/s、GPU allocated/reserved 峰值。
这是新池短测，不能代表满池随机访问；长跑时还需对成熟 pool 单独测量取样与保存。
脚本没有 CPU RSS 项，Linux 可配合 `/usr/bin/time -v`、pidstat 与 nvidia-smi dmon。

`configs/cfr/hunl_4090.json` 显式请求 BF16、fused Adam 和 TF32；这些选项仅在 CUDA 生效。
CPU 测试可用 `--device cpu` 覆盖；配置不支持多进程、compile 或 GPU reservoir，这些仍需后续实现。

验收次序：原有规则/隐藏信息/精确 signed regret 测试 → 小游戏 sanity → GPU 数值/恢复测试 →
固定样本量吞吐 → 固定墙钟时间的多种子策略评估。对战成绩应报告置信区间；
标准 HUNL、旧 Botzone 与不同筹码深度不能直接混算。

本地回归测试结果将在 PR 提交前更新；本地 CPU 测试不替代服务器 CUDA 数值、吞吐和恢复验收。
