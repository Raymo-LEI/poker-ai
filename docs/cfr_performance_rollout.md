# CFR P0–P3 实施与 4090 验证

日期：2026-10-27。实现基础：`0c475ba`。4090 对比由用户在服务器执行；本次开发环境为 macOS ARM、CPU、PyTorch 2.14.1，未执行 CUDA 测试。

## 已实现

- **P0**：新 checkpoint 的外部引用和数据文件尺寸先验证，再原子发布；数据和目录同步后才清理无人引用的旧 generation。每个 checkpoint 有带文件指纹的引用 sidecar；同目录旧 `.pt`、`.pth`、`.ckpt` 和有 sidecar 的其他文件名均参与保留判断。无法读取任一 checkpoint 时停止清理。保存使用目录级文件锁；写入失败清理未发布代，发布后同步失败保留新文件引用的数据。
- **P0 空间预算**：写入前计算保留池、新模型/优化器和跨文件系统策略副本空间，默认额外留 1GiB。旧 checkpoint 在新文件发布前一直有效。hardlink 失败退回复制时再检查剩余空间。历史策略全部保留；手工保留的独立 checkpoint 也会继续占空间，空间不足时抛错停止保存。
- **P1**：`HunlTable.clone` 只复制 PokerKit 可变容器及动作列表，共享 frozen card、street、operation、规则值。预采样手牌期间 RNG 只读；reset/set_rng_state 时分离 RNG。HUNL 观察改为整段 NumPy 打包，避免逐 token 小 Tensor，DFS 各帧仍拥有独立观察存储。复用 tokenizer，并按需建立根。
- **流式调度**：原分支已经支持树结束即补根，本次保持其行为并继续验证。只要还有待办根就补齐并发树；最后几棵树已经没有新根可补时，batch=1 仍可能出现。附件的 16 根、tree-batch=32 短测本来就无法持续填满 32 棵树。
- **P2**：`spawn` CPU worker + 主进程唯一推理/训练服务。输入、概率和 regret 走每 worker 独占共享槽，队列/pipe 传元数据与确认。请求包含 worker/request/player/policy version，固定序列形状；按玩家聚批，到容量或超时即处理。主进程唯一写入全局 Algorithm-R reservoir；双方策略冻结到采样屏障结束。worker RNG 在 checkpoint 保存、恢复，worker 异常/死亡/无进展会失败并释放资源。
- **P3**：沿用已经存在的 BF16 autocast、FP32 参数/head/target/loss、fused Adam 和 TF32 开关，新增训练期完整 GPU 样本池。CPU RAM/memmap 仍为权威数据；每次只上传当前玩家完整保留池，GPU 有放回均匀抽样，fit 返回时释放副本。GPU sampler 使用独立、可恢复 RNG。

GPU token 存储使用 int32，避免 uint16 eager 索引支持问题；其他字段不压缩。长度 64 时 CPU 每条 1313 字节，GPU 每条 1441 字节：单玩家 2M 为 2.882GB，双方 CPU 为 5.252GB。GPU 池按已保留行数分配，不预占空槽；默认额外要求 2GiB 空闲余量。`pool_upload_s` 已包含在 `fit_s` 中；余量检查不能代替真实激活峰值测试。

PyTorch API 依据：[2.8 Tensor dtype](https://docs.pytorch.org/docs/2.8/tensor_attributes.html)、[2.8 fused Adam](https://docs.pytorch.org/docs/2.8/generated/torch.optim.Adam.html)。没有改成 AdamW，也没有增加训练步数或 batch 大小来制造吞吐收益。

## 默认值与恢复

原配置仍默认单进程、CPU 样本池。新 `configs/cfr/hunl_4090_parallel.json` 显式配置 8 worker、每 worker 32 棵树、256 行推理上限、1ms 聚批等待、BF16/fused Adam，以及训练期单玩家 GPU 池。8 worker 是待测试起点。

`train_cfr.py` 和 `benchmark_cfr.py` 支持 `--workers`、`--tree-batch`、`--inference-batch`、`--inference-wait-ms`、`--train-buffer-device`；benchmark 也支持 AMP/fused/TF32 覆盖及 `--baseline`。

从旧 checkpoint 可以开启多进程或 GPU 池。更换 worker 数/执行路径时会显式提示创建确定性新 RNG 流；同样的执行配置会恢复已有流。多进程样本到达顺序不固定，Algorithm-R 的均匀保留语义不变，但不保证两次运行逐位相同。单进程 CPU 的原精确续跑测试继续通过。旧 CSV 表头会在新列出现时升级，保留原有记录。

Checkpoint 文件应与 `checkpoint_data` 放在同一目录；备份到其他目录时复制整套目录。GC 只识别同一目录内的 checkpoint，不能追踪单独复制到其他地方、仍试图引用原目录的文件。不要把同一 checkpoint_data 目录作为活跃 memmap。引用 sidecar 可重建；旧 checkpoint 首次被发现时需要读取一次，RAM 嵌入池可能较大。

## 本地实测证据

原始输出保存在 `docs/benchmarks/cfr_20261027/`，附用户提供的 4090 基线 JSON/CSV（CSV 仅规范换行）；没有将 macOS 结果与 4090 计算加速比。

相同 seed、每玩家 16 根、tree-batch=32、每玩家 32 个训练 step、batch=256，原实现与最终 P0/P1 各跑两轮。两轮节点数、有效 regret 数和推理 batch 保持一致：

- 第一轮采样总时间：2.8384s → 1.4672s，**1.93×**；完整迭代 19.7653s → 18.2163s，**1.09×**。
- 第二轮采样总时间：0.5474s → 0.3063s，**1.79×**；完整迭代 17.4319s → 17.0126s，**1.02×**。
- 第一轮含 evaluator/运行时冷启动；这是单次短测，不能外推满池、长跑或 4090 收益。
- P2 的早期 2-worker/每 worker 4 树/16 根 CPU smoke 完成，短任务没有提速，平均推理 batch 反而偏小。此输出用于功能验证；它使用独立 worker RNG，树大小也不同，不能把耗时差直接称作同工作量加速比。后续 tokenizer 打包优化后的最终路径另做了小规模 smoke。

本地回归：86 passed、3 skipped（CUDA 不可用）。覆盖跨街轻量 clone 对照 deepcopy、父子/RNG 隔离、观察打包等价、解析 signed regret、Kuhn 收敛、Algorithm-R 均匀性、真实 spawn、共享槽多行传输/释放、恢复迁移、checkpoint 清理/失败保护/复制退回和完整池 schema。CUDA 测试已加入但未在本机执行。

## 服务器运行顺序

所有 benchmark 输出目录必须为空。环境使用现有 PyTorch 2.8.0/CUDA 12.8，不需更换 torch wheel。

先跑 GPU 样本池与 BF16/fused 恢复回归：

```bash
PYTHONPATH=src python -m pytest tests/test_cfr_device_pool.py tests/test_cfr_parallel.py tests/test_cfr_trainer.py -q
```

第一步：P0/P1 与附件完全相同的小预算；已有 4090 配置里的 BF16/fused 参数保持启用，不把其收益重新记到这次修改。

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python scripts/benchmark_cfr.py --config configs/cfr/hunl_4090.json --device cuda \
  --workers 0 --train-buffer-device cpu --iterations 2 --traversals 16 \
  --tree-batch 32 --train-steps 32 --batch-size 256 \
  --baseline docs/benchmarks/cfr_20261027/4090_baseline.json \
  --out-dir runs/benchmark_4090_p01
```

第二步：先建立 512 根的单进程新基线，再对比 8 worker。保持相同根数与训练抽样预算，不用增加数据量假装加速。

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python scripts/benchmark_cfr.py --config configs/cfr/hunl_4090_parallel.json --device cuda \
  --workers 0 --train-buffer-device cpu --iterations 3 --traversals 512 \
  --tree-batch 32 --train-steps 32 --batch-size 256 \
  --out-dir runs/benchmark_4090_serial512

PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python scripts/benchmark_cfr.py --config configs/cfr/hunl_4090_parallel.json --device cuda \
  --workers 8 --train-buffer-device cpu --iterations 3 --traversals 512 \
  --tree-batch 32 --inference-batch 256 --inference-wait-ms 1 \
  --train-steps 32 --batch-size 256 \
  --baseline runs/benchmark_4090_serial512/benchmark.json \
  --out-dir runs/benchmark_4090_p2
```

第三步：同样 worker/预算开启完整 GPU 池，fit 包含整池上传。可依次扫描 8/12/16 worker、16/32/64 tree-batch；每次保留独立输出。推理容量需至少覆盖单 worker 的 tree-batch。

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python scripts/benchmark_cfr.py --config configs/cfr/hunl_4090_parallel.json --device cuda \
  --workers 8 --train-buffer-device cuda --iterations 3 --traversals 512 \
  --tree-batch 32 --inference-batch 256 --inference-wait-ms 1 \
  --train-steps 32 --batch-size 256 \
  --baseline runs/benchmark_4090_p2/benchmark.json \
  --out-dir runs/benchmark_4090_p3
```

报告包含实际 batch 分桶/单行比例所需计数、主进程队列等待、观察/根构造、打包、推理、reservoir 插入、fit/整池上传、checkpoint、RSS 与 GPU allocated/reserved 峰值。worker CPU 时间及 RSS 是各进程值的和，会重叠，不能加到总 wall time。`performance.csv` 是每轮阶段计时，`checkpoint_performance.jsonl` 单独记录保存时间与磁盘余量。总 benchmark 时间还包含 worker 退出。

`--baseline` 会检查环境和根/训练预算，不匹配时报告 `comparable=false`，不输出加速比。环境字段相同仍需人工确认是同一服务器、相同负载；开启多进程后查看 regret 样本吞吐与节点计数，不能只看 wall time。所有新池短测都不代表成熟 2M 池的上传、抽样和保存成本；满池及固定算时策略质量仍需后续服务器实测。
