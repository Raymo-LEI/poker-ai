# 第 3 步训练：启动、AMP、checkpoint 与续训

只写了代码，还没有正式训练过。CPU 上单元测试和 2～3 步冒烟已通过。

## 环境
```
python -m venv .venv && . .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements-transformer.txt
# GPU：换成和驱动匹配的 CUDA 版 torch，例如
pip install torch --index-url https://download.pytorch.org/whl/cu121
pytest                                                # CPU 即可，含 pokerkit 单测
```

## AMP（混合精度）
- 3070 笔记本配置（`configs/transformer/laptop_8gb.json`）和 Colab 配置都默认 `train.amp: true`。命令行可用 `--amp` / `--no-amp` 覆盖。
- 只在 CUDA 上生效，在 CPU 上会自动关闭，不报错。用的是 `torch.autocast` 加 `GradScaler`，GradScaler 的状态会存进 checkpoint。
- 策略头、价值头、非法动作 mask 和 softmax 强制用 fp32。原因是 mask 值 -1e9 在 fp16 下会变成 -inf，导致 NaN。encoder 部分仍走 autocast。
- 如果 loss 出现 NaN 或 inf，先加 `--no-amp` 对照。

## Checkpoint
- 目录是 `train.checkpoint_dir`：本地配置默认 `runs/ckpt`，Colab 配置默认 `/content/drive/MyDrive/poker-ai/ckpt`。可用 `--checkpoint-dir` 覆盖。
- 每 `train.checkpoint_every` 次更新存一份 `update_XXXXXXX.pt`（可用 `--checkpoint-every` 覆盖），跑完时最后一步也会存。只保留最近 `train.keep_last`（默认 3）份。
- 内容：模型、优化器、GradScaler、步数、配置、对手池（含它的 RNG），以及 Python random、NumPy、torch CPU/CUDA 的 RNG 状态。另外还有每张牌桌的完整状态：牌桌 RNG、打到一半的手，和对手 actor 的 RNG。rlcard 牌桌 pickle 的是 rlcard 环境（含发牌 RNG）；pokerkit 牌桌 pickle 的是 PokerKit `State`、已发生的动作，以及这张牌桌自己的发牌 RNG。blob 里有 `backend`，不能把一份 checkpoint 载进另一个后端。旧的 rlcard blob（只有 `rl` 和 `fields`）仍能载入 rlcard 环境。所以同一后端续训后，自博弈 rollout 和不中断时一模一样。CPU 上逐位一致已经测过；GPU 不保证逐位一致。
- 写入是原子的：先写 `.tmp`，fsync，再 rename。中途断电只会留下 `.tmp`，不会损坏已有的 checkpoint。
- `--resume`（不带参数）会挑 `checkpoint_dir` 里最新的一份；目录为空时从头开始，所以可以一直带着它启动。也可以 `--resume path/to/update_XXXXXXX.pt` 指定某一份。
- `--updates` 是总步数，不是“再跑多少步”。例如之前跑到 100，想跑到 200，就写 `--updates 200 --resume`。

## 3070 笔记本（Ryzen 9 5900HX / RTX 3070 Laptop 8GB / 32GB）
```
export PYTHONPATH=src
python scripts/train_transformer.py --dry-run                       # 打印模型规模
python scripts/train_transformer.py --config configs/transformer/laptop_8gb.json --updates 2 --checkpoint-every 1   # 冒烟
python scripts/train_transformer.py --config configs/transformer/laptop_8gb.json --resume                           # 正式跑 / 续训
```
- 想无人值守、断了自动重来：Linux 用 `until python scripts/train_transformer.py ... --resume; do sleep 60; done`。Windows PowerShell 用 `while ($true) { python scripts\train_transformer.py ... --resume; if ($LASTEXITCODE -eq 0) {break}; Start-Sleep 60 }`。
- 插电时别让它睡眠：`powercfg /change standby-timeout-ac 0`。
- 显存不够就调小 `ppo.minibatch_size`，或改用 `--no-amp` 做对照。rollout 默认在 CPU 上跑（`rollout_device: cpu`）。

## Colab（L4）
```python
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/Raymo-LEI/poker-ai && cd poker-ai && git checkout <PR #4 分支>
%cd /content/poker-ai
!pip install -q -r requirements-transformer.txt
!mkdir -p /content/drive/MyDrive/poker-ai/ckpt
!PYTHONPATH=src python scripts/train_transformer.py --config configs/transformer/colab_l4.json --resume
```
- 私密仓库要先认证才能 clone，比如用 GitHub token，或者把代码 zip 上传到 Drive。
- checkpoint 写到 Drive，会话断了就重新执行上面几格，`--resume` 会接上最新一份。
- `train.out_dir`（metrics.csv、tb/）默认在 `/content` 下，会话结束就没了。要保留就加 `--out-dir /content/drive/MyDrive/poker-ai/runs`。

## 落地待办

1. **GPU 验证**：在 RTX 3070 笔记本上实测 AMP 训练路径（`train.amp: true` / `--amp`）和 `--resume` 续训；确认 loss 有限、checkpoint 能写读、显存占用可接受。CPU 单测不能代替 CUDA 上的 GradScaler 行为。
2. **Colab 拉代码**：由仓库主在 GitHub 创建 **fine-grained、只读** 的 Personal Access Token，权限仅覆盖本仓库（`poker-ai`），用于 Colab 里 `git clone` 私有仓；不要把 token 写进仓库或 checkpoint。
3. **对手池的体积**：checkpoint 已经能分别序列化 rlcard 和 pokerkit 牌桌（见上文）。对手池仍存完整 `state_dict` 副本，后续可改为按 id 引用磁盘上的权重以省内存和 checkpoint 体积。

## 3070 / Colab AMP 冒烟（T-A）

只做冒烟，不是正式训练。用的是 main 上原有的训练循环（`run_training`），`scripts/smoke_metrics.py` 只在外面包一层计时和显存统计，不改训练代码。输出目录是全新的，不碰正式 run 和 `runs/ckpt`。

- **笔记本（Git Bash / WSL / Linux）**：`bash scripts/smoke_3070.sh`。默认用 `laptop_8gb.json`、`--amp`，跑 300 个 update，每 50 个打印一次：steps/s、rollout 耗时（CPU）、update 耗时（GPU）、loss、`torch.cuda.max_memory_allocated`（GB）、GradScaler 当前 scale。快速试跑：`STEPS=10 bash scripts/smoke_3070.sh`。
- **PowerShell**：`$env:PYTHONPATH="src"; python scripts\smoke_metrics.py --config configs\transformer\laptop_8gb.json --steps 300 --log-every 50 --amp --out-dir runs\smoke_3070`
- **预计耗时**：笔记本上每个 update 约 7–9 秒，主要花在 CPU rollout 上，所以 300 步大约 35–45 分钟。
- **产出**：`runs/smoke_3070/` 下的 `smoke_summary.md`（直接贴群里的 markdown 表）、`smoke_summary.json/csv`（按 50 步一个窗口）、`smoke_steps.csv`（逐步）、`metrics.csv`、`smoke.log`、`ckpt/`。
- **验收**：没有 NaN/inf loss、没有 OOM、峰值显存 < 7GB，就是 PASS（退出码 0），否则 FAIL（退出码 1）。默认不跑周期性 eval（要跑加 `--with-eval`）。显存吃紧可以加 `--minibatch 128`、`--num-envs`、`--rollout-steps` 调小 batch。
- **Colab**：打开 `scripts/colab_smoke.ipynb`，选 L4，在 Colab 的 Secrets 里加 `GITHUB_TOKEN`（只读、只给本仓库）。它会挂载 Drive、用这个 token clone 私密仓库（不打印、不落盘，clone 完就从 remote URL 里去掉），然后用 `colab_l4.json --amp` 跑同样的 300 步，结果写到 `MyDrive/poker-ai/smoke_colab`，最后显示汇总表。
