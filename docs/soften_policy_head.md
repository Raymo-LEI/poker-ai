# soften_policy_head.py：BC 策略软化工具

把 BC checkpoint 的策略头最后一层 `policy_head`（plain `nn.Linear`，键 `policy_head.weight` / `policy_head.bias`）的 weight 和 bias 同时除以 T，等价于 logits/T。价值头和其余所有张量逐位不变，合法动作的 argmax 不变。输出仍是 PPO 格式 ckpt，另加 `soften_metadata`（T、目标熵、val 路径、前后熵、源 ckpt）。

## 放置

把 `soften_policy_head.py` 放到 worktree `poker-ai-m10` 的 `scripts/`，测试 `test_soften_policy_head.py` 放到 `tests/`（可选）。脚本会自己把 `src` 加进 `sys.path`。

## 用法（Windows PowerShell，在 worktree 根目录）

按目标熵自动找 T（推荐）：

```powershell
$env:PYTHONPATH="src"
python scripts/soften_policy_head.py --ckpt runs/bc_rulebot_train/bc_v2.pt --val-data runs/bc_rulebot/merged_600k --target-entropy 0.3 --out runs/bc_rulebot_train/bc_v2_soft.pt
```

直接指定 T：

```powershell
python scripts/soften_policy_head.py --ckpt runs/bc_rulebot_train/bc_v2.pt --temperature 3.0 --out runs/bc_rulebot_train/bc_v2_soft.pt
```

可选参数：
- `--val-all`：用 `--val-data` 的全部行，不再取 10% 留出局（给单独的 val 目录用）。
- `--allow-sharpen`：目标熵低于当前熵时允许 T<1（默认拒绝并报错）。
- `--max-temperature`：二分上界，默认 1000。
- `--force`：覆盖已有 `--out`。

注意：`runs/bc_rulebot_train/bc_v2.pt` 和 `runs/bc_rulebot/merged_600k` 是我对路径的猜测，请换成笔记本上的真实路径。真 ckpt 在笔记本上，不在我这台机器上，工具只在 fixture 小模型上测过。

## 输出内容

打印 val 行数；软化前后的平均熵（nats 和 bits，全部行 / 仅 ≥2 个合法动作的行）；每个动作的平均概率；每个动作的 argmax 频率（前后）；选中的 T；argmax 是否有变化（有变化会打印大写 WARNING）；价值头最大绝对差（应为 0）。

val 行的取法与 `train_bc_rulebot.py` 相同：`split_game_mask(game_id, metadata.seed, 0.1)`。合并数据集时 seed 是第一个目录的 seed（`merge_bc_datasets.py` 的约定），所以和 BC 训练留出的是同一批局。若 ckpt 里记录的 `val_game_ids` 与这里算出的不一致，会打印 WARNING（说明测熵的行里混有 BC 训练局）。

加载 60 万行数据约占 1.5 GB 内存，只前向 val 的约 6 万行，CPU 上几十秒。

## 接 PPO（用 --init-weights，不要 --resume）

```powershell
python scripts/train_transformer.py --config configs/transformer/botzone_laptop_8gb_bc_init.json --init-weights runs/bc_rulebot_train/bc_v2_soft.pt --allow-cross-backend-init
```

软化后的 ckpt 里保留了 BC 的 Adam 状态，仅为格式兼容。用 `--resume` 会把这份状态连同缩放过的策略头一起载入，不要用。用 `train_autoresume.ps1` 时，没有 PPO ckpt 的首次启动传这个软化文件作 `--init-weights`。

## 校验与测试

- 工具内置校验：写盘后重新加载，核对除策略头外所有张量逐位相同，策略头 == 原值 / T，顶层键无增减。
- `python -m pytest -q tests/test_soften_policy_head.py`（15 项）。
