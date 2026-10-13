# 控制组卷积网与 c_1048 行为克隆

只写了代码和 CPU 冒烟，没有正式训练。目标是：PPO、5 维策略头、非法动作掩码、训练脚本都不变，实验只改 `encoder`。

| `encoder` | 网络 | 观测 |
|-----------|------|------|
| `transformer` | 下注历史 token 上的小 Transformer（第 3 步） | `token_ids` / `type_ids` / `numeric` / `pad_mask` / `legal_mask` |
| `alphanl_conv` | PyTorch 移植的 TF 1.15 `NlHoldemNet` | `card_info (4,13,6)`、`action_info (4,5,25)`、`extra_info (2,)`、`legal_mask (5,)` |

`env.backend` 可以是 `rlcard` 或 `pokerkit`（见 `docs/step3_transformer.md`）。两种后端都走同一套 PPO 和掩码。Transformer 从 `HandSnapshot` 做 token；控制组在 pokerkit 上把已发生的抽象动作重建成与 rlcard wrapper 相同的历史张量（`PokerkitTable.wrapper_history` / `rlcard_raw_obs`）。**`c_1048` 行为克隆的数据导出仍只在 Python 3.7 + rlcard 的 TF 环境里跑**；灌权重或 BC 之后，PPO 可以在 pokerkit 上继续训控制组。

两份笔记本 / Colab 配置：

- `configs/transformer/laptop_8gb.json`、`configs/transformer/colab_l4.json`：Transformer
- `configs/transformer/laptop_8gb_conv.json`、`configs/transformer/colab_l4_conv.json`：控制组

控制组大约 **424,710** 个参数，宽度是上游写死的 16 / 32 / 64，不看 `model.d_model`。`model.*` 留在 JSON 里只是为了和另一份配置同一套字段；换编码器时不要同时改 PPO。Colab 那份只把 batch 加大，网络本身仍是这一个小卷积网，8GB 笔记本也跑得下那份网络，但 32×256 的 rollout 会先卡在 rlcard 上。

## 和 TF 1.15 原网络对齐了什么

`src/alphanl/transformer/conv_net.py` 按 `agi/nl_holdem_net.py` 移植：

- 牌面 `(4, 13, 6)`、动作历史 `(4, 5, 25)`，通道在最后一维，和 wrapper 一致。
- 三个 stage：`[16, stride 1, 1 block]`、`[32, stride 2, 2 blocks]`、`[64, stride 2, 2 blocks]`。
- 每个 block：3×3（带 stride）→ ReLU → 3×3（stride 1）→ 可选 1×1 shortcut →（stage 下标 > 1 才相加）→ ReLU。
- stride 用在该 stage 的**每一个** block 上，不只是第一个。这是原代码的循环，不是漏写。
- Flatten 用 NHWC 顺序，这样 `extra_fc` / `fc_1` 的核才能和 TF 对上。牌面和历史各自 flatten 成 64 维。
- 后面是 256 → 128 → 64，策略头 `conv_fuse`（5 维），价值头 `value_out`（1 维）。

观测编码在 `src/alphanl/nl_obs.py`，按 `NlHoldemEnvWrapper._get_observation`：花色 `CDHS`、点数 `23456789TJQKA`、每条街只保留前 6 个动作、最后一维整片写成当前座位号。这个文件只有 NumPy，给 Python 3.7 的导出脚本用，不进 PyTorch。

## 对不齐、或者故意没搬进 PPO 的地方

1. **`extra_fc` 不读 `extra_info`。** 上游把这个 16 维 ReLU 接在牌面 flatten 上，2 维筹码向量虽然是 Keras 输入，前向不用。移植保持这个接法，否则 `c_1048` 的核形状对不上。batch 里仍然要有 `extra_info`（uint8 截断后的双方筹码，和 wrapper 一致）。
2. **非法动作掩码的填充值。** TF 是 `logits + max(log(mask), float32.min)`。共享 PPO 在 softmax 之前把非法 logit 写成 `-1e9`（`masked_log_probs`）。两边合法动作的概率在 raw logit 一致时差远小于 `1e-4`；填进去的那个常数本身不同。控制组和 Transformer 用的是同一份掩码，所以互相比较时这一项不是变量。
3. **卷积 L2。** TF 的卷积核带 `l2(1e-4)`。共享 PPO 损失不加这项。加上的话，控制组和 Transformer 的目标函数就不只差编码器了。加载权重做推理不受影响。
4. **头的宽度跟编码器绑在一起。** 控制组保留 256-128-64 和 `conv_fuse` / `value_out`，因为 `c_1048` 的策略就长在这上面。Transformer 仍是 GELU trunk 加自己的线性头。两边都输出 5 个 logit 和一个 value，掩码和 PPO 相同。不能把 Transformer 的头套到 `c_1048` 上。
5. **从零初始化。** TF 是 He normal 加 `normc(0.01)`。这里是 Kaiming normal 加 orthogonal gain 0.01。加载 `c_1048` 之后会被覆盖。
6. **BatchNorm。** 上游两行都注释掉了，没有移植。
7. **Value 的尺度。** `c_1048` 的 value 是 IMPALA 在筹码回报上拟合的。这边 PPO 的回报是 bb（大盲 = 2 筹码）。策略分布可以对齐；value 不能当成同一尺度的热启动，PPO 会把它重新拉向 bb 回报。
8. **TF `padding="same"` 且 stride 为 2。** PyTorch 的 `padding="same"` 不允许 stride ≠ 1，而且偶数长度上 TF 的 pad 不对称。`tf_same_pad` 按 TF 的公式补。

`c_1048.pkl` **可以**直接灌进控制组（卷积、shortcut、`extra_fc`、三层全连接、策略头、价值头）。键名按路径末尾的 `层名/kernel|bias` 匹配，卷积核从 HWIO 转成 PyTorch 的 OIHW，全连接从 `(in, out)` 转成 `(out, in)`。Transformer **不能**装这份权重。

## 行为克隆

两个环境不要装在一起。

1. **Python 3.7 + `requirements-alphanl.txt`** 里跑 `scripts/dump_c1048_bc.py`。它用 TF 1.15 的 `c_1048` 自己跟自己打（或打随机 / 其他 pkl），把每次决策写成 npz：卷积观测、合法掩码、教师概率、采样动作，以及一份 JSON（手牌、历史、盲注座位），好让 PyTorch 这边再做成 Transformer 的 token。
2. **Python 3.10+ + `requirements-transformer.txt`** 里跑 `scripts/train_bc.py`。损失是对教师概率的 KL，或对动作的交叉熵。非法动作会从教师分布里去掉再归一化，学生的 softmax 用和 PPO 相同的掩码。某一行教师把质量全放在非法动作上时，退回成合法动作上的均匀分布。

BC checkpoint（`update = -1`）可以用两种方式接进 PPO：

- `--init-weights`：只加载模型，优化器、牌桌、步数都从头开始（推荐）。
- `--resume 指向该文件`：因为 `update` 是 -1，PPO 的第一步编号仍是 0，并会带上 BC 的 Adam 状态。牌桌状态不在这份文件里，所以自博弈牌桌是新的。

## 笔记本（RTX 3070 Laptop 8GB）

导出必须用 Step 1 的环境。权重在子模块里：`git submodule update --init --recursive`。

```bash
# --- 环境 A：Python 3.7，不要装 PyTorch 训练那份依赖 ---
conda activate alphanl
cd /path/to/poker-ai
export PYTHONPATH=src
python scripts/dump_c1048_bc.py \
  --weights third_party/AlphaNLHoldem/weights/c_1048.pkl \
  --hands 200 --opponent self --seed 0 \
  --out data/c1048_self.npz

# 对随机对手，只记录教师座位（每手换边）
python scripts/dump_c1048_bc.py --hands 200 --opponent random --out data/c1048_vs_random.npz

# 对其他 pkl。每个对手一份 TF 图，池子别一下放太多。
python scripts/dump_c1048_bc.py --hands 100 --opponent pool \
  --pool third_party/AlphaNLHoldem/weights/c_1048.pkl \
  --out data/c1048_vs_pool.npz
```

```bash
# --- 环境 B：Python 3.10+ ---
pip install -r requirements-transformer.txt
# GPU 换成和驱动匹配的 CUDA 版 torch

export PYTHONPATH=src
export ALPHANL_PYTHON=/path/to/alphanl/bin/python   # 仅当下面用 --dump 时需要

# 控制组：先把 c_1048 灌进去，再 KL 微调，再 PPO
python scripts/train_bc.py \
  --config configs/transformer/laptop_8gb_conv.json \
  --data data/c1048_self.npz \
  --init-c1048 third_party/AlphaNLHoldem/weights/c_1048.pkl \
  --loss kl --steps 2000 --batch-size 256 \
  --out runs/bc_conv/bc.pt

python scripts/train_transformer.py \
  --config configs/transformer/laptop_8gb_conv.json \
  --init-weights runs/bc_conv/bc.pt \
  --updates 2 --checkpoint-every 1          # 冒烟；正式跑去掉 --updates

# Transformer：不能 --init-c1048，只能 BC
python scripts/train_bc.py \
  --config configs/transformer/laptop_8gb.json \
  --data data/c1048_self.npz \
  --loss kl --steps 2000 \
  --out runs/bc_transformer/bc.pt

python scripts/train_transformer.py \
  --config configs/transformer/laptop_8gb.json \
  --init-weights runs/bc_transformer/bc.pt
```

`--dump` 会在 npz 不存在时用 `ALPHANL_PYTHON`（或 `python3.7`）另起进程导出，不会在当前进程里 `import tensorflow`。

显存：控制组 0.42M 参数，Adam 状态大约几 MB。和 Transformer 一样，rollout 在 CPU，PPO 的 minibatch 才上 GPU。3070 8GB 用 `laptop_8gb_conv.json` 即可。不够就减小 `ppo.minibatch_size`。

## 和 TF 模型对一下动作分布

目标：同一条观测上，两边概率的最大绝对差 < `1e-4`。

两个环境不能进同一个进程，所以对照分成两步。本仓库的云端环境没有 TF 1.15，**第一步没有跑过，对照结果未验证**。缺权重或 TF 起不来时脚本打印 `UNVERIFIED` 并以 0 退出；PyTorch 这边真的比过且超过阈值则以 1 退出。

```bash
# 环境 A（Python 3.7）
conda activate alphanl
export PYTHONPATH=src
python scripts/compare_c1048_probs.py dump-tf \
  --weights third_party/AlphaNLHoldem/weights/c_1048.pkl \
  --hands 20 --out data/c1048_probs.npz

# 环境 B（Python 3.10+）
export PYTHONPATH=src
python scripts/compare_c1048_probs.py compare \
  --weights third_party/AlphaNLHoldem/weights/c_1048.pkl \
  --data data/c1048_probs.npz
```

## Colab（L4 / T4）

Colab 的 Python 装不了 TF 1.15。在笔记本上导出 npz，上传到 Drive，再在 Colab 里只跑 PyTorch：

```python
from google.colab import drive
drive.mount('/content/drive')
# 代码与 data/c1048_self.npz 放到 /content/poker-ai 之后：
!pip install -q -r requirements-transformer.txt
!PYTHONPATH=src python scripts/train_bc.py \
    --config configs/transformer/colab_l4_conv.json \
    --data /content/drive/MyDrive/poker-ai/c1048_self.npz \
    --init-c1048 /content/drive/MyDrive/poker-ai/c_1048.pkl \
    --loss kl --steps 2000 \
    --out /content/drive/MyDrive/poker-ai/bc_conv.pt
!PYTHONPATH=src python scripts/train_transformer.py \
    --config configs/transformer/colab_l4_conv.json \
    --init-weights /content/drive/MyDrive/poker-ai/bc_conv.pt \
    --resume
```

Transformer 热启动把配置换成 `configs/transformer/colab_l4.json`，并且不要加 `--init-c1048`。

## CPU 冒烟

```bash
export PYTHONPATH=src
python3 -m pytest -q tests/test_bc_baseline.py
python3 scripts/compare_c1048_probs.py    # 无权重时打印 UNVERIFIED
```

覆盖：配置开关、两种编码器的 logit/value 形状、TF same-pad、`extra_info` 不影响输出、权重布局往返、非法教师质量被丢掉、合成数据上 BC loss 下降、BC checkpoint 能被 PPO 的 `load_checkpoint` + `ppo_update` 读入、卷积观测能进同一条 rollout。
