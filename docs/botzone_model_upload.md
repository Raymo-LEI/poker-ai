# Botzone 模型入口上传说明（T-M12）

把 `laptop_8gb` 的 Botzone transformer 导出成 fp32 `.npz`，再用一个**不导入 torch** 的 Python 入口在 TexasHoldem2p 上出牌。用户自己上传。本文不注册 Botzone，也不提交排名。

仓库和这台云主机上都没有 `update_0000999.pt`。开发和下面的实测用的是同一份网络结构的**随机初始化**权重（`PokerTransformer` 构造时的初值，没有跑训练）。换成真实 checkpoint 只改导出那一条命令，入口和 zip 不用改。

## 英文摘要 / English

Export is one command: checkpoint in, `.npz` out. The zip is source only. Upload `model.npz` separately into account storage; the entry reads `data/model.npz`. The checked-in development weights are a random init of `configs/transformer/botzone_laptop_8gb_maskfold.json` because `update_0000999.pt` is not in the repo. Numpy forward vs torch stays under `1e-4` on 1000 states. Python 3.6.5 (Miniconda) ran the entry and `botzone_bots/rulebot.zip`.

## 两条命令

导出（真实权重）：

```text
python3 tools/export_numpy_weights.py \
  --checkpoint path/to/update_0000999.pt \
  --out model.npz
```

这次机器上没有那份 checkpoint，用的是：

```text
python3 tools/export_numpy_weights.py \
  --random \
  --config configs/transformer/botzone_laptop_8gb_maskfold.json \
  --out /tmp/model.npz \
  --seed 1 \
  --write-checkpoint /tmp/random_init.pt
```

`--random` 只调用和训练相同的 `build_model`，不跑 rollout，不改配置。`--write-checkpoint` 可选，写出训练脚本能读的 `.pt`，方便以后对照。

打包（zip 里没有权重）：

```text
python3 tools/pack_botzone_model.py
```

产物是 `botzone_bots/model.zip`。根目录是 `__main__.py`，旁边是 `decide.py`、`reconstruct.py`、`betting.py`、`observe.py`、`forward.py`、`weights.py`。

## 上传时怎么选

1. 打开 Botzone，「我的 Bot」，新建 Bot。
2. 游戏选 **TexasHoldem2p（双人德州扑克）**。不要选 TexasHoldem2p-complex。
3. 语言选 **python3**。不要选 python2。Wiki 写明 python 运行库里已经有 numpy，不需要为了 numpy 去选 C++ 的「带很多库」编译器。
4. 交互方式选 **JSON 交互**。不要选简化交互，不要开长时运行。
5. 源码上传 `botzone_bots/model.zip`。
6. 权重不要打进 zip。到头像菜单的「管理存储空间」，把导出的 `model.npz` 上传成存储空间里的 `model.npz`（见下一节的假设）。
7. 先调试，确认单步输出是一行 `{"response": <整数>}`。确认之前不要点排名、不要加入计分赛。

抓取时游戏页写的是没有公开比赛。这个包不会替你提交。

## 数据文件怎么读（Wiki 原文，以及必须你确认的假设）

2026-10-16 用 WebFetch 读了这些页：

- https://wiki.botzone.org.cn/index.php?title=Bot
- https://wiki.botzone.org.cn/index.php?title=Bot/en
- https://wiki.botzone.org.cn/index.php?title=%E7%89%B9%E6%AE%8A:%E6%88%91%E7%9A%84%E8%AF%AD%E8%A8%80/TexasHoldem2p

Wiki **写明**的：

- 「数据文件请不要打包，而是使用账户菜单里的『用户存储空间』功能上传，然后在程序中使用 `'data'` 路径访问。」
- Bot「可以读写用户存储空间下的文件」，细节让用户自己在网页的「管理存储空间」里看。Wiki 没有把存储空间的目录布局、文件名规则、容量上限写在词条里。
- JSON 交互里的 `data` 字段是另一回事：上一回合保存的字符串，**最大 100KB**，而且对局结束后不进 log。2.4MB 的权重放不进这个字段。正式比赛中 bot 不能往用户存储空间**写**。本入口只读。
- python2 和 python3 都提供 numpy。词条**没有 numpy 的版本号**。中文页写 python 3.6 的 pytorch 是 1.8.0；英文页仍写 1.4.0。两页都没写 numpy。
- 评测机是 Ubuntu 16.04、x86-64、一个 CPU 核。默认时限 1 秒、256MB；Python 再乘 6。每场第一回合时限再乘 2。
- TexasHoldem2p「只支持 JSON 交互」。

**必须你在网页上确认的假设**（词条没写，入口按这个实现；不对就改 `weights.py` 里的相对路径，不要猜着改训练代码）：

1. 存储空间里的文件，进程当前目录下的相对路径是 `data/<上传时的文件名>`。入口只打开 `data/model.npz`。请把文件名传成 `model.npz`。若管理页显示的路径不是这个，把实际路径发回来再改。
2. 大约 2.4MB 的单个文件在存储配额以内。Wiki 没有公布配额。这次导出的文件是 2387938 字节（592390 个 fp32 参数，权重本体 2369560 字节）。
3. 平台 python3 能 `import numpy`。我们在 Python 3.6.5 上用 numpy 1.16.6 跑通了入口；numpy 1.19.5 在不限制地址空间时 logits 也对，但在 256MB 地址空间上限下，BLAS 的 `matmul` 会段错误（见文末）。平台若是 1.19 且同样在 256MB 虚拟地址上分配失败，这一步会落到回退动作，而不是算出网络输出。

本地测试可以设 `POKER_WEIGHTS=/绝对路径/model.npz`，Botzone 不会设这个变量。`POKER_SAMPLE=1` 改为按掩码后的 softmax 抽样；默认是 argmax。npz 里的 `sample` 字段同样可以打开抽样，这次导出是 `false`。

## 网络和观测

`.npz` 里每个参数都是 float32，另有一段 `meta_json`：

- `arch`：`d_model=128`，4 头，4 层，FFN 256，`max_seq_len=64`，GELU，pre-norm，`vocab_size` 与训练 tokenizer 一致。
- `mask_free_fold`：从导出时的配置来。maskfold 这份配置是 `true`。前向在环境合法掩码之外，把 `to_call == 0` 且还能过牌的 fold logit 设成 `-1e9`。
- `legal_py_sha256`：导出时 `src/alphanl/botzone/legal.py` 的 sha256。这次是 `df77491a9f472c10cb2902e4b0e1e51275d21bcaa56d98f3bb7b013881188b6d`。
- `environment.version`：`botzone-simple-50-100-20000`。这是导出器写的规则标签（盲注 50/100，起始 20000），不是 Botzone 公布的版本号。
- `source.random_init`：随机初值是 `true`。从真实 `.pt` 导出时是 `false`，并记下 checkpoint 路径和 update。

观测按训练环境的 `HandSnapshot` + `BettingHistoryTokenizer` 从重建后的局面生成：牌面 token、街道、动作类型、底池比例桶、`log1p(筹码/大盲)/log1p(200)`。动作类型走训练的 `infer_action_type`（弃牌必须是 id 0；欠注为 0 的加注记成 `bet` 而不是裁判字符串里的 `raise`）。合法动作掩码是环境的五维掩码，fold 在免费过牌时仍然标成合法；`mask_free_fold` 只在前向里生效。

41 场真实牌（`tests/fixtures/botzone_real.jsonl.gz`）上，重建出的筹码、底池、街道、`round_raise`（等于训练 `bound_b / 2`）、本街投入，以及 token / type / numeric / pad / legal mask，都和训练环境逐位相同。7785 个决策，状态不一致 0，观测不一致 0。

## 出牌和回退

正常路径和规则机器人入口同一套裁判映射：网络选中的抽象动作打成 `-1` / `0` / `safe` / `max(safe*2, pot)` / `-2`，裁判接受就原样送出；不接受则改成 `0`（只要裁判接受），否则 `-2`，再否则 `-1`。

模型失败（权重文件不在、异常、logits 非有限）时：局面重建是自洽的，并且裁判接受 `0`，就送 `0`（过牌，或筹码够跟注时的跟注）；否则送 `-1`。读不懂 stdin、或 `my_chips` 对不上，送 `-1`。进程不抛栈。

## 实测

随机初值，`/tmp/model.npz`（2387938 字节）。验收命令：

```text
python3 tools/accept_botzone_model.py --weights /tmp/model.npz --hands 200 --seed 0
```

`taskset -c 0`，子进程 `RLIMIT_AS` 256MB。单核。时间从进程启动算到结束，包含 import numpy、读 npz、一次前向。

前向对齐（当前解释器 torch 2.14.1+cpu、numpy 1.26.4）：1000 个不同的自打局面，原始 logits 和掩码后的 logits 最大绝对误差都 `< 1e-4`（测试 `test_numpy_forward_matches_torch_on_1000_states` 通过）。同一套近似 GELU 在 8 行随机输入上对 torch 的最大绝对误差是 `1.4901161193847656e-08`。

41 场重放与 200 手对局跑完了。命令是上面那条，退出码 0。完整 JSON 在当次标准输出里，摘要如下。

| 项 | 数值 |
|----|------|
| 重放决策 | 7785；状态不一致 0；观测不一致 0；裁判判非法 0 |
| 对随机 | 200 手，入口决策 391，`errors` 空，INVALID/INSUFFICIENT 0，最慢 0.2564662350000617 秒，平均 0.15128922133759432 秒 |
| 对规则机器人 | 200 手，入口决策 484，`errors` 空，INVALID/INSUFFICIENT 0，最慢 0.18197432300030414 秒，平均 0.14268311465496084 秒 |
| 开局一次（exec 之后的进程） | `returncode` 0，`{"response": 0}`，VmPeak 62836 KB，VmHWM 39664 KB |
| `RUSAGE_CHILDREN.ru_maxrss` | 226652 KB |

单步都小于 1 秒。入口自己的驻留峰值是 VmHWM 39664 KB（约 39MB），虚拟峰值 VmPeak 62836 KB（约 61MB），都低于 256MB。`ru_maxrss` 226652 KB 仍低于 256MB，但它是在父进程已经为了重放加载 torch 之后 fork 出来的，会计入 fork 瞬间父进程的驻留；bot 进程自己的峰值以 `/proc/<pid>/status` 的 VmHWM 为准。

## Python 3.6

尝试顺序：

1. **Miniconda 成功。** `Miniconda3-4.5.4-Linux-x86_64.sh` 装到 `/tmp/miniconda36`，解释器 `Python 3.6.5 :: Anaconda, Inc.`。本机没有现成的 conda / docker / pyenv，也没有 apt 里的 python3.6。conda 已经给出真实的 3.6，就没有再装 docker、pyenv 或 deadsnakes。
2. numpy：Wiki 没有版本号。先装 **1.16.6**，再试 **1.19.5**。

| 环境 | 结果 |
|------|------|
| 3.6.5 + numpy 1.16.6，`taskset -c 0`，地址空间 256MB，跑 `botzone_bots/model/__main__.py` 一个翻前决策 | 退出码 0，`{"response": 0}`，0.1558 秒 |
| 同上，跑现成的 `botzone_bots/rulebot.zip` | 退出码 0，`{"response": -2}`，0.0301 秒 |
| 3.6.5 + numpy 1.16.6，8 行随机输入对 torch logits | 最大绝对误差 `1.4901161193847656e-08` |
| 3.6.5 + numpy 1.19.5，不限制地址空间，同一 8 行 | 最大绝对误差相同；入口退出码 0，`{"response": 0}`，0.149 秒 |
| 3.6.5 + numpy 1.19.5，地址空间 256MB | 段错误（`rc -11`），栈在 `forward.py` 的 `np.matmul`。不限制时同一进程能出牌 |

开发机上的验收用的是 Python 3.12 + numpy 1.26.4。这个组合在 256MB 地址空间下跑完了同一步（0.1378 秒，`{"response": 0}`）。另外从一个几乎不占内存的父进程拉起过入口：那次请求被多包了一层，所以走到回退，输出 `{"response": -1}`，`ru_maxrss` 是 32264 KB。它只说明父进程很小时，fork 记账不会冒到两百多 MB。完整前向的峰值以验收脚本里的 VmHWM 39664 KB 为准。

入口源码没有海象运算符、`dataclasses`、`from __future__ import annotations`、f-string 的 `=`。`py_compile` 在 3.6.5 上通过。

全套测试：

```text
python3 -m pytest -q --override-ini='addopts=' --tb=line
```

结果：`213 passed, 1 skipped in 308.98s`，退出码 0。新测试覆盖导出、前向对齐、观测逐位一致、回退、抽样、免费过牌时的 fold 掩码，以及 zip 不含 torch。
