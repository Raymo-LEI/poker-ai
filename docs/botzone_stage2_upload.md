# Botzone Stage 2 前向入口上传说明

把 Stage 2（50 手上下文残差 PPO，内含 Stage 1 蒸馏参考）的 NumPy 前向打成 Botzone **TexasHoldem2p** 可以直接运行的包。源码 zip 与 checkpoint 无关，已提交为 `botzone_bots/stage2.zip`；权重由 checkpoint 导出成 `stage2_model.npz`，上传到 Botzone 用户存储空间。用户自己上传；本文不注册 Botzone，也不提交排名。

仓库里**没有**真实训练出的 Stage 2 checkpoint。下面的实测用的是正式 1.49M 结构（`configs/cfr/stage1_149m_8m_equity_league_2400.json`，192/6/4/512，开启 equity 特征）的随机初值，残差头另外加了随机扰动，保证残差路径真的参与计算。比分没有意义，只能说明协议、合法性、时间和内存满足要求。

## 平台规则（2026-10-29 用 WebFetch 重新核对）

来源：[Bot](https://wiki.botzone.org.cn/index.php?title=Bot)、[Bot/en](https://wiki.botzone.org.cn/index.php?title=Bot/en)、[TexasHoldem2p](https://wiki.botzone.org.cn/index.php?title=TexasHoldem2p)、[BundleMultiPython](https://wiki.botzone.org.cn/index.php?title=BundleMultiPython)。

| 规则 | Wiki 原文要点 | 本入口的做法 |
|------|---------------|--------------|
| 评测机 | Ubuntu 16.04、x86-64、只有一个 CPU 核 | 导入 NumPy 前把 BLAS/OMP 线程固定为 1；验收用 `taskset -c 0` |
| 时间 | 默认 1 秒，Python 乘 6；每场第一回合再乘 2 | 实测单步约 0.15 秒（含解释器启动、`import numpy`、读 6MB 权重、equity、前向） |
| 内存 | 256MB | 验收给每个子进程设 `RLIMIT_AS` 256MB；实测峰值 RSS 约 39MB |
| 语言 | python3（3.6）；提供 numpy，**不写 numpy 版本**；中文页说 3.6 的 pytorch 是 1.8.0，英文页写 1.4.0 | 入口只依赖 NumPy；用 Python 3.6.5 + NumPy 1.19.5 实测 |
| 多文件 | zip 根目录放 `__main__.py`，整包上传 | zip 根目录是 `__main__.py`，模块在 `stage2_runtime/` |
| 数据文件 | 不要打包，用「用户存储空间」上传，程序里用 `data` 路径读 | 权重读 `data/stage2_model.npz` |
| 交互 | TexasHoldem2p **只支持 JSON 交互**；输入是单行 JSON：`requests`、`responses`、`data`、`globaldata`、`time_limit`、`memory_limit` | 读一行 stdin，输出一行 `{"response":…,"data":…,"debug":…}` |
| `data` | 上回合保存的信息，最大 100KB，不进 log | 保存对手 HUD 和本场进度，实测最大约 2.9KB |
| `debug` | 写入 log，最大 1KB | 写解释器/NumPy 版本、是否正常加载权重、耗时；截断到 1000 字符 |
| 长时运行 | 输出 `>>>BOTZONE_REQUEST_KEEP_RUNNING<<<` 才开启；开启后 `data` 失效 | 不开启。上下文靠 `data` 跨进程保存，开长时运行会丢上下文 |
| 比赛 | 一场 50 手；每手筹码重置为 20000，盲注 50/100；庄家每手轮换；按累计赢得筹码 / 100 排名 | 与 Stage 2 训练环境的 50 手比赛一致 |
| request | `dealer_id`、`my_id`、`my_chips`、`my_cards`、`public_cards`、`history`（每条有 `round`、`player_id`、`action`、`action_type`），以及 `hand`、`max_hand`、`total_win_chips`、`total_win_games` | 局面只从最后一条 request 重建；`hand` 和 `total_win_chips` 驱动比赛上下文 |
| response | `-1` 弃牌，`-2` 全下，`0` 跟注/过牌，正整数是**额外**投入的筹码 | 网络的 5 个抽象动作映射成这四种整数 |
| 加注下限 | 不少于本轮最大加注的 2 倍；本轮首个加注不少于大盲；剩余筹码必须**大于**所需筹码 | 半池/满池额度不足时抬到裁判最小加注；仍不合法的分支不开放 |
| 非法操作 | 崩溃、超时、非整数、筹码不够的跟注/加注、有人全下后还跟注/加注，都按弃牌处理 | 选中的动作先用裁判规则核对；任何异常都退回裁判一定接受的动作 |
| 正式比赛 | bot 不能往用户存储空间写 | 入口只读 |

Wiki 正文里写「每步要在 1 秒内完成」，那是基础时限；Bot 词条写明 Python 乘 6。入口单步约 0.15 秒，按 1 秒算也满足。

## 必须由你在网页上确认的三件事

Wiki 没写，入口按下面的假设实现。不对就把实际情况发回来，不要猜着改：

1. **存储空间路径**：入口依次找 `data/stage2_model.npz`（相对当前目录）、zip 所在目录下的 `data/stage2_model.npz`、`/data/stage2_model.npz`。请把文件名传成 `stage2_model.npz`。
2. **容量**：权重文件约 6MB（1,508,683 个 fp32 参数，6,055,781 字节）。Wiki 没公布配额。
3. **NumPy 版本**：equity 特征用到 `np.random.default_rng`，需要 NumPy ≥ 1.17。Python 3.6 能装的最高版本是 1.19.x，大概率满足，但 Wiki 没写。第一局打完后看 log 里的 `debug`：正常是 `py3.6.x np1.xx.x ok stage2_model.npz …ms`；若出现 `fallback`，后面就是失败原因。

## 生成与上传

在有 torch 的机器上，对 Stage 2 的 `train.pt`：

```bash
python3 tools/pack_botzone_stage2.py \
  --checkpoint runs/stage2_candidate/train.pt \
  --weights-out stage2_model.npz
```

它总会重写 `botzone_bots/stage2.zip`（源码只依赖仓库代码，和 checkpoint 无关），并导出 `stage2_model.npz`（权重和 `meta_json`：配置、残差设置、`numeric_version`、是否开 `--safety`）。不带 `--checkpoint` 时只重打 zip。`--safety` 与单文件导出相同：高 SPR 下把很小的全下概率移到跟注，属于部署干预，需要单独评测，默认关闭。

上传：

1. Botzone「我的 Bot」→ 新建 Bot。游戏选 **TexasHoldem2p**（不要选 TexasHoldem2p-complex）。
2. 语言选 **python3**；交互方式选 **JSON 交互**；不要开长时运行。
3. 源码上传 `botzone_bots/stage2.zip`。
4. 头像菜单 →「管理存储空间」，上传 `stage2_model.npz`，文件名保持不变。
5. 先开调试对局，确认 log 里的 `debug` 是 `ok`。确认之前不要加入排名或计分赛。

本地复现单步：

```bash
mkdir -p /tmp/bz/data && cp stage2_model.npz /tmp/bz/data/
cd /tmp/bz && echo '{"requests":[{"num_players":2,"dealer_id":0,"my_id":1,"my_chips":19950,"my_cards":[51,23],"public_cards":[],"history":[],"hand":0,"max_hand":50,"total_win_chips":[0,0],"total_win_games":[0,0]}],"responses":[],"data":"","globaldata":""}' \
  | python3 /path/to/poker-ai/botzone_bots/stage2.zip
```

## 和单文件导出的区别

`scripts/export_stage2_agent.py` 生成的单文件把 6MB 权重 base64 后嵌进源码，模块里还有 `dataclasses` 和 `from __future__ import annotations`，在 Botzone 的 Python 3.6 上会直接报错，也违反「数据文件不要打包」的要求。Botzone 包与它共用同一套源码收集逻辑（`alphanl.stage2.export._sources`），区别如下：

- 去掉 `from __future__ import annotations`；NumPy 版 `torch` 替身补上注解里用到的 `device`，`hunl` 片段补上 `typing` 导入。
- 平台没有 `dataclasses` 时（3.6），入口装上 `dataclasses_compat.py`。它只实现打包模块用到的 `dataclass(frozen/order)`、`field`、`replace`、`asdict`，以及继承。
- 矩阵乘保留 `einsum(..., optimize=False)`，不换成 `np.matmul`：NumPy 1.19.5 在 256MB 地址空间下走 BLAS GEMM 会段错误（见 [模型入口说明](botzone_model_upload.md)）。
- equity 的 `numeric_version` 由 zip 内嵌的预翻牌表字节现算；加载时和 npz 记录的版本比对，zip 与权重不配套就报错并退回。
- 权重外置，zip 约 29KB。

## 决策与回退

正常路径与单文件导出相同（`portable_runtime.respond`）：

- 从最后一条 request 重建局面，投影成 Stage 1 的观测（裁判庄家 = 标准 BB 座位），经 `CFRActionGuard`、tokenizer、equity，跑参考网络和残差头，按概率**抽样**一个抽象动作。
- 半池/满池换算成额外筹码，不足裁判最小加注时抬到最小值；结果再用裁判规则核对一遍。免费过牌时不开放弃牌。
- `data` 保存比赛上下文；同一请求重放时直接返回上次的动作（幂等）。Botzone 把 `data` 原样或作为字符串还回来都能读。跨手比分读 `total_win_chips`。

回退：读不懂 stdin 时送 `-1`；权重缺失、格式不符、版本不配套或任何异常时，局面能自洽重建且裁判接受 `0` 就送 `0`（过牌或跟注），否则送 `-1`。原因写进 `debug` 和 stderr，进程不抛栈。

## 验收

`tools/accept_botzone_stage2.py` 用仓库里的真实裁判（`tests/fixtures/botzone_judge.py` 的 `judge()`）按平台方式维护 log，驱动完整的 50 手比赛。Stage 2 每一步都是新进程（`taskset -c 0`、`RLIMIT_AS` 256MB），喂累计的 `requests`/`responses`、字符串形式的 `data`、`globaldata`、时限。出现以下任一情况都算错误，脚本以非 0 退出：裁判判定非 OK、裁判拒绝英雄的动作、模型走回退、`data` 超过 100KB 或 `debug` 超过 1KB、单步超时（首回合 12 秒，其后 6 秒）、上下文手数与 `hand` 不符。

```bash
python3 tools/accept_botzone_stage2.py --weights stage2_model.npz \
  --python /path/to/python3.6 --matches 4 --opponent random
python3 tools/accept_botzone_stage2.py --weights stage2_model.npz \
  --python /path/to/python3.6 --matches 2 --opponent rule
```

本机 Python 3.6.5 是 `Miniconda3-4.5.4` 自带的解释器，另装 NumPy 1.19.5。权重为上述 1.49M 随机初值，zip 为提交的 `botzone_bots/stage2.zip`：

| 环境 / 对手 | 场次 | 入口决策 | 错误 | 最慢单步 | 平均单步 | 子进程峰值 RSS |
|-------------|------|----------|------|----------|----------|----------------|
| 3.6.5 + NumPy 1.19.5，对随机合法动作 | 4 | 373 | 0 | 0.156 秒 | 0.134 秒 | 39112 KB |
| 3.6.5 + NumPy 1.19.5，对规则机器人 | 2 | 173 | 0 | 0.358 秒 | 0.138 秒 | 39340 KB |

`debug` 样例：`py3.6.5 np1.19.5 ok stage2_model.npz 103ms`；`data` 最大 2905 字节。验收父进程不导入 torch，所以子进程 RSS 就是 bot 自己的量级。

反向检查：把权重换成格式不对的 npz，同一脚本报出每一步的 `model fallback` 并以退出码 1 结束；回退动作仍被裁判接受。

单元测试 `tests/test_botzone_stage2.py`：

- zip 内前向在一整手四条街上与 torch 的 Stage 2 模型逐位比对观测、logits 误差 < 2e-5；
- 强制装上 `dataclasses_compat` 时，局面、合法动作、裁判整数和比赛上下文与真实 `dataclasses` 完全相同；
- 输出单行 JSON，`debug` ≤ 1KB、`data` < 100KB；`data` 以字符串还回时幂等；
- 缺权重或 stdin 不是 JSON 时退回合法动作；
- zip 里没有 `from __future__ import annotations`、torch 导入或 `np.matmul(`；
- 提交的 `botzone_bots/stage2.zip` 与当前源码重打的结果逐字节一致（改了 `src/` 里被打包的模块后要重跑 `tools/pack_botzone_stage2.py`）；
- 用小模型跑一整场真实裁判比赛；
- 设了 `POKER_PY36=/path/to/python3.6` 时，在 3.6 解释器上比对同一组请求（logits 容差 1e-5，其余完全一致）。

全套测试（`POKER_PY36` 指向上述 3.6.5）：`552 passed, 6 skipped`。
