# Botzone Transformer 里程碑（M1–M12）

依据：已合进 `main` 的 `docs/botzone_transformer_design.md` §6（以及 §5.4 过关线）。
旧草稿 `/workspace/t-d1`、`/workspace/t-d2` 仅作参考，正式以设计文档为准。

## 总原则

- **一次只改一件事。** 每一行「只做」之外的文件与行为一律不动。
- **前件未绿不开后件。** 某步验收没过，不开始依赖它的下一步。
- **Gate = M1 绿。** M1 未绿之前，M2–M12 一律不开工。
- 角色约定：
  - **云端 agent**：复杂（碰规则分歧，或跨牌桌 / 训练循环 / checkpoint），默认 Grok 4.7 high。
  - **Code Bot**：简单代码（行为已写死，改动落在一两个文件，验收=新测试变绿）。
  - **人**：笔记本或 Colab 上真跑；agent 不改算法。

## M1–M12

| 步 | 只做这一件事 | 谁来做 | 依赖 | 怎样算做完（验收） |
|----|----------------|--------|------|---------------------|
| **M1** | 合法性纯函数和下界 A/B，含五手棋。不发牌、不训练 | 云端 agent | —（Gate） | 设计文档 §1.4 五手棋测试绿 |
| **M2** | `BotzoneTable`：发牌、按钮、盲注 50/100、起始 20000、摊牌、烧牌、筹码守恒。还不进 PPO；不改 rlcard/pokerkit | 云端 agent 写 `src/alphanl/botzone/table.py`；五手棋之外的随机对局守恒测试交给 Code Bot | M1 | 1000 手 `剩余+底池=40000`、盈亏和为 0；M1 五手棋仍绿；奇数底池余 1 给座位号较小方 |
| **M3** | 5 档动作（fold / check_call / raise_min / raise_large / all_in）只通过 M1 的 `legal_response` 变成响应；掩码与发包同一函数 | 云端 agent | M1+M2 | 每个未屏蔽 id 的响应合法；被屏蔽 id 无响应；全押后仅 -1/-2 |
| **M4** | `env.backend=botzone` 接上 `SelfPlayEnv`；checkpoint 记 backend+盲注；botzone 下默认拒绝 `--init-weights` | 云端 agent | M3 | 2 步 rollout + 1 次 PPO 接线测试绿；rlcard checkpoint 载入 botzone 必须失败；反向亦然 |
| **M5** | 数值特征改成 `norm=log1p(筹码/大盲)/log1p(200)`；词表 / token id 不动 | Code Bot | M4 | 原有 token id 测试仍绿；数值有单独期望值断言（1bb≈0.131、200bb=1） |
| **M6** | 仅 botzone 后端：奖励 = 筹码变化/大盲/200；日志乘回 bb；其他 backend 不变 | Code Bot | M4 | 已知一手的奖励与手算一致；日志 bb/手 = 奖励×200 |
| **M7** | 新规则 bot + 评测脚本（bb/100、95% CI、50 手一场分）。优先级表以设计文档 §5.3 为准，禁止自创强度 | Code Bot | M3（可与 M5/M6/M8 并行） | 规则 bot 对随机打出带区间的 bb/100，记为 R（过关尺子） |
| **M8** | 一份新 json：`backend=botzone`、盲注 50/100、起始 20000；`model` 段与 `laptop_8gb.json` 相同 | Code Bot | M4 | `--dry-run` 参数量仍约 592,390 |
| **M9** | 笔记本跑 200 个 update（AMP 开）；记 `--profile-rollout`。**不改算法** | 人（用户/幕僚长） | M5+M6+M8（建议 M7 也齐） | 无 NaN/Inf；峰值显存 <7GB；若合法熵 <0.2 且「可过牌仍弃牌」>0.5 → 塌缩，停，进 M10；否则跳过 M10 直接 M11（设计 §5.4.3） |
| **M10** | **仅塌缩时**：规则 bot 牌谱 + 现有 `train_bc.py` KL；不用 c_1048 | 云端 agent | M9 判定塌缩 + M7 | KL 下降；载入后熵回到 >0.2 |
| **M11** | 正式 PPO，训到过关；对随机、对规则 bot 各 ≥5000 手（必要时 20000）；小网络不加宽 | 人（笔记本或 Colab） | M9 未塌，或 M10 过 | 对规则 bot 的 95%CI 下界 >0；对随机点估计 ≥ R；报 bb/100 与一场分（≥100 场）分列（设计 §5.4.4） |
| **M12** | 无 torch 的 NumPy 前向；对拍 logits；量 RSS 与单步耗时 | 云端 agent | M11 有可提交权重（或至少有未塌 checkpoint 可对拍结构） | logits 最大绝对误差 <1e-4；RSS/耗时写入文档。**本步不注册 Botzone、不提交排名** |

## 并行机会（M1 绿之后）

- **M4 绿之后：** M5 ∥ M6 ∥ M7 ∥ M8（M7 实际只依赖 M3，可更早；建议与 M5/M6/M8 同批推进）。
- M2 的「1000 手守恒测试」可与云端写牌桌尾部重叠，由 Code Bot 接。
- **M10 仅在 M9 判定塌缩时才开**；未塌则跳过。

## 刻意不做 / 另开

- **多进程 rollout：** 单独里程碑，不绑 M4。现有剖析入口是 `scripts/smoke_metrics.py --profile-rollout`。
- **真实 Botzone 对局对拍：** 有回放后若下界要改，只重开 M1，再跑一遍 M3；网络宽度不动。
- **Slumbot / c_1048 / Botzone 排名：** 非 v1 过关线。

## 与旧 T-D1 / T-D2 的关系

旧 `/workspace/t-d1`、`/workspace/t-d2` 可作参考，但以合进 `main` 的设计文档为准。M2 按文档新建 `src/alphanl/botzone/table.py`，不要静默把未合入的旧适配器当正式 backend。
