# 德扑AI（单挑NLHE）开源项目调研
数据来源：GitHub API（2026-10-13 查询，许可证/最后推送时间/是否归档）。标【未核实】的为记忆或二手信息。

| 项目 | URL | 许可证 | 维护状态 | 可复用 | 坑 |
|---|---|---|---|---|---|
| AlphaNLHoldem | github.com/bupticybee/AlphaNLHoldem | **AGPL-3.0** | 最后推送 2023-07，110★ | 50bb 单挑基线、动作抽象、有权重，可直接作对手/起点 | AGPL 传染：私有仓库内用没问题，若对外提供服务/发布需开源；训练未收敛；依赖版本老（Ray 等，【未核实具体版本】） |
| AlphaHoldem 论文 (AAAI 2022, Zhao et al.) | 论文，官方代码【未核实是否开源】 | — | — | Trinal-Clip PPO、卡牌/动作张量编码、K-best 自博弈 | 无官方代码则需对照 AlphaNLHoldem 实现 |
| RLCard | github.com/datamllab/rlcard | MIT | 最后推送 2024-06，3.5k★，更新放缓 | NLHE/Limit/Leduc 环境、DQN/NFSP/CFR 基线、评测工具 | 无限注环境下注抽象粗；单挑NLHE性能一般 |
| OpenSpiel | github.com/google-deepmind/open_spiel | Apache-2.0 | 活跃（2026-08） | CFR/CFR+/MCCFR/DeepCFR/NFSP、exploitability 计算（Leduc/Kuhn）、universal_poker | 大游戏 exploitability 算不了；C++ 编译较重 |
| pokerkit | github.com/uoftcprg/pokerkit | MIT | 非常活跃（2026-10-12） | 规则正确、文本手牌历史 → 适合做 Transformer 的 betting-history token 化 | 纯 Python，自博弈吞吐低，需批量/多进程 |
| PokerRL | github.com/EricSteinberger/PokerRL | MIT | 2023-03 后停更 | 分布式 Deep CFR / SD-CFR 训练框架、LBR 评测 | 依赖老旧，可读不建议直接跑 |
| Deep-CFR | github.com/EricSteinberger/Deep-CFR | MIT | 2020 停更 | Deep CFR 参考实现 | 同上 |
| ReBeL | github.com/facebookresearch/rebel | Apache-2.0 | **已归档** | 公开信念状态 + 搜索思路（后期升级） | 官方只发布了 Liar's Dice 实现，无德扑代码【据记忆，建议核实】 |
| DeepStack-Leduc | github.com/lifrordi/DeepStack-Leduc | 无许可证声明 | 2018 停更 | 连续再求解、反事实价值网络 | 无许可证=默认不可复用代码；Lua/Torch7 |
| DeepHoldem | github.com/happypepper/DeepHoldem | 无许可证声明 | 2018 停更 | DeepStack 的 NLHE 移植 | 同上，仅参考思路 |
| slumbot2017 | github.com/ericgjackson/slumbot2017 | MIT | 2019 停更 | Slumbot 的 CFR 代码 | 不是 API 客户端 |
| Slumbot API | slumbot.com（官网提供 Python 示例客户端【未核实当前可用性】） | — | — | 200bb 单挑在线基准，按 bb/100 评测 | 需大量手数降方差；注意 200bb vs AlphaNLHoldem 50bb 的深度差异 |
| Botzone TexasHoldem2p | botzone.org.cn/game/ranklist/63dcfaddee1bce5e6c8f4b53 | — | — | 天梯对战 | 交互协议/时限需读规则【未核实】 |
| Transformer 尝试：AlphaExploitem / PokerGPT / Poker_Transformers | AlphaExploitem 的仓库地址此次未查到【未核实】；PokerGPT 为 arXiv 论文(2024)【代码未核实】 | — | — | 动作历史序列建模、对手建模思路 | Poker_Transformers 曾坍缩为 always-fold：需熵正则、混合策略输出 |

## 建议
1. 环境：pokerkit 做规则与手牌历史；RLCard 只用于小游戏快速验证。
2. 基线：跑通 AlphaNLHoldem 权重 + 评测（注意 AGPL，保持私有）。
3. 正确性：Leduc 上用 OpenSpiel 算 exploitability 验证算法实现。
4. 评测：Slumbot API（需处理 50bb/200bb 差异）+ Botzone；做 AIVAT/大样本降方差。
5. 改造：只替换编码器为小 Transformer（下注历史 token），保留 PPO 与概率策略头，加熵项防坍缩。
6. 后期：参考 ReBeL/DeepStack 思路做实时搜索（DeepStack 系代码无许可证，只读不抄）。
