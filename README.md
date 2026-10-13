# poker-ai

单挑无限注德州扑克（heads-up no-limit Texas Hold'em）AI。

## 目标

以 [AlphaNLHoldem](https://github.com/bupticybee/AlphaNLHoldem) 为基线，用 [pokerkit](https://github.com/uoftcprg/pokerkit) 做规则引擎。评测对标 Slumbot API 与 Botzone TexasHoldem2p。改造时一次只改一件事：先换成下注历史 token 上的小型 Transformer 编码器，保留 PPO 和概率策略头，并加上熵正则。

算力：笔记本 RTX 3070、家中 RTX 4070 Super（经 Tailscale），Colab 作补充。

## 路线

1. **复现，并打过随机 / 规则机器人。** 跑通 AlphaNLHoldem，规则交给 pokerkit。先在随机对手和简单规则机器人上建立可训练、可对战的基线。
2. **Slumbot 基线。** 对接 Slumbot API，按 bb/100 记录成绩，并同时接 Botzone TexasHoldem2p。AlphaNLHoldem 是 50bb，Slumbot 是 200bb，深度不同，两边的数字不能直接横比，报告里要写明这个差距。
3. **换成小型 Transformer。** 用下注历史 token 上的小 Transformer 编码器替换原编码器。保留 PPO 与概率策略头，加入熵正则，防止策略坍缩。这一步只换编码器。代码与运行方式见 [docs/step3_transformer.md](docs/step3_transformer.md)（只实现，不在本步训练）。

以后再做：对手建模、100bb、实时搜索。

调研笔记见 [docs/reference_survey.md](docs/reference_survey.md)。

## 许可

AlphaNLHoldem 为 **AGPL-3.0**。本仓库因此保持私有：在私有仓库内使用没有问题；若对外提供服务或发布衍生作品，需按 AGPL 开源。pokerkit 为 MIT。
