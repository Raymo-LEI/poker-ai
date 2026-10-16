# Botzone 规则机器人上传说明（T-M12b-lite）

只玩现有规则机器人，不加载权重，不导入 torch。用户自己上传。本文不注册 Botzone，也不提交排名。

入口源码在 `botzone_bots/rulebot/`，可上传的 zip 是 `botzone_bots/rulebot.zip`（由 `tools/pack_botzone_rulebot.py` 生成）。zip 根目录是 `__main__.py`，旁边是 `decide.py`、`policy.py`、`reconstruct.py`。

## 上传时怎么选

1. 打开 Botzone，进入「我的 Bot」，新建 Bot。
2. 游戏选 **TexasHoldem2p（双人德州扑克）**。不要选 TexasHoldem2p-complex（那是筹码跨局继承，规则不同，这个包会算错）。
3. 语言选 **python3**。不要选 python2。不需要「带很多库」的编译器：这个包只用标准库。
4. 交互方式选 **JSON 交互**。不要选简化交互，不要开长时运行。
5. 把 `botzone_bots/rulebot.zip` 整包上传（Botzone 的 Python 多文件入口就是 zip 根目录的 `__main__.py`）。
6. 先本地或平台调试，确认单步输出是一行 `{"response": <整数>}`。确认之前不要点排名、不要加入计分赛。

抓取时游戏页 [TexasHoldem2p](https://en.botzone.org.cn/game/TexasHoldem2p) 写的是 **No open contests available**。没有可以替你勾选的公开比赛。若你自己的群组里有比赛，在网页上加入；这个包不会替你提交。

## 交互格式（已用 WebFetch 核对，不是猜测）

- [Bot](https://wiki.botzone.org.cn/index.php?title=Bot)：平台有 JSON 交互和简化交互两种。每次进程重启，stdin 带上此前全部 `requests` / `responses`。默认时限 1 秒、256MB；Python 时限再乘 6。第一回合时限加倍。
- [TexasHoldem2p](https://wiki.botzone.org.cn/index.php?title=%E7%89%B9%E6%AE%8A%3A%E6%88%91%E7%9A%84%E8%AF%AD%E8%A8%80/TexasHoldem2p) 写明：**「本游戏只支持 JSON交互。」** 所以没有采用任务里的简化 I/O 默认（读完全部 stdin 后只打印一个整数）。简化 I/O 在这个游戏上不可用。

官方样例的读法是：

```python
requests = json.loads(input())["requests"]
action = get_action(requests[-1])
print(json.dumps({"response": action}))
```

因此 stdin 是一个 JSON 对象，`requests` 的元素是对象（样例直接用 `requests[-1]["my_id"]`）。stdout 是一行 `{"response": <int>}`。`response` 的取值是裁判的四种码：`-1` 弃牌，`-2` 全下，`0` 跟注或过牌，正整数是这一下额外投入的筹码（不是 raise-to）。

实现还多接受一种包法：若某个 `requests[i]` 是字符串，会再 `json.loads` 一次。通用 Bot 词条的示意图把 request 写成字符串，游戏样例写成对象，两种都覆盖。状态只从**最后一条** request 重建（`dealer_id`、`my_id`、`my_chips`、`my_cards`、`public_cards`、本手 `history`）。更早的手和 `responses` 不参与记账。盲注不在 `history` 里。

和裁判夹具的对应关系：`tests/fixtures/botzone_judge.py` 的 `make_request_json` 把上述字段放进 `content[座位]`，`history` 里的加注 `action` 是额外筹码，跟注/过牌是 `0`，全下是 `-2`，终局弃牌没有 `round`。重放 `tests/fixtures/botzone_real.jsonl.gz` 时，请求就是按这个结构从活裁判抄出来的。

## 和裁判源码核对后的记账

任务摘要和 `Holdem.player_action` / `deal_cards_and_blind` 一致，没有改摘要：

- 盲注预填：非庄家 50，庄家 100。二人局庄家是大盲。
- `round_raise` 是本街被接受的单次正数下注的最大值（盲注走加注分支，所以翻前盲注之后是 100）。跟注/过牌不更新它。全下记 `-2`，也不是正数下注，不更新它。
- 每条翻后街在 `_next_round` 里把 `round_raise` 重置为 `big_blind // 2`，即 50。此时无人下注，最小正数额外筹码是 `2 * 50 - 0 = 100`。
- 加注记录里的 `action` 是额外筹码。`0` 的金额用 `round_bet - own` 推。全下金额用「起始 20000，减去已经投入的」推。
- 非庄家（小盲）翻前、翻后都先行动。

补充一处代码路径，不改变上面的结论：裁判只在「弃牌后只剩一名玩家」时省略 `round`（`botzone_judge.py` 约 279–283 行）。弃牌后还不止一人时，会落到后面带 `round` 的记录。TexasHoldem2p 固定 2 人，每次弃牌都会结束本手，所以实牌里的弃牌记录都没有 `round`。重放 41 场时终局弃牌确实没有 `round`。

维基**样例程序**里的 `round_raise` 不是裁判的那个变量。样例把翻前初始值设成 `2 * 100 = 200`，新街重置成 `100`，再用 `max(round_raise, 2 * action)` 更新。裁判存的是「单次正数下注的最大值」（翻前后 100，新街 50）。本入口跟裁判，不跟样例。样例若按它自己的 `round_raise` 去卡下界，翻前小盲会把 200 当成最小加注；裁判接受的最小额外筹码是 150。

## 规则机器人怎么落到裁判合法的整数

策略与 `src/alphanl/botzone/rule_bot.py` 相同：翻前点数、翻后真实牌面的 1–9 档、四张优先级表都不改。五个动作仍按 `actions.py` 打包：`-1`、`0`、`safe`、`max(safe * 2, pot)`、`-2`。`safe` 与 `src/alphanl/botzone/legal.py` 的 `safe_extra` 相同，也就是 `policy.py` 里的 `max(1, bound_b - own)`（`bound_b` 存 `2 * round_raise`）。不再取更严的 `max(下界 A, 下界 B)`。翻前小盲的最小加注因此是 150，和裁判一致。`decide.map_target` 的降级没有改：裁判接受就原样送出，否则能跟注就送 `0`，再不行送 `-2`，仍被拒绝才送 `-1`。

zip 里是一份 Python 3.6 / 标准库的副本，不导入 `legal.py`（那里有 `dataclasses` 和 `from __future__ import annotations`，3.6 跑不了，而且任务禁止改它）。测试把副本的翻前全部起手、翻后抽样、以及整手动作整数对回 `BotzoneRuleBot`。

映射：

- 规则机器人给出的整数若裁判接受，原样送出。
- 不接受则降成跟注 `0`（只要 `0` 合法）。
- 跟注也不合法（对手已全下，或剩余筹码不够）则送 `-2`；`-2` 仍被拒绝才送 `-1`。
- 任何异常：能过牌就送 `0`，否则送 `-1`。读不懂 stdin 时送 `-1`。不崩溃。异常路径不用 `0` 去跟注，避免 `INSUFFICIENT_CHIPS`。

## 需要你确认的事

- 游戏是 TexasHoldem2p，不是 complex。盲注 50/100、起始 20000 是这个游戏的裁判写死的，`initdata` 只改得了 `max_hand` 和牌序。
- 交互是 JSON。若上传页误选简化交互，平台喂进来的不是 JSON，这个入口会输出 `{"response": -1}`，那一步会被当成非法。
- 公开比赛列表在抓取时是空的。要打哪个群组赛由你在网页上决定。
- `requests` 的元素按官方样例当对象解析，同时兼容 JSON 字符串。若你从平台复制的「玩家输入」两种都不是，把第一行发回来再改解析。

## 实测资源

命令（单核 + 地址空间 256MB，时间从进程启动算到结束）：

```text
python3 tools/accept_botzone_rulebot.py --hands 200 --seed 0
```

`taskset -c 0`，子进程里 `resource.setrlimit(RLIMIT_AS, (256*1024*1024, 256*1024*1024))`。41 场重放 7785 个决策，筹码 / 对手本街投入 / `round_raise` / 最小加注不一致 0，裁判拒绝入口输出 0。对随机 200 手（入口 348 次决策）和对规则机器人 200 手（入口 727 次决策）的 `INVALID` / `INSUFFICIENT_CHIPS` / 崩溃都是 0。

| 项 | 数值 |
|----|------|
| 单步最慢（对随机，`max_turn_s`） | 0.030719599999997627 秒 |
| 单步最慢（对规则机器人，`max_turn_s`） | 0.03810702999999194 秒 |
| 单步平均 | 0.01819246079023206 秒 / 0.01790765748830789 秒 |
| 全部入口进程的最大 RSS（`ru_maxrss`） | 32864 KB |
| 一次开局采样的 VmPeak / VmHWM | 18268 KB / 11024 KB |
| 地址空间上限 | 256 MB，没有进程被杀掉 |

单步都小于 1 秒，RSS 和采样到的虚拟内存峰值都小于 256 MB。开局那一次的 VmHWM 可能采在进程退出前的瞬间，跨全部决策的峰值以 `ru_maxrss` 的 32864 KB 为准。

## Python 版本

Botzone 的 python3 是 3.6。本机 `uv python install 3.6` 没有对应的 CPython 构建，环境里也没有 apt / conda / pyenv 的 3.6。用 vermin 1.9.1 扫四个入口文件，报告的最低版本是 **2.6 / 3.0**（3.6 能跑），再用当前解释器 `python3 -m py_compile` 通过。入口没有海象运算符、`dataclasses`、`from __future__ import annotations`、f-string 的 `=` 调试语法。
