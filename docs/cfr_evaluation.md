# SD-CFR 评测

`scripts/eval_cfr_readiness.py` 在标准 200BB HUNL 上评测冻结的策略档案。
每手双方 20,000 筹码、50/100 盲注，seat 1 为 SB/button，翻后 BB 先行动。
它不操作训练状态，也不修改训练配置。

## 定期评测

`scripts/eval_cfr_periodic.py` 是可被定时器反复调用的一次轮询。它读取训练方
原子发布的 `strategies/index.json`，按里程碑冻结完整的 1..T 策略前缀，
调用离线评测并保存结果。默认里程碑为 50、100、200、400、500、600、800、1000。
默认每次最多完成一个新任务，从最早未完成的里程碑补起。

先复制 [配置示例](../configs/cfr/evaluation_periodic.example.json)，修改
`strategies` 和 `out_dir` 为实际路径。配置中的相对路径相对于配置文件所在目录。

```bash
cp configs/cfr/evaluation_periodic.example.json /path/to/evaluation.json
# 编辑 /path/to/evaluation.json 中的实际输入/输出路径与预算
python scripts/eval_cfr_periodic.py --config /path/to/evaluation.json
```

该命令需要 Linux/macOS 和 CFR 的 Python 依赖，可手动重复运行，也可交给
cron 或 systemd。尚无已发布模型时记录等待状态；不读取未被 index 发布的
快照。一个输出根目录有一个非阻塞进程锁，避免定时器和手动任务重叠。

- 冻结档案位于 `out_dir/archives/iteration_XXXXXXX/strategies/`。训练器以
  原子替换发布快照，因此同盘使用硬链接既保留原始内容又节省磁盘；跨盘回退
  到复制，多个完整前缀会占用额外空间，应预留磁盘预算。
- 冻结期间若发布方替换历史前缀，丢弃临时冻结并留待下次轮询。已有冻结
  不会更新；若相同训练目录中的历史已被重训/恢复替换，明确报错，需为新
  训练历史使用新 `out_dir`，避免混合两个实验。
- 已完成任务在后续轮询中跳过。失败或中断任务继续使用同一结果目录和
  `--resume`，复用已完成且原始日志校验通过的对战。
- 预算、评测代码、模型运行库版本变化时使用新的结果目录，保留旧结果。
  新配置会从最早可用里程碑重新评测；增加未来里程碑不重跑既有里程碑。
- `out_dir/status.json` 保存当前可用轮次、任务状态、报告路径和历史结论。
  结果在 `out_dir/results/iteration_XXXXXXX/<配置与代码指纹>/report.json`，
  同目录 `evaluation.log` 是运行日志。`NO_GO` 是评测已完成的结论，不会因此
  无限重跑；进程异常、档案缺失或日志损坏会明确失败。

### Linux systemd user timer

仓库提供 [service](../tools/cfr_evaluation/cfr-evaluation.service)、
[timer](../tools/cfr_evaluation/cfr-evaluation.timer) 和
[环境文件示例](../tools/cfr_evaluation/cfr-evaluation.env.example)。
先在运行训练的主机编辑环境文件，指定真实的仓库目录、CFR Python 和上述
评测配置文件路径，再执行：

```bash
mkdir -p ~/.config/systemd/user ~/.config/poker-ai
cp tools/cfr_evaluation/cfr-evaluation.env.example ~/.config/poker-ai/cfr-evaluation.env
# 编辑 ~/.config/poker-ai/cfr-evaluation.env，替换三个 /opt/... 示例路径
cp tools/cfr_evaluation/cfr-evaluation.service tools/cfr_evaluation/cfr-evaluation.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now cfr-evaluation.timer
systemctl --user list-timers cfr-evaluation.timer
journalctl --user -u cfr-evaluation.service
```

timer 在启动后约 5 分钟检查一次，此后每次 service 结束 15 分钟后再检查，
可在 timer 文件里调整。service 使用 `Nice=10` 和一核 CPU 配额；默认评测
也限制为一个 CPU 线程。长评测不会被 15 分钟间隔中断，也不启动第二份。
需要用户登出后继续运行时，由主机管理员为该用户配置 systemd user lingering。
停止后续轮询可执行 `systemctl --user disable --now cfr-evaluation.timer`；
正在运行的 service 需单独停止，已完成报告与原始日志保留。
参数语义见 [systemd.service](https://www.freedesktop.org/software/systemd/man/latest/systemd.service.html)
和 [systemd.timer](https://www.freedesktop.org/software/systemd/man/latest/systemd.timer.html)。

这些文件是可选的主机安装模板，提交/合并代码不会自动安装服务，也不要求把
私有训练模型上传 GitHub。定期评测应在能访问本地训练档案的主机运行。

没有 systemd user manager 的容器可在宿主机已有的 cron 中调用同一入口，
例如每 15 分钟执行（先替换所有示例路径）：

```cron
*/15 * * * * /usr/bin/nice -n 10 /path/to/python /path/to/poker-ai/scripts/eval_cfr_periodic.py --config /path/to/evaluation.json >> /path/to/cfr-evaluation-cron.log 2>&1
```

轮询入口本身的目录锁同样防止 cron 重叠；线程数由 JSON 配置限制。该示例不
自动安装 cron 或更改主机现有任务。

## 运行

在仓库根目录使用已装 `requirements-cfr.txt` 的 Python：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python scripts/eval_cfr_readiness.py \
  --strategies /path/to/frozen/strategies --hero-iteration 50 \
  --milestones 10,25,50 --out runs/readiness_t0050
```

默认每个对战 2,500 个配对牌堆（5,000 手）；每个规则对手另测 500 场独立
50 手比赛；自博弈健康检查 10,000 手。完整预算可能需要较长 CPU 时间。
`--stages duplicate,health` 可以先跑配对收益与自博弈。
`--opponents random,call,maniac,overbet,nit,equity` 选择对手。
`--skip-ladder` 跳过内部策略对战。

脚本运行在 CPU；`--threads 1` 为默认。`--cache-size 64` 是每个策略对象的
CPU 快照 LRU 上限，不会裁剪采样分布。它不是 Botzone 部署配置。少量快照的
缓存有助于本地评测；1000 轮全缓存和实际部署的内存预算应另行衡量。

## 统计与复现

- 平均策略每手按迭代号 t 加权选择一次，整手固定，不对当前信息集的局部概率直接求平均。
- 同一个随机种子生成同一牌堆；两局交换策略身份，seat 对应的底牌、烧牌与公共牌不变。
- `mbb_per_hand = mean((SB收益 + BB收益)/2)/100*1000`。
  95% CI 使用**配对均值**的样本标准差（ddof=1）和独立牌对数。
- 独立 50 手比赛不用重复牌堆，每场随机起始座位，然后交替盲位；双方每手重置筹码。
  胜率为总收益严格大于零的场次占比，平局单列；胜率附 Wilson CI，收益 CI 以场为簇。
- 每个任务用从任务名导出的固定 seed。日志逐行记录牌堆种子与原始收益。
  报告记录模型 SHA-256、评测源码 SHA-256、解释器与 PyTorch 版本。
- `--resume` 只复用输入、配置和源码均一致的已完成任务；未完成的任务会从头重跑。
  同一个输出目录一次只运行一个进程。

`report.json` 在每个任务结束后原子更新；同目录的 JSONL 是原始对战记录。
同一候选可在不同输出目录分别运行对手以使用多核，不要超过机器预算。
多组 CI 是逐组区间，没有做多重比较校正。

## 对手定义

`random` 在当前合法五行动中均匀采样；`call` 总是 check/call；`maniac` 尽可能
all-in；`overbet` 尽可能 pot raise，否则 all-in；`nit` 只用 check/call 和 fold，
跟注要求对随机牌的 equity 至少 max(62%, pot odds+12%)。
`equity` 对未知对手底牌和未发公共牌做 Monte Carlo，默认 64 次，按 pot odds+3%
决定跟注，以 65% 和 80% 为半池/满池价值下注阈值。所有规则对手仅接收
`HandSnapshot`，读不到对手暗牌或未来牌堆。

这些是可复现的诊断基线，不是最优响应，也不代表 Botzone 对手整体强度。

## 健康与上线门槛

记录 SB 首次行动的 VPIP/200BB open shove、BB 首次面对开局加注的 fold、
各街行动计数、非法概率泄漏、非法选择、所有 regret 输出的 NaN/Inf、非强制
行动熵和近确定性策略比例。近确定性决策本身不能证明 regret 爆炸。

阈值是本项目的实验目标，不构成“健康 GTO”的理论保证，CFR 对任意两个
中间版本的表现也不保证单调。自我提升门槛要求 1000 vs 100 至少 2500 个
牌对，均值 >=35 mbb/手且 CI 下界 >0。Gauntlet 收益门槛为 random >1500、
call/nit >250、maniac >400、equity >80 mbb/手；500 场 50 手比赛的胜率目标
分别为 random >95%、call/nit >75%、equity >60%。健康门槛为 10000 手自博弈，
合并双方 SB 首次行动开推 <1.5%，非法概率/非法行动/NaN/Inf 为零。

满足样本预算但未达标输出 `FAIL`，不必等训练到 1000 轮才报告已有候选失败。
缺失证据输出 `INCOMPLETE`；任何已确认失败或运行出错使整体为 `NO_GO`。
外部校准、旧模型适配和部署包验收尚未实现，因此本版不能单独给出完整上线
`GO`；它负责离线评测并明确列出仍缺的证据。

下列项目仍须独立完成，当前报告明确保留未完成状态：

- PPO/PFSP 或 c_1048 的规则兼容性核验及同规则基线对战；旧 Botzone simple
  裁判翻后仍由 SB 先行动，与本训练规则不同。
- Slumbot 的 200BB SD-CFR 接口、至少 5,000 手外部校准、off-tree 下注与分街遥测。
  旧 `eval/slumbot_client.py` 默认 50BB cap 且仅支持旧模型，不可直接当本模型成绩。
- 20 快照 FP16 包装与真实 reach-weighted 平均策略蒸馏。直接平均各快照在任意
  信息集的概率，不能当作正确的 SD-CFR 蒸馏标签。
- 对真实部署包做进程冷启动、单线程 CPU 延迟/50 手 CPU 时间、峰值 RSS 和
  裁判差分验收。评测中的 PyTorch 延迟/RSS 仅为诊断数据。

存储预算按实际格式计算：一次迭代快照包含**双方两个网络**，默认结构每份
约 4.8MB。保留双方的 20 份 FP16 快照约需 48MB（尚未包含具体打包开销），
1000 份 FP32 约需 4.8GB。稀疏/量化打包是后续工作，不影响评测的全历史采样。

仓库归档的 `iteration_T.pt` 是本轮遍历使用的 sigma_T，即更新前 theta_(T-1)。
`vs_single_T` 比较的是这个单快照，不将它误称为更新后的 theta_T。
