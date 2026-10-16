# Windows 上自动续训（重启笔记本之后不用人手点）

训练进程被杀掉（关机、重启、睡眠把后台进程清掉）之后，下一次登录要自己把 PPO 拉起来，从最近一份**写完的** checkpoint 继续，直到配置里的 `train.num_updates` 跑完。

脚本是 `tools/train_autoresume.ps1`。它循环调用：

```text
python scripts/train_transformer.py --config <配置> --resume auto --updates <目标步数>
```

- 退出码 **0**：目标步数已经跑完，脚本自己结束，不会空转。
- 退出码 **2**：配置哈希和 checkpoint 对不上。这不是暂时故障，脚本**不再重试**。先停任务，再决定改回原配置，或显式加上 `-AllowConfigMismatch`。
- 其他非 0（包括进程被直接杀掉）：等 30 秒，再带 `--resume auto` 启动。

`--resume auto` 只认 `train.checkpoint_dir` 里的 `update_*.pt`。它会跳过读不出来的文件，退回到上一份完整文件；目录里一份都没有就从头训练。半截写入留在同目录的 `.tmp` 上，不会盖掉已经写好的 `update_*.pt`。

牌桌打到一半的状态**不恢复**。续训时各环境重新发牌。checkpoint 里仍有每张牌桌的发牌 RNG，所以同一次边界上的重发是确定的。`--resume`（不带 `auto`）的行为跟以前一样，会接着未打完的那一手。

## 先改配置（第一次开跑之前）

默认 `checkpoint_every` 是 50，`keep_last` 是 3。自动续训请在要用的 JSON 里改成：

```json
"checkpoint_every": 25,
"keep_last": 20
```

25 次更新大约 5 分钟一份。`keep_last` 至少留 20 份，这样最新一份损坏时，回退用的前一份还在。这两个数一旦写进 checkpoint 的配置哈希，后面每次启动都要保持一致；中途改了会退出码 2。

不要改已经在跑的 `runs/` 目录。新的输出目录用配置里的 `out_dir` / `checkpoint_dir`，或命令行 `--out-dir`、`--checkpoint-dir`。

命令行也可以覆盖（每次启动都要带上同样的参数，脚本用 `-ExtraArgs`）：

```powershell
--checkpoint-every 25 --keep-last 20
```

## 手动试一次

在仓库根目录、PowerShell 里：

```powershell
cd C:\poker-ai
.\tools\train_autoresume.ps1 `
  -Config configs\transformer\laptop_8gb.json `
  -Python C:\poker-ai\.venv\Scripts\python.exe `
  -LogFile C:\poker-ai\runs\autoresume.log
```

`-TargetUpdates` 可以不写，脚本读配置里的 `train.num_updates`。要改目标步数就显式传入，脚本会把它当成 `--updates` 传给训练（这是总步数，不是“再跑多少”）。

日志：`runs\autoresume.log`，每次尝试还有 `train-<时间>.out.txt` 和 `.err.txt`。

## 登录时自动启动

把下面的路径换成这台机器上的仓库路径。触发器是**用户登录**（`onlogon`），不是开机：开机时往往还没登录，GPU 会话也不在。

任务计划程序默认会在 72 小时后杀掉任务。训练远比这长，所以用 `Register-ScheduledTask` 把执行时间限制设为 0（不限制）。`schtasks /create` 写不出这个限制，只适合做对照。

### 推荐：Register-ScheduledTask

在 PowerShell 里整段执行。`ExecutionTimeLimit` 为 0 表示不限时。接电、用电池都允许跑，不要因为切到电池就停。

```powershell
$repo = "C:\poker-ai"
$arg = "-NoProfile -ExecutionPolicy Bypass -File `"$repo\tools\train_autoresume.ps1`" -Config `"$repo\configs\transformer\laptop_8gb.json`" -Python `"$repo\.venv\Scripts\python.exe`" -LogFile `"$repo\runs\autoresume.log`""
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arg -WorkingDirectory $repo
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Seconds 0) `
    -RestartCount 0 `
    -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName "AlphaNL-TrainAutoresume" -Action $action -Trigger $trigger -Settings $settings -User $env:USERNAME
```

登录之后任务会拉起脚本；脚本再拉起训练。笔记本重启、人重新登录后，不需要再手工开训练。

### 对照：schtasks /create

```powershell
schtasks /create /tn "AlphaNL-TrainAutoresume" /sc onlogon /tr "powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\poker-ai\tools\train_autoresume.ps1 -Config C:\poker-ai\configs\transformer\laptop_8gb.json -Python C:\poker-ai\.venv\Scripts\python.exe -LogFile C:\poker-ai\runs\autoresume.log"
```

`/tr` 有长度上限。路径一长就用上面的 `Register-ScheduledTask`。用 `schtasks` 创建之后，仍要到“任务计划程序”里把“如果任务运行时间超过以下时间，停止任务”改成不限制，否则几天后任务会被系统杀掉，脚本的重试循环也一起没了。

## 怎么停

只结束这一次训练、下次登录还要自动续上：停掉 python，别删任务。

```powershell
Stop-ScheduledTask -TaskName "AlphaNL-TrainAutoresume"
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -match "train_transformer.py" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

`Stop-ScheduledTask` 在有的系统上只对“正在运行的任务”有效。任务已经停了、只是 python 还在，就执行上面的 `Stop-Process`。

连下次登录也不要再启动：

```powershell
Disable-ScheduledTask -TaskName "AlphaNL-TrainAutoresume"
# 或者删掉
Unregister-ScheduledTask -TaskName "AlphaNL-TrainAutoresume" -Confirm:$false
schtasks /delete /tn "AlphaNL-TrainAutoresume" /f
```

先禁用或删除任务，再停 python。顺序反了的话，如果任务还开着且设置了失败重启，它可能把训练再拉起来。这个脚本自己的重试在进程里面；任务被禁用后，当前这次 python 停掉就不会再有下一次。

查看状态：

```powershell
Get-ScheduledTask -TaskName "AlphaNL-TrainAutoresume" | Get-ScheduledTaskInfo
Get-Content C:\poker-ai\runs\autoresume.log -Tail 40
```
