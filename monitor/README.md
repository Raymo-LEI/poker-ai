# Overwatch monitor

整套控制台随 `poker-ai` 仓库维护。前端保留神经／纸本两种主题；监控只读已有日志，不加载模型。
仅需 Python 3.10+ 标准库；远程模式另需 OpenSSH 和已有的 SSH 登录配置，无需安装训练依赖或 npm。

## 本地与服务器端

从仓库根目录（或独立包解压后的根目录）运行：

```bash
python3 scripts/monitor_training.py --runs-dir /absolute/path/to/runs --port 6008
```

访问 `http://127.0.0.1:6008`。`--runs-dir` 可重复，支持 run 本身或其直接父目录。
macOS 可双击根目录 `打开训练监控.command`，默认读取根目录 `runs/`。
服务默认绑定回环地址，Ctrl+C 只关闭监控。

## 本机查看远程日志

先在服务器运行上面的日志服务，再在本机执行：

```bash
python3 scripts/monitor_remote.py --ssh-host your-ssh-alias --remote-port 6008 --open-browser
```

访问 `http://127.0.0.1:6009`；内部 SSH 转发端口默认 6012。
使用已有 `~/.ssh/config` 管理主机、端口和身份文件。默认别名为 `seetacloud-poker`，
可通过 `--ssh-host` 替换；仓库不包含 SSH 密钥或个人启动项。
macOS 可双击 `连接服务器监控.command` 使用默认别名。此入口运行当前 checkout 的代码，
不会读取某个固定用户的 Application Support 副本。若端口已被已有服务占用会直接报错，
请先关闭旧监控或通过 `--port` / `--tunnel-port` 选择其他端口。
网页服务先启动，SSH 断线自动重连；“日志已连接”不代表训练进程仍存活。

## 独立打包与复制部署

```bash
python3 scripts/package_monitor.py --output /tmp/overwatch-monitor.zip
```

把 ZIP 复制到目标机器并解压，进入 `overwatch-monitor/` 后按上面的命令启动。
包内提供三份服务端源码、打包脚本、全部五个前端文件、双击启动脚本和文档。
`MANIFEST.json` 记录每个文件的 SHA-256 和字节数；源文件清单在 `scripts/package_monitor.py`。
不会打包模型、训练数据、运行日志、SSH 文件、机器上的安装副本或整个 Git 仓库。
正式训练代码与 monitor 可以分别更新；Stage 2 完整动作轨迹由训练侧 writer 生成。

## 代码位置

- `scripts/monitor_training.py`：本地或服务器日志 HTTP 服务。
- `scripts/monitor_workbench.py`：实验、评测、归档及完整手牌回放的有界只读适配器。
- `scripts/monitor_remote.py`：本机常驻页面与 SSH 重连网关。
- `tools/training_monitor/`：HTML、JS 和两套主题 CSS。
- `docs/training_monitor.md`：日志契约、API、边界及历史安装说明。

Stage 2 的 `config.json`、`metrics.jsonl`、`stage2_evaluations/update_*.json` 均由训练器写入。
回放使用 `evaluations/<source>/report.json` 旁的 `hands.jsonl`。
更新时同步完整包，避免新 HTML 搭配旧后端；退出旧监控后再启动新版本。
