# Step 1：复现 AlphaNLHoldem 基线评测

## 目标

在 **50bb 单挑无限注**（上游 rlcard 环境，非 ACPC 100bb）下，用公开发布的权重 `c_1048.pkl` 跑通推理，并验证对 **随机机器人** 与 **简单规则机器人** 的胜率（mbb/hand + 95% 置信区间）。

## 为何用 Git Submodule（`third_party/AlphaNLHoldem`）

| 方案 | 取舍 |
|------|------|
| **Submodule（选用）** | 保留上游 AGPL 代码边界清晰、diff 小、权重 `weights/c_1048.pkl` 随上游提交；`git submodule update --init` 即可拉取。 |
| Vendor 拷贝 | 离线/CI 更简单，但仓库体积大、与上游 diff 难追踪。 |

本仓库为私有仓库，在私有环境内使用 AGPL 上游符合许可说明（见根目录 README）。

## 环境

上游原始依赖较老（TensorFlow 1.15、Ray 0.8.3、Python 3.7）。在 2026 年裸 `pip install -r requirements.txt` 会因 **protobuf / gym / importlib-metadata** 等传递依赖升级而失败。本仓库提供 **`requirements-alphanl.txt`**（在 upstream 基础上增加兼容 pin）。

推荐（与评测机一致）：

```bash
git submodule update --init --recursive

# Miniconda + Python 3.7（系统若无 3.7）
# curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh && bash miniconda.sh -b -p $PWD/.miniconda
# $PWD/.miniconda/bin/conda create -y -n alphanl python=3.7
# source $PWD/.miniconda/bin/activate alphanl

pip install -r requirements-alphanl.txt
```

- **CPU**：默认即可（云 agent / 笔记本均可）。
- **CUDA（可选，如 RTX 3070）**：`eval/eval_step1.py --cuda`；需本机 TensorFlow 1.15 GPU 与驱动匹配【未在 CI 中验证 GPU 路径】。

## 代码布局

| 路径 | 作用 |
|------|------|
| `third_party/AlphaNLHoldem/` | 上游子模块（环境、网络、权重，**未改网络结构**） |
| `src/alphanl/inference.py` | 加载 `NlHoldemNet` + `NNAgent`，贪心出牌 |
| `src/alphanl/bots.py` | `RandomBot`、`RuleBot`（本仓库新增，非上游） |
| `src/alphanl/matchup.py` | 对局循环；座位平衡逻辑对齐 `agi.evaluation_tools.death_match` |
| `eval/eval_step1.py` | CLI 评测入口 |

### 指标

- 筹码盈亏来自 rlcard `get_payoffs()`；大盲 **BB = 2** 筹码（100 筹码起始 ≈ 50bb）。
- **mbb/hand** = 每手平均盈亏（以大盲为单位）× 1000。
- 95% CI：正态近似，\( \bar{x} \pm 1.96 \cdot s / \sqrt{n} \)（mbb 尺度）。
- 默认 `--hands 2000`：与上游 `death_match` 相同，**1000 手** hero 映射到 seat0 视角 + **1000 手** 交换映射（共 2000 手）。

## 运行评测

```bash
export PYTHONPATH=src
python eval/eval_step1.py --hands 2000 --seed 42
# 可选：python eval/eval_step1.py --hands 2000 --progress --cuda
```

## 复现结果（本环境）

运行环境：Linux，Python 3.7.16，`requirements-alphanl.txt`，CPU，`--hands 2000`，`--seed 42`，权重 `third_party/AlphaNLHoldem/weights/c_1048.pkl`。

| 对手 | mbb/hand | 95% CI | n |
|------|----------|--------|---|
| Random | **1661.8** | [941.6, 2381.9] | 2000 |
| Rule | **1618.0** | [811.8, 2424.2] | 2000 |

随机种子 42；Alpha 显著为正，满足 Step 1「打过随机 / 规则机器人」目标。

## 未验证 / 已知限制

1. **规则引擎**：仍使用上游 rlcard NLHE（作者自述 pot/行动顺序等问题）；本步未切换到 pokerkit。
2. **权重**：子模块内 `c_1048.pkl`；Google Drive 全量 league 未下载【未核对是否与 c_1048 一致】。
3. **GPU 推理**：`--cuda` 路径未在云端 VM 实测。
4. **RuleBot**：本仓库启发式实现，强度有限，仅作「强于纯随机」的冒烟对手。上表 Rule 一行是旧版测得的：旧版在翻牌、转牌补假牌 `S2` 凑满 7 张再评牌力，会凭空多出对子、顺子、同花（例如翻牌任何高牌都算一对）。现已改为只评真实牌，这一行需要在 Python 3.7 环境重跑。
5. **训练 / 改网**：本步未进行。

## 参考

- 上游评测：`third_party/AlphaNLHoldem/agi/evaluation_tools.py`（`death_match`）
- 路线说明：根目录 `README.md`、`docs/reference_survey.md`
