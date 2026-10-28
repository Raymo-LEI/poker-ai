"use strict";

const $ = id => document.getElementById(id);
const COLORS = ["#c1ec91", "#74c7bb", "#e7ba79", "#9baee9", "#ee9c87", "#d4b9dc"];
const OPPONENTS = [
  ["Random", "random"], ["Calling station", "calling_station"],
  ["Aggressive", "aggressive"], ["Slumbot", "slumbot"],
  ["首轮策略", "iter_1"], ["上一检查点", "prev_checkpoint"],
  ["Overbet", "overbet"], ["Tight passive nit", "nit"], ["Pot odds equity", "equity"],
];
const ACTIONS = ["fold", "call", "half_pot", "pot", "all_in"];
const ACTION_LABELS = ["Fold", "Check / Call", "½ Pot", "Pot", "All-in"];
let current = null;
let busy = false;
let generation = 0;
let activeRequest = null;
let refreshTimer = null;
let failures = 0;
const responseCache = new Map();

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function finite(v) { return typeof v === "number" && Number.isFinite(v); }
function fmt(v, digits = 2) {
  if (!finite(v)) return "—";
  return new Intl.NumberFormat("en-US", {maximumFractionDigits: digits}).format(v);
}
function short(v) {
  if (!finite(v)) return "—";
  const a = Math.abs(v);
  if (a >= 1e9) return fmt(v / 1e9, 1) + "B";
  if (a >= 1e6) return fmt(v / 1e6, 1) + "M";
  if (a >= 1e3) return fmt(v / 1e3, 1) + "k";
  return fmt(v, a < 1 ? 3 : 1);
}
function bytes(v) {
  if (!finite(v)) return "—";
  for (const [size, unit] of [[2 ** 30, "GiB"], [2 ** 20, "MiB"], [1024, "KiB"]]) {
    if (v >= size) return fmt(v / size, 1) + " " + unit;
  }
  return fmt(v, 0) + " B";
}
function date(v) { return finite(v) ? new Date(v * 1000).toLocaleString("zh-CN", {hour12: false}) : "—"; }
function age(v) {
  if (!finite(v)) return "—";
  const seconds = Math.max(0, Date.now() / 1000 - v);
  if (seconds < 60) return "不到 1 分钟前";
  if (seconds < 3600) return Math.floor(seconds / 60) + " 分钟前";
  if (seconds < 86400) return Math.floor(seconds / 3600) + " 小时前";
  return Math.floor(seconds / 86400) + " 天前";
}
function card(label, value, unit, sub, primary = false, mini = false) {
  const node = el("div", mini ? "mini-card" : "metric-card" + (primary ? " primary" : ""));
  const title = el("div", "metric-label", label);
  if (!mini) title.append(el("span", "metric-symbol", primary ? "↗" : "◦"));
  const content = el("div", "metric-value", value);
  if (unit) content.append(el("small", "", unit));
  node.append(title, content, el("div", "metric-sub", sub || "未采集"));
  return node;
}
function mini(id, entries) {
  $(id).replaceChildren(...entries.map(([label, value, unit, sub]) => card(label, value, unit, sub, false, true)));
}
function emptyChart(id, title, text) {
  const container = $(id);
  container.className = "chart empty";
  const state = el("div", "empty-state");
  state.append(el("div", "empty-icon", "⌁"), el("strong", "", title), el("p", "", text));
  container.replaceChildren(state);
}
function svgNode(tag, attrs = {}, text) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
  if (text !== undefined) node.textContent = text;
  return node;
}
function legend(entries) {
  const wrap = el("div", "legend chart-legend");
  for (const [name, color] of entries) {
    const item = el("span", "legend-item");
    const dot = el("i"); dot.style.background = color;
    item.append(dot, document.createTextNode(name)); wrap.append(item);
  }
  return wrap;
}

function chart(id, rows, series, options = {}) {
  const available = series.filter(s => rows.some(r => finite(r[s.key])));
  if (!available.length) {
    emptyChart(id, options.emptyTitle || "尚无这项记录", options.emptyText || "日志写入对应指标后，曲线会自动出现在这里。");
    return;
  }
  const points = rows.filter(r => available.some(s => finite(r[s.key])));
  const width = 540, height = 208, left = 49, right = 15, top = 19, bottom = 30;
  const plotW = width - left - right, plotH = height - top - bottom;
  const xs = points.map(r => r.step);
  let xmin = Math.min(...xs), xmax = Math.max(...xs);
  if (xmin === xmax) { xmin -= .5; xmax += .5; }
  const values = [];
  for (const row of points) for (const s of available) {
    if (!finite(row[s.key])) continue;
    values.push(row[s.key]);
    if (s.lower && finite(row[s.lower])) values.push(row[s.lower]);
    if (s.upper && finite(row[s.upper])) values.push(row[s.upper]);
  }
  let ymin = Math.min(...values), ymax = Math.max(...values);
  if (options.zero) { ymin = Math.min(0, ymin); ymax = Math.max(0, ymax); }
  const pad = (ymax - ymin || Math.max(Math.abs(ymax) * .1, 1)) * .12;
  ymin -= pad; ymax += pad;
  const x = v => left + (v - xmin) / (xmax - xmin) * plotW;
  const y = v => top + (ymax - v) / (ymax - ymin) * plotH;
  const svg = svgNode("svg", {viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": available.map(s => s.label).join(" / ") + " 趋势图"});
  for (let i = 0; i <= 4; i++) {
    const v = ymin + (ymax - ymin) * i / 4;
    svg.append(svgNode("line", {x1: left, x2: width - right, y1: y(v), y2: y(v), class: "chart-grid"}));
    svg.append(svgNode("text", {x: left - 9, y: y(v) + 3, "text-anchor": "end", class: "chart-axis"}, short(v)));
  }
  const ticks = points.length === 1 ? [points[0].step] : [...new Set(Array.from({length: 5}, (_, i) => Math.round(Math.min(...xs) + (Math.max(...xs) - Math.min(...xs)) * i / 4)))];
  for (const v of ticks) svg.append(svgNode("text", {x: x(v), y: height - 10, "text-anchor": "middle", class: "chart-axis"}, fmt(v, 0)));
  if (ymin < 0 && ymax > 0 && options.zero) svg.append(svgNode("line", {x1: left, x2: width - right, y1: y(0), y2: y(0), stroke: "#6b8263", "stroke-dasharray": "3 4", "stroke-width": .7}));
  for (const s of available) {
    const color = s.color || COLORS[series.indexOf(s) % COLORS.length];
    if (s.lower && s.upper) {
      let band = [];
      const flush = () => {
        if (!band.length) return;
        const d = band.map((r, i) => `${i ? "L" : "M"}${x(r.step)},${y(r[s.upper])}`).join(" ") + " " + [...band].reverse().map(r => `L${x(r.step)},${y(r[s.lower])}`).join(" ") + " Z";
        svg.append(svgNode("path", {d, fill: color, opacity: .1})); band = [];
      };
      for (const row of rows) {
        if (finite(row[s.key]) && finite(row[s.lower]) && finite(row[s.upper])) band.push(row); else flush();
      }
      flush();
    }
    let d = "", pen = false;
    for (const row of rows) {
      if (!finite(row[s.key])) { pen = false; continue; }
      d += `${pen ? " L" : " M"}${x(row.step)},${y(row[s.key])}`;
      pen = true;
    }
    svg.append(svgNode("path", {d, stroke: color, fill: "none", "stroke-width": 1.8, "stroke-linejoin": "round", "stroke-linecap": "round"}));
    const visible = points.filter(r => finite(r[s.key]));
    if (visible.length <= 30) for (const row of visible) {
      const circle = svgNode("circle", {cx: x(row.step), cy: y(row[s.key]), r: 2.6, fill: color, stroke: "#182019", "stroke-width": 1.5});
      circle.append(svgNode("title", {}, `#${row.step} ${s.label}: ${fmt(row[s.key], 4)}`)); svg.append(circle);
      if (s.lower && s.upper && finite(row[s.lower]) && finite(row[s.upper])) {
        svg.append(svgNode("line", {x1: x(row.step), x2: x(row.step), y1: y(row[s.lower]), y2: y(row[s.upper]), stroke: color, opacity: .5, "stroke-width": 1}));
      }
    }
  }
  const crosshair = svgNode("line", {x1: 0, x2: 0, y1: top, y2: height - bottom, class: "chart-crosshair", visibility: "hidden"});
  svg.append(crosshair);
  const hit = svgNode("rect", {x: left, y: top, width: plotW, height: plotH, class: "chart-hit"});
  svg.append(hit);
  const container = $(id); container.className = "chart";
  const tip = el("div", "tooltip-card hidden");
  hit.addEventListener("pointermove", event => {
    const rect = svg.getBoundingClientRect();
    const coordinate = (event.clientX - rect.left) / rect.width * width;
    const nearest = points.reduce((a, b) => Math.abs(x(a.step) - coordinate) < Math.abs(x(b.step) - coordinate) ? a : b);
    crosshair.setAttribute("x1", x(nearest.step)); crosshair.setAttribute("x2", x(nearest.step)); crosshair.setAttribute("visibility", "visible");
    tip.replaceChildren(el("div", "tooltip-step", (current?.step_key || "step") + " #" + nearest.step));
    for (const s of available) {
      if (!finite(nearest[s.key])) continue;
      tip.append(el("div", "", `${s.label}  ${fmt(nearest[s.key], 4)}`));
      if (s.lower && s.upper && finite(nearest[s.lower]) && finite(nearest[s.upper])) tip.append(el("div", "", `95% CI [${fmt(nearest[s.lower])}, ${fmt(nearest[s.upper])}]`));
    }
    tip.classList.remove("hidden");
    const box = container.getBoundingClientRect();
    tip.style.left = Math.max(4, Math.min(event.clientX - box.left + 12, box.width - tip.offsetWidth - 8)) + "px";
    tip.style.top = Math.max(4, event.clientY - box.top - tip.offsetHeight - 12) + "px";
  });
  hit.addEventListener("pointerleave", () => { tip.classList.add("hidden"); crosshair.setAttribute("visibility", "hidden"); });
  container.replaceChildren(svg, legend(available.map(s => [s.label, s.color || COLORS[series.indexOf(s) % COLORS.length]])), tip);
}

function evaluationView(data, requested = "latest") {
  const hasEvaluation = row => Object.entries(row).some(([key, value]) => key.startsWith("eval/") && finite(value));
  const hasHud = row => Object.entries(row).some(([key, value]) => key.startsWith("hud/") && key !== "hud/rollout_legal_entropy" && finite(value));
  const snapshots = data.rows.filter(row => hasEvaluation(row) || hasHud(row));
  const selected = snapshots.find(row => String(row.step) === requested);
  return {snapshots, history: data.rows.filter(hasEvaluation), selection: selected ? requested : "latest",
    evaluation: selected ? (hasEvaluation(selected) ? selected : {}) : data.eval_latest || {},
    hud: selected ? (hasHud(selected) ? selected : {}) : data.hud_latest || {}};
}

function renderStage2(data, enabled) {
  $("stage2").classList.toggle("hidden", !enabled);
  $("stage2-nav").classList.toggle("hidden", !enabled);
  if (!enabled) return;
  const latest = data.latest || {}, rows = data.rows;
  mini("stage2-cards", [
    ["完整比赛", fmt(latest["stage2/matches"], 0), "场", "每场固定同一对手 50 手"],
    ["训练手数", fmt(latest["stage2/hands"], 0), "手", "不作为独立评测胜率"],
    ["KL(final || ref)", fmt(latest["stage2/base_kl"], 6), "nats", "冻结 Stage 1 参考策略"],
    ["KL 系数", fmt(latest["stage2/beta_kl"], 3), "", "前 10 手较强锚定"],
  ]);
  chart("stage2-kl-chart", rows, [{key:"stage2/base_kl", label:"KL to reference"}, {key:"stage2/kl_penalty", label:"KL penalty"}]);
  chart("stage2-return-chart", rows, [{key:"stage2/hero_net_bb", label:"Match net BB"}], {zero:true});
  const opponents = [["equity","Pot odds equity"],["nit","Tight passive nit"],["call","Calling station"],["maniac","Aggressive"],["overbet","Overbet"],["history","Stage 1 history"]];
  chart("stage2-pfsp-chart", rows, opponents.map(([key,label]) => ({key:`stage2/pfsp/${key}`,label})));
  const fields = [["vpip","VPIP"],["pfr","PFR"],["three_bet","3-Bet"],["fold_to_preflop_raise","Fold to preflop raise"], ["flop_aggression","Flop aggression"],["turn_aggression","Turn aggression"],["river_aggression","River aggression"],["fold_to_postflop_bet","Fold to postflop bet"],["all_in_frequency","All-in frequency"],["showdown_win","Showdown win"]];
  mini("stage2-context", fields.map(([key,label]) => [label, fmt(latest[`stage2/context/${key}`] * 100, 1), "%", "本场公开行动 · Bayesian smoothing"]));
}

function render(data) {
  const view = evaluationView(data, current?.id === data.id ? $("eval-select").value : "latest");
  current = data;
  const latest = data.latest || {}, recentEvaluation = data.eval_latest || {}, probe = data.diagnostic_latest || {};
  const {evaluation, hud} = view;
  $("eval-select").replaceChildren(...[["latest", "最新评测"], ...view.snapshots.slice().reverse().map(row => [String(row.step), `T=${row.step}`])].map(([value, label]) => {
    const option = el("option", "", label); option.value = value; return option;
  }));
  $("eval-select").value = view.selection;
  const isStage2 = data.kind === "Stage 2 PPO";
  const isPPO = data.kind === "PPO" || isStage2;
  renderStage2(data, isStage2);
  const shared = data.config?.separate_player_models === false || finite(latest.shared_train_steps);
  const rows = data.rows.slice(-Number($("window").value));
  $("run-name").textContent = data.name;
  $("run-kind").textContent = data.kind;
  $("run-device").textContent = data.environment?.gpu || data.environment?.device || data.config?.device || "设备未记录";
  $("run-age").textContent = "最后写入 " + age(data.updated_at) + " · " + date(data.updated_at);
  const lastStep = latest.step;
  const evalKey = "eval/winrate_vs_random_mbb";
  const checkpoint = data.checkpoint;
  $("summary").replaceChildren(
    card(data.evaluation_only ? "评测策略迭代 / ITERATION" : isPPO ? "当前更新 / UPDATE" : "当前迭代 / ITERATION", fmt(lastStep, 0), "", `${data.rows.length} 条合并记录 · ${data.kind}${data.evaluation_only ? " 评测" : ""}`, true),
    card("最新单轮耗时", fmt(latest.iteration_s ?? latest.seconds), "s", "日志中的完整轮次耗时"),
    card("对 Random 胜率", fmt(recentEvaluation[evalKey], 1), "mbb/h", finite(recentEvaluation[evalKey]) ? `评测于 #${recentEvaluation.step} · ${finite(recentEvaluation[evalKey + "_ci_low"]) && finite(recentEvaluation[evalKey + "_ci_high"]) ? "有 95% CI" : "未提供置信区间"}` : "等待独立对战评测"),
    card("最近训练 Checkpoint", checkpoint ? bytes(checkpoint.bytes) : "—", "", checkpoint ? date(checkpoint.updated_at) : "未发现本目录 train.pt")
  );
  $("loss-title").textContent = isPPO ? "策略与价值损失" : "Regret 回归损失";
  $("loss-description").textContent = isPPO ? "PPO policy loss / value loss" : shared ? `迭代加权 ${data.config?.loss === "huber" ? "Smooth L1" : "MSE"} · 共享位置条件网络 · Target / ${data.config?.regret_target_scale || 100} BB` : "迭代加权 MSE · 两个独立玩家";
  chart("loss-chart", rows, isPPO ? [{key: "train/policy_loss", label: "Policy loss"}, {key: "train/value_loss", label: "Value loss"}] : shared ? [{key: "shared_loss", label: "共享 Loss"}] : [{key: "train/loss_p0", label: "P0 loss"}, {key: "train/loss_p1", label: "P1 loss"}]);
  $("buffer-title").textContent = isPPO ? "累计训练步数" : "样本与训练进度";
  $("buffer-description").textContent = isPPO ? "Timesteps · 实际采集步数" : "Reservoir 中保留的 regret 样本";
  $("buffer-unit").textContent = isPPO ? "TIMESTEPS" : "SAMPLES";
  $("buffer-foot").textContent = isPPO ? "训练回报与独立对战胜率分别展示。" : shared ? "统一 Reservoir 按座位显示保留量；训练每批精确抽取 50% BB / 50% SB。" : "标准 HUNL：P0 = BB，P1 = SB / Button；外采样不保证双方样本数相等。";
  chart("buffer-chart", rows, isPPO ? [{key: "timesteps", label: "Timesteps"}] : [{key: "p0_buffer_size", label: "P0 buffer"}, {key: "p1_buffer_size", label: "P1 buffer"}]);
  mini("optimization-extra", isPPO ? [
    ["策略熵", fmt(latest["train/entropy"], 4), "nats", "PPO entropy"],
    ["Approx. KL", fmt(latest["train/approx_kl"], 5), "", "策略更新幅度"],
    ["Clip fraction", finite(latest["train/clip_fraction"]) ? fmt(latest["train/clip_fraction"] * 100, 1) : "—", "%", "裁剪样本比例"],
    ["Explained variance", fmt(latest["train/explained_variance"], 4), "", "价值估计诊断"],
  ] : [
    ["P0 / BB 加权 R²", fmt(probe["train/probe_r2_p0"], 4), "", finite(probe.step) ? `固定训练探针 · T=${probe.step}` : "等待诊断记录"],
    ["P1 / SB 加权 R²", fmt(probe["train/probe_r2_p1"], 4), "", finite(probe.step) ? `固定训练探针 · T=${probe.step}` : "等待诊断记录"],
    ["P0 End / Start", fmt(probe["train/probe_loss_ratio_p0"], 4), "", "历史起始 Loss 未记录时留空"],
    ["P1 End / Start", fmt(probe["train/probe_loss_ratio_p1"], 4), "", "两次测量使用同一个批次"],
    ...(finite(latest["study/fixed_probe_source_iteration"]) ? [
      ["共同固定探针 BB / SB R²", fmt(latest["train/fixed_probe_r2_p0"], 4) + " / " + fmt(latest["train/fixed_probe_r2_p1"], 4), "", `源 T=${latest["study/fixed_probe_source_iteration"]} · 跨分支同一批训练样本`],
      ["共同固定探针 BB / SB 归一 MSE", fmt(latest["train/fixed_probe_normalized_mse_p0"], 4) + " / " + fmt(latest["train/fixed_probe_normalized_mse_p1"], 4), "", "按样本轮次权重总和归一 · 非独立验证集"],
      ["分支训练计时", fmt(latest["study/branch_wall_seconds"], 1), "s", "自共同起点 · 包含固定探针和周期 checkpoint"],
    ] : []),
    ["Buffer P0 / P1", finite(latest.p0_buffer_size) && latest.p1_buffer_size > 0 ? fmt(latest.p0_buffer_size / latest.p1_buffer_size, 2) : "—", "×", "累计保留样本，非对局数"],
    ["P0 新样本 / Root", data.config?.traversals_per_player > 0 ? fmt(latest.p0_regret_samples / data.config.traversals_per_player, 2) : "—", "", `本轮样本 ${fmt(latest.p0_regret_samples, 0)}`],
    ["P1 新样本 / Root", data.config?.traversals_per_player > 0 ? fmt(latest.p1_regret_samples / data.config.traversals_per_player, 2) : "—", "", `本轮样本 ${fmt(latest.p1_regret_samples, 0)}`],
    [shared ? "共享网络更新次数" : "单轮拟合预算 P0 / P1", shared ? fmt(latest.shared_train_steps, 0) : fmt(latest.p0_train_steps, 0) + " / " + fmt(latest.p1_train_steps, 0), "", shared ? "每轮一个训练循环 · Warm-start" : "两个独立训练循环"],
    ...(shared ? [["训练抽样 BB / SB", fmt(latest.p0_training_draws, 0) + " / " + fmt(latest.p1_training_draws, 0), "", latest.fit_deferred_missing_seat ? "缺少座位样本，已延迟拟合" : "每个 Mini-batch 精确 1:1"], ["拟合验收 BB / SB", (latest.p0_fit_acceptance ? "PASS" : "未达标") + " / " + (latest.p1_fit_acceptance ? "PASS" : "未达标"), "", "两席均需 R² > 0.50 且 End/Start < 0.30"]] : []),
  ]);
  const probeMetric = $("probe-metric").value;
  chart("probe-chart", rows, [0, 1].map(p => ({key: `train/${probeMetric}_p${p}`, label: p === 0 ? "P0 / BB" : "P1 / SB"})), {zero: probeMetric.endsWith("r2"), emptyText: probeMetric === "probe_loss_ratio" ? "历史训练未记录起点。新训练代码在拟合前后测量同一批次；Checkpoint 回测只提供 End / Zero 和 R²。" : "训练边界或离线 Checkpoint 探针写入后显示。"});
  chart("samples-chart", rows, [{key: "p0_regret_samples", label: "P0 / BB"}, {key: "p1_regret_samples", label: "P1 / SB"}]);
  $("probe-note").textContent = finite(probe.step) ? `最近 T=${probe.step} · 每人 ${fmt(probe["train/probe_samples_p0"], 0)} / ${fmt(probe["train/probe_samples_p1"], 0)} 条保留训练样本` : "等待固定训练探针；历史曲线不伪造补值";
  $("eval-tag").textContent = Object.keys(evaluation).length ? "已记录" : "未采集";
  $("eval-tag").className = "tag" + (Object.keys(evaluation).length ? " good" : "");
  $("eval-step").textContent = finite(evaluation.step) ? (evaluation["diagnostic/eval_average_policy"] === 1 ? `平均策略 σ1…σ${evaluation.step}` : "评测快照 · T=" + evaluation.step) : "所选轮次未采集对战评测";
  $("hud-eval-note").textContent = finite(hud["hud/eval_hands"]) ? `T=${hud.step} · ${fmt(hud["hud/eval_hands"], 0)} 手自博弈` : view.selection !== "latest" ? `T=${view.selection} 未采集动作统计` : "仅展示实际采集的决策统计";
  const winrateRows = view.history.filter(r => OPPONENTS.some(([, k]) => finite(r[`eval/winrate_vs_${k}_mbb`])));
  $("eval-history-note").textContent = winrateRows.length ? `全部 ${winrateRows.length} 次评测 · T=${winrateRows.map(r => r.step).join(" / ")}。右侧快照和 C 面板随评测轮次切换；训练显示范围不裁剪此曲线。` : "等待独立对战评测；不会从训练 Loss 推算。";
  chart("winrate-chart", winrateRows, OPPONENTS.map(([label, k]) => ({key: `eval/winrate_vs_${k}_mbb`, label, lower: `eval/winrate_vs_${k}_mbb_ci_low`, upper: `eval/winrate_vs_${k}_mbb_ci_high`})), {zero: true, emptyTitle: "棋力尚未评测", emptyText: "当前日志只有训练数据。写入真实的 eval/ 指标后，这里显示胜率和置信区间。"});
  $("opponents").replaceChildren(...OPPONENTS.map(([label, k]) => {
    const key = `eval/winrate_vs_${k}_mbb`, value = evaluation[key];
    const row = el("div", "opponent-row");
    row.append(el("span", "opponent-name", label));
    const content = el("div", "opponent-value " + (!finite(value) ? "missing" : value >= 0 ? "positive" : "negative"), finite(value) ? (value > 0 ? "+" : "") + fmt(value, 1) : "未采集");
    if (finite(evaluation[key + "_ci_low"]) && finite(evaluation[key + "_ci_high"])) content.append(el("small", "", `95% CI [${fmt(evaluation[key + "_ci_low"], 1)}, ${fmt(evaluation[key + "_ci_high"], 1)}]`));
    else if (finite(value)) content.append(el("small", "", "未提供置信区间"));
    if (finite(evaluation[key + "_pairs"])) content.append(el("small", "", `${fmt(evaluation[key + "_pairs"], 0)} 对发牌 · ${fmt(evaluation[key + "_pairs"] * 2, 0)} 手`));
    row.append(content); return row;
  }));
  mini("eval-extra", [
    ["SB / Button 胜率", fmt(evaluation["eval/positional_sb_mbb"], 1), "mbb/h", "按位置单独统计"],
    ["BB 胜率", fmt(evaluation["eval/positional_bb_mbb"], 1), "mbb/h", "按位置单独统计"],
    [finite(evaluation["eval/botzone_50hand_match_win_rate"]) ? "Botzone 50 手比赛胜率" : "标准 HUNL 50 手比赛胜率", finite(evaluation["eval/botzone_50hand_match_win_rate"] ?? evaluation["eval/hunl_50hand_match_win_rate_vs_random"]) ? fmt((evaluation["eval/botzone_50hand_match_win_rate"] ?? evaluation["eval/hunl_50hand_match_win_rate_vs_random"]) * 100, 1) : "—", "%", finite(evaluation["eval/botzone_50hand_match_win_rate"]) ? "20,000 筹码 · 50 / 100 盲注" : "对手 Random · 标准 HUNL 规则"],
    ["评测所处轮次", fmt(evaluation.step, 0), "", "评测可能晚于或少于训练日志"],
  ]);
  const hudFields = [["SB VPIP", "preflop_vpip_sb", "preflop_vpip_sb_hands"], ["SB PFR", "preflop_pfr_sb", "preflop_pfr_sb_hands"], ["BB VPIP", "preflop_vpip_bb", "preflop_vpip_bb_hands"], ["BB 3-Bet", "preflop_3bet_bb", "preflop_3bet_bb_opportunities"]];
  $("hud-context").classList.toggle("hidden", isPPO);
  $("hud-context").textContent = (shared ? "共享位置网络：P0 / P1 是 BB / SB 座位统计，共用同一个网络。" : "双网络：P0 对应 BB，P1 对应 SB。") + (data.config?.preflop_action_guard ? " 翻前 guard 开启：深筹码开局 All-in 受动作约束，0% 不代表模型自主学会不推。" : " 翻前 guard 未启用。");
  mini("hud-cards", [...hudFields.map(([label, key, denominator]) => {
    const measured = finite(hud["hud/" + key]);
    const count = hud["hud/" + denominator];
    return [label, measured ? fmt(hud["hud/" + key] * 100, 1) : count === 0 ? "无机会" : "未采集", measured ? "%" : "", finite(count) ? `T=${hud.step} · ${fmt(count, 0)} ${key.includes("3bet") ? "次可再加注机会" : "手已发底牌"}` : "等待评测轨迹统计"];
  }),
    ...[0, 1].flatMap(p => ["fold", "all_in"].map(action => [`P${p} / ${p === 0 ? "BB" : "SB"} 翻前 ${action === "fold" ? "Fold" : "All-in"}`, finite(hud[`hud/action_freq_p${p}_preflop/${action}`]) ? fmt(hud[`hud/action_freq_p${p}_preflop/${action}`] * 100, 1) : "—", "%", `T=${hud.step ?? "—"} · ${fmt(hud[`hud/p${p}_preflop_decisions`], 0)} 次翻前决策`])),
    ["SB 开局推 200BB", finite(hud["hud/sb_open_all_in"]) ? fmt(hud["hud/sb_open_all_in"] * 100, 1) : "—", "%", "分母为 SB 首次行动机会"],
    ["BB 面对开局加注 Fold", finite(hud["hud/bb_fold_to_open_raise"]) ? fmt(hud["hud/bb_fold_to_open_raise"] * 100, 1) : "—", "%", "分母为首次面对 SB 加注的机会"],
    ["合法行动策略熵", fmt(hud["hud/policy_entropy_mean"], 3), "nats", "实际行动分布，包含配置的 fallback"],
    ["正 Regret 行动数", fmt(hud["hud/positive_regret_actions_mean"], 2), "/ 5", finite(hud["hud/positive_regret_decisions"]) ? `${fmt(hud["hud/positive_regret_decisions"], 0)} 次决策 · 仅统计 guard 后合法行动` : "等待 raw regret 采集"],
    ...(isPPO ? [
      ["Rollout legal entropy", fmt(latest["hud/rollout_legal_entropy"], 3), "nats", "PPO rollout 统计，独立于评测 HUD"],
      ["免费 Check 时 Fold", finite(latest.free_check_fold_ratio) ? fmt(latest.free_check_fold_ratio * 100, 1) : "—", "%", "PPO rollout 决策统计"],
    ] : hud["hud/free_check_opportunities"] > 0 ? [
      ["免费 Check 时 Fold · 评测", finite(hud["hud/free_check_fold_ratio"]) ? fmt(hud["hud/free_check_fold_ratio"] * 100, 1) : hud["hud/free_check_opportunities"] === 0 ? "无机会" : "未采集", finite(hud["hud/free_check_fold_ratio"]) ? "%" : "", finite(hud["hud/free_check_opportunities"]) ? `T=${hud.step} · ${fmt(hud["hud/free_check_opportunities"], 0)} 次非强制免费 Check 机会` : "等待评测轨迹统计"],
    ] : []),
  ]);
  $("action-legend").replaceChildren(...legend(ACTION_LABELS.map((v, i) => [v, COLORS[i]])).childNodes);
  $("street-bars").replaceChildren(...["preflop", "flop", "turn", "river"].flatMap((street, index) => [0, 1].map(player => {
    const block = el("div", "street-block");
    const values = ACTIONS.map(a => hud[`hud/action_freq_p${player}_${street}/${a}`]);
    const complete = values.every(v => finite(v) && v >= 0 && v <= 1);
    const total = values.reduce((sum, v) => sum + (finite(v) ? v : 0), 0);
    const valid = complete && Math.abs(total - 1) < .02;
    const top = el("div", "street-top"); top.append(el("span", "", ["Preflop · 翻前", "Flop · 翻牌", "Turn · 转牌", "River · 河牌"][index] + ` · P${player} / ${player === 0 ? "BB" : "SB"}`), el("span", "", valid ? `T=${hud.step} · n=${fmt(hud[`hud/p${player}_${street}_decisions`], 0)}` : "未采集完整分布"));
    const bar = el("div", "stacked-bar"), detail = el("div", "street-values");
    if (valid) values.forEach((v, i) => {
      const part = el("span"); part.style.width = (v / total * 100) + "%"; part.style.background = COLORS[i]; part.title = `${ACTION_LABELS[i]}: ${fmt(v * 100, 1)}%`; bar.append(part);
      detail.append(el("span", "", `${ACTION_LABELS[i]} ${fmt(v * 100, 1)}%`));
    });
    else detail.textContent = complete ? "分布总和异常，请检查记录口径" : "等待 hud/action_freq_ 指标";
    block.append(top, bar, detail); return block;
  })));
  chart("timing-chart", rows, isPPO ? [{key: "seconds", label: "Update time"}] : [{key: "p0_collect_s", label: "P0 采样"}, {key: "p1_collect_s", label: "P1 采样"}, {key: "shared_fit_s", label: "共享拟合"}, {key: "p0_fit_s", label: "P0 拟合"}, {key: "p1_fit_s", label: "P1 拟合"}]);
  chart("throughput-chart", rows, [{key: "system/regret_samples_per_s", label: "Regret samples / s"}], {emptyText: isPPO ? "该项为 CFR regret 采样吞吐；PPO 日志无对应数据。" : "需要同一轮的 regret 样本数与采样耗时。"});
  mini("performance-extra", [
    ["平均推理 Batch", fmt(latest["system/average_inference_batch"], 2), "rows", "总推理行数 / 总调用数"],
    [shared ? "共享训练吞吐" : "P0 训练吞吐", fmt(shared ? latest.shared_training_draws_per_s : latest.p0_training_draws_per_s, 1), "/ s", "训练样本抽取速率"],
    [shared ? "训练 Batch" : "P1 训练吞吐", fmt(shared ? data.config?.minibatch_size : latest.p1_training_draws_per_s, shared ? 0 : 1), shared ? "rows" : "/ s", shared ? "BB / SB 各占一半" : "训练样本抽取速率"],
    ["本轮 Checkpoint 耗时", fmt(latest.checkpoint_s, 3), "s", "当前轮未保存时留空"],
  ]);
  $("sources").replaceChildren(...data.sources.map(source => {
    const row = el("tr");
    [source.name, bytes(source.bytes), date(source.updated_at), "只读"].forEach(value => row.append(el("td", "", value)));
    return row;
  }));
  $("config").textContent = JSON.stringify({environment: data.environment, config: data.config, strategies: data.strategies, evaluation_notes: data.evaluation_notes}, null, 2);
  $("notices").replaceChildren(...data.warnings.map(w => el("div", "notice", w)));
  if (!data.rows.length) $("notices").prepend(el("div", "notice", "当前实验尚无完整日志记录，等待训练写入。"));
  $("history-note").textContent = `最多展示最近 ${fmt(data.history_limit, 0)} 条 · 只读`;
  $("export").disabled = false;
}

async function getJSON(url, signal) {
  const controller = new AbortController();
  const cancel = () => controller.abort(signal.reason);
  signal.addEventListener("abort", cancel, {once: true});
  if (signal.aborted) cancel();
  const timeout = setTimeout(() => controller.abort(new DOMException("请求超时", "TimeoutError")), 45000);
  const cached = responseCache.get(url);
  try {
    const response = await fetch(url, {cache: "no-store", signal: controller.signal,
      headers: cached ? {"If-None-Match": cached.etag} : {}});
    if (response.status === 304 && cached) return cached.data;
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    const etag = response.headers.get("ETag");
    if (etag) {
      responseCache.delete(url);
      responseCache.set(url, {etag, data});
      if (responseCache.size > 8) responseCache.delete(responseCache.keys().next().value);
    }
    return data;
  } catch (e) {
    if (controller.signal.reason?.name === "TimeoutError") {
      throw new Error(`读取${url.startsWith("/api/runs") ? "实验列表" : "实验数据"}超过 45 秒，服务器或 SSH 链路暂时无响应`);
    }
    throw e;
  } finally {
    clearTimeout(timeout);
    signal.removeEventListener("abort", cancel);
  }
}
function scheduleRefresh() {
  clearTimeout(refreshTimer);
  if ($("auto-refresh").checked && !document.hidden) {
    refreshTimer = setTimeout(() => refresh(), Math.min(60000, 5000 * 2 ** failures));
  }
}
function error(message) {
  $("error").textContent = message; $("error").classList.remove("hidden");
  $("connection").textContent = "连接异常"; $("connection").classList.add("offline");
}
async function refresh(force = false) {
  if (busy && !force) return;
  clearTimeout(refreshTimer);
  activeRequest?.abort();
  activeRequest = new AbortController();
  const signal = activeRequest.signal;
  busy = true;
  const request = ++generation;
  $("refresh").disabled = true;
  try {
    const data = await getJSON("/api/runs", signal);
    if (request !== generation) return;
    const selected = $("run-select").value;
    let remembered = null;
    try { remembered = localStorage.getItem("overwatch-run"); } catch (_) { /* optional */ }
    const ident = data.runs.some(r => r.id === selected) ? selected : data.runs.some(r => r.id === remembered) ? remembered : data.runs[0]?.id;
    $("run-select").replaceChildren(...data.runs.map(run => {
      const option = el("option", "", run.name); option.value = run.id; return option;
    }));
    if (!ident) {
      $("run-select").append(el("option", "", "暂无训练实验"));
      current = null;
      $("export").disabled = true;
      render({name: "暂无训练实验", kind: "SD-CFR", rows: [], latest: {}, eval_latest: {}, hud_latest: {}, config: {}, environment: {}, sources: [], warnings: ["监控目录里还没有日志。启动训练，或使用 --runs-dir 指向实际训练输出目录。"], history_limit: 5000});
      current = null;
      $("export").disabled = true;
    } else {
      $("run-select").value = ident;
      try { localStorage.setItem("overwatch-run", ident); } catch (_) { /* optional */ }
      const run = await getJSON("/api/run?id=" + encodeURIComponent(ident), signal);
      if (request !== generation) return;
      if (current !== run) render(run);
    }
    $("error").classList.add("hidden"); $("connection").classList.remove("offline");
    $("connection").textContent = (data.source || "日志服务") + "已连接";
    $("refreshed").textContent = "读取于 " + new Date().toLocaleTimeString("zh-CN", {hour12: false});
    failures = 0;
  } catch (e) {
    if (request === generation && !signal.aborted) {
      failures = Math.min(failures + 1, 4);
      const retry = $("auto-refresh").checked ? `；${Math.min(60, 5 * 2 ** failures)} 秒后自动重试` : "；可点右上角刷新重试";
      error(`${e.message}${retry}。页面保留上次成功读取的数据。`);
    }
  } finally {
    if (request === generation) { busy = false; $("refresh").disabled = false; scheduleRefresh(); }
  }
}

$("refresh").addEventListener("click", () => refresh(true));
$("run-select").addEventListener("change", () => refresh(true));
$("window").addEventListener("change", () => { if (current) render(current); });
$("probe-metric").addEventListener("change", () => { if (current) render(current); });
$("eval-select").addEventListener("change", () => { if (current) render(current); });
$("auto-refresh").addEventListener("change", () => { clearTimeout(refreshTimer); if ($("auto-refresh").checked) refresh(); });
document.addEventListener("visibilitychange", () => {
  clearTimeout(refreshTimer);
  if (!document.hidden && $("auto-refresh").checked) refresh();
});
$("expand-config").addEventListener("click", () => {
  const show = $("config").classList.contains("hidden");
  $("config").classList.toggle("hidden", !show);
  $("expand-config").setAttribute("aria-expanded", String(show));
  $("expand-config").textContent = show ? "收起配置 −" : "展开配置 ＋";
});
$("export").addEventListener("click", () => {
  if (!current) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(current, null, 2)], {type: "application/json"}));
  const link = el("a"); link.href = url; link.download = current.name + "-monitor.json"; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
document.querySelectorAll(".nav-link").forEach(link => link.addEventListener("click", () => {
  document.querySelectorAll(".nav-link").forEach(node => node.classList.toggle("active", node === link));
}));
const observer = new IntersectionObserver(entries => {
  const visible = entries.filter(e => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
  if (visible) document.querySelectorAll(".nav-link").forEach(link => link.classList.toggle("active", link.hash === "#" + visible.target.id));
}, {rootMargin: "-5% 0px -65% 0px", threshold: 0});
["overview", "strength", "strategy", "performance", "stage2", "details"].forEach(id => observer.observe($(id)));
$("export").disabled = true;
refresh();
