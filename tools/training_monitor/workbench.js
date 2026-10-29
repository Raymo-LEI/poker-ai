"use strict";

(() => {
  const isNumber = v => typeof v === "number" && Number.isFinite(v);
  const CATEGORIES = {training: "训练", branch: "分支", evaluation: "评测", smoke: "Smoke", benchmark: "Benchmark", other: "其他"};
  const BUCKETS = {sb_open_fold: "SB 开局弃牌", other_preflop_terminal: "其他翻前终局", hero_postflop_fold: "模型翻后弃牌",
    villain_postflop_fold: "对手翻后弃牌", showdown_hero_last_aggressor: "模型最后主动下注后摊牌",
    showdown_villain_last_aggressor_large_hero_call: "对手主动下注 / 模型大额跟注后摊牌", other_showdown: "其他摊牌"};
  const OPPONENT_LABELS = {random: "Random", calling_station: "Calling station", aggressive: "Aggressive", equity: "Pot odds equity", nit: "Tight passive nit", overbet: "Overbet", iter_1: "首轮策略", prev_checkpoint: "上一检查点"};

  function coordinates(rows, axis) {
    let elapsed = 0, samples = 0, timeKnown = true, samplesKnown = true;
    return rows.map(row => {
      const duration = row.iteration_s ?? (row.stage2 === 1 ? row.seconds : null);
      if (isNumber(duration)) elapsed += duration;
      else if (axis === "time") timeKnown = false;
      if (isNumber(row.p0_regret_samples) && isNumber(row.p1_regret_samples)) samples += row.p0_regret_samples + row.p1_regret_samples;
      else if (axis === "samples") samplesKnown = false;
      return {...row, originalStep: row.step,
        step: axis === "time" ? (timeKnown ? elapsed : null) : axis === "samples" ? (samplesKnown ? samples : null) : row.step};
    }).filter(row => isNumber(row.step));
  }

  function matchedEndpoints(datasets) {
    const curves = datasets.map(data => coordinates(data.rows, "time"));
    if (!curves.length || curves.some(rows => !rows.length)) return null;
    const budget = Math.min(...curves.map(rows => rows[rows.length - 1].step));
    return {budget, points: curves.map(rows => rows.filter(row => row.step <= budget).slice(-1)[0] || null)};
  }

  function freshness(recordedAt, now = Date.now() / 1000) {
    if (!isNumber(recordedAt)) return "unknown";
    return now - recordedAt > 90 ? "stale" : "recent";
  }

  function latestEvaluation(items) {
    const recorded = items.filter(e => !e.intervention && e.results?.some(r => isNumber(r.mbb_per_hand)));
    if (!recorded.length) return undefined;
    const step = Math.max(...recorded.map(e => e.step));
    return recorded.find(e => e.step === step && e.selection === "full") || recorded.find(e => e.step === step);
  }

  function stage2EvaluationCards(data, format) {
    const e = data.stage2_eval_latest || {}, number = (key,digits) => format(e[key],digits);
    return [
      ["独立评测 · Equity",number("stage2/eval/equity/mbb_per_hand",1),"mbb/hand",`评测 U=${e["stage2/eval/update"] ?? "—"}`],
      ["Equity 比赛胜率",format(isNumber(e["stage2/eval/equity/match_win"]) ? e["stage2/eval/equity/match_win"]*100 : null,1),"%","完整 50 手比赛；平局不算胜"],
      ["参考保留率测试",number("stage2/eval/reference/mbb_per_hand",1),"mbb/hand","候选对冻结学生参考"],
      ["Retention gate",isNumber(e["stage2/eval/retention_pass"]) ? e["stage2/eval/retention_pass"] === 1 ? "PASS" : "FAIL" : "—","","以评测器发布结果为准"]];
  }

  if (typeof module !== "undefined" && module.exports) module.exports = {coordinates, matchedEndpoints, freshness, latestEvaluation, stage2EvaluationCards};
  if (typeof document === "undefined") return;

  const api = window.Overwatch;
  const $ = id => document.getElementById(id);
  const {el, fmt, bytes, age, date, chart, mini} = api;
  let current = null, catalog = [], catalogAt = 0, catalogBusy = false, health = null;
  let selected = new Set(), comparison = [], comparisonAt = null, evalChoices = {};
  let tracePage = null, traceOffset = 0, hand = null, handStep = 0, traceGeneration = 0, handGeneration = 0;
  try { selected = new Set(JSON.parse(localStorage.getItem("overwatch-compare") || "[]").slice(0, 4)); } catch (_) { /* optional */ }

  function request(url) { return api.getJSON(url, new AbortController().signal); }
  function note(id, message, show = true) { $(id).textContent = message; $(id).classList.toggle("hidden", !show); }
  function human(value) {
    if (value === undefined || value === null) return "未记录";
    if (typeof value === "boolean") return value ? "开启" : "关闭";
    if (typeof value === "object") return JSON.stringify(value);
    return String(value);
  }
  function fact(label, value, sub = "") {
    const block = el("div", "wb-fact");
    block.append(el("div", "wb-fact-label", label), el("strong", "", value));
    if (sub) block.append(el("small", "", sub));
    return block;
  }
  function table(headers, rows) {
    const wrap = el("div", "table-scroll"), result = el("table"), head = el("thead"), tr = el("tr");
    headers.forEach(text => tr.append(el("th", "", text))); head.append(tr); result.append(head);
    const body = el("tbody");
    rows.forEach(values => {
      const row = el("tr");
      values.forEach(value => { const cell = el("td"); cell.append(value instanceof Node ? value : document.createTextNode(human(value))); row.append(cell); });
      body.append(row);
    });
    result.append(body); wrap.append(result); return wrap;
  }
  function button(label, action, secondary = true) {
    const node = el("button", secondary ? "button secondary" : "button", label); node.addEventListener("click", action); return node;
  }
  function option(value, label) { const node = el("option", "", label); node.value = value; return node; }
  function storeSelected() { try { localStorage.setItem("overwatch-compare", JSON.stringify([...selected])); } catch (_) { /* optional */ } }

  function route() {
    const key = location.hash.slice(1) || "overview";
    const mapping = {overview: ["overview", "summary", "workbench-home"], experiments: ["experiments"],
      stage2: ["stage2"], optimization: ["optimization"], strength: ["strength", "strategy"], strategy: ["strength", "strategy"],
      replay: ["replay"], assets: ["assets"], operations: ["operations", "performance", "details"],
      performance: ["operations", "performance", "details"], details: ["operations", "performance", "details"]};
    const visible = mapping[key] || mapping.overview;
    ["overview", "summary", "workbench-home", "stage2", "experiments", "optimization", "strength", "strategy", "replay", "assets", "operations", "performance", "details"].forEach(id => $(id).classList.toggle("route-hidden", !visible.includes(id)));
    const active = key === "strategy" ? "strength" : ["details", "performance"].includes(key) ? "operations" : mapping[key] ? key : "overview";
    document.querySelectorAll(".nav-link").forEach(link => link.classList.toggle("active", link.hash === "#" + active));
    if (key === "experiments") loadCatalog();
    if (key === "replay" && current && !tracePage && $("wb-trace-select").value) loadHands();
  }

  function stage2Facts(data) {
    const c = data.workbench?.effective || {}, policy = c.policy || {}, ppo = c.ppo || {};
    return [fact("冻结参考模型", human(c.reference), "学生参考路径来自实际配置；页面不加载权重"),
      fact("历史对手归档", human(c.strategies), "整场固定对手身份"),
      fact("残差幅度 α", human(policy.alpha), "ref_logits + α × tanh(delta)"),
      fact("KL β：前期 / 末期", `${human(policy.beta_kl)} / ${human(policy.late_beta_kl)}`, "按手进度调度；不是 update 轮次调度"),
      fact("PPO 配置", `${human(ppo.epochs)} epochs · batch ${human(ppo.minibatch_size)}`, `γ=${human(ppo.gamma)} · λ=${human(ppo.gae_lambda)}`),
      fact("每次更新 / 对手池", `${human(c.matches_per_update)} 场 · window ${human(c.pool?.window_size)}`, `实际配置权重 ${human(c.pool?.weights)}`)];
  }

  function drawStage2(data) {
    const s2 = data.workbench?.stage2 || {}, enabled = s2.enabled, r = data.latest || {};
    note("s2-note", enabled ? `${s2.scope} 最近 ${s2.window_updates || 0} 个更新、${s2.window_matches || 0} 场（按已记录采样数加权）。` : "Stage 2 面板已就绪。请在左侧选择 Stage 2 实验；训练写入 config.json 与 metrics.jsonl 后自动接入。当前选择仍是 Stage 1。");
    $("s2-config").replaceChildren(...(enabled ? stage2Facts(data) : []));
    const rows = enabled ? data.rows.filter(row => row.stage2 === 1).slice(-Number($("window").value)) : [];
    const pct = value => isNumber(value) ? fmt(value * 100, 1) : "—";
    mini("s2-cards", enabled ? [
      ["完整比赛 / 累计手数", `${fmt(r["stage2/matches"],0)} / ${fmt(r["stage2/hands"],0)}`, "", "训练日志计数 · 含续训历史"],
      [s2.block_updates ? "本次更新平均净收益" : "最近一场净收益", fmt(r["stage2/hero_net_bb"],2), "BB/场", s2.block_updates ? `混合对手：${Object.entries(s2.latest_sampled || {}).map(([name,n]) => `${name} × ${n}`).join(" · ")}` : `对手 ${s2.latest_opponent || "未记录"}`],
      ["最近窗口均值", fmt(s2.window_net_bb_mean,2), "BB/场", `最近 ${s2.window_matches || 0} 场 · 非独立评测`],
      ["窗口比赛胜场", fmt(s2.window_wins,0), "场", isNumber(s2.window_losses) ? `${s2.window_draws} 和 / ${s2.window_losses} 负` : "旧日志未记录平局数；不从更新均值推算胜负"],
      ["参考策略 KL", fmt(r["stage2/base_kl"],6), "nats", "冻结学生参考 · 非 exploitability"]] : []);
    chart("s2-kl", rows, [{key:"stage2/base_kl",label:"KL · epoch mean"},{key:"stage2/final_base_kl",label:"KL · after update"}], {zero:true});
    chart("s2-return", rows, [{key:"stage2/hero_net_bb",label:"50 手净 BB"}], {zero:true});
    chart("s2-beta", rows, [{key:"stage2/beta_kl",label:"平均 beta"},{key:"stage2/kl_penalty",label:"KL penalty"}], {zero:true});
    const names = ["equity","nit","call","maniac","overbet","push_fold","reference","history","self","TAG_Pro","Trapper_CR"];
    chart("s2-pfsp", rows, names.map(name => ({key:`stage2/pfsp/${name}`,label:name})), {zero:true});
    $("s2-opponents").replaceChildren(s2.block_updates ? table(["对手","本次实际采样","滚动窗口场数","窗口 mbb/hand","窗口比赛胜率"], (s2.opponents || []).map(o => [o.name,fmt(o.sampled,0),fmt(o.matches,0),fmt(o.mbb_per_hand,1),pct(o.match_win)+"%"])) : table(["对手","记录场数","平均净 BB / 场","胜 / 和 / 负"], enabled ? (s2.opponents || []).map(o => [o.name,fmt(o.matches,0),fmt(o.mean_net_bb,2),`${o.wins} / ${o.draws} / ${o.matches-o.wins-o.draws}`]) : []));
    mini("s2-eval-cards", enabled ? stage2EvaluationCards(data,fmt) : []);
    const evalRows = data.rows.filter(row => isNumber(row["stage2/eval/update"]));
    chart("s2-evaluation", enabled ? evalRows : [], ["equity","reference"].map(name => ({key:`stage2/eval/${name}/mbb_per_hand`,label:name+" · 独立评测 mbb/hand"})), {zero:true});
    const hud = [["vpip","VPIP"],["pfr","PFR"],["three_bet","3-bet"],["fold_to_preflop_raise","面对翻前加注 Fold"],["flop_aggression","Flop aggression"],["turn_aggression","Turn aggression"],["river_aggression","River aggression"],["fold_to_postflop_bet","面对翻后下注 Fold"],["all_in_frequency","All-in"],["showdown_win","摊牌胜率"],["check_raise_freq","Check-raise / opportunity"],["cbet_freq","Flop C-bet / opportunity"]];
    mini("s2-context", enabled ? hud.map(([key,label]) => [label,pct(r[`stage2/context/${key}`]),"%","最后一场对手 · 含先验"]) : []);
  }

  function drawEvents(id, events) {
    $(id).replaceChildren(...events.map(item => {
      const row = el("div", "wb-event");
      row.append(el("span", "wb-event-time", date(item.time)), el("strong", "", item.label), el("small", "", item.detail));
      return row;
    }));
    if (!events.length) $(id).append(el("p", "wb-description", "当前目录没有可展示的发布记录。"));
  }

  function renderRuntime(data) {
    const wb = data.workbench || {}, runtime = wb.runtime || {}, effective = wb.effective || {};
    const stateFresh = freshness(runtime.recorded_at), resourceFresh = freshness(runtime.resource_time);
    const step = runtime.training_step;
    const isStage2 = wb.stage2?.enabled;
    const target = isStage2 ? null : isNumber(effective.iterations) ? effective.iterations : null;
    const evaluations = (wb.evaluations || []).filter(e => e.results.some(r => isNumber(r.mbb_per_hand)));
    const evalStep = evaluations.length ? Math.max(...evaluations.map(e => e.step)) : null;
    const progress = (title, value, total, foot) => {
      const block = el("article", "wb-progress panel"); block.append(el("div", "wb-fact-label", title));
      const line = el("div", "wb-progress-value", isNumber(value) ? `${isStage2 ? "U" : "T"}=${fmt(value, 0)}` : "尚未记录");
      if (isNumber(total)) line.append(el("small", "", ` / ${fmt(total, 0)}`)); block.append(line);
      if (isNumber(value) && total > 0) { const track = el("div", "wb-progress-track"), bar = el("span"); bar.style.width = Math.min(100, value / total * 100) + "%"; track.append(bar); block.append(track); }
      block.append(el("p", "wb-description", foot)); return block;
    };
    let estimate = "完整训练日志中的最近完成轮次";
    const times = data.rows.map(r => r.iteration_s).filter(isNumber).slice(-10).sort((a, b) => a - b);
    if (times.length >= 5 && isNumber(step) && target > step) {
      const low = (target - step) * times[Math.floor(times.length * .25)] / 60;
      const high = (target - step) * times[Math.floor(times.length * .75)] / 60;
      estimate += ` · 近期耗时估算剩余 ${fmt(low, 0)}–${fmt(high, 0)} 分钟，不含额外评测`;
    }
    $("wb-progress").replaceChildren(progress("训练日志进度", step, target, estimate),
      progress("独立评测进度", evalStep, target, evaluations.length ? `${evaluations.length} 个已记录快照 / 变体 · 与训练进度分别计数` : "等待已完成的独立评测结果"));
    const stale = stateFresh === "stale";
    const message = stale ? `阶段文件已过期：${age(runtime.recorded_at)}，记录为“${runtime.phase_label}”${isNumber(runtime.source_iteration) ? ` / T=${runtime.source_iteration}` : ""}。当前训练日志为 T=${step ?? "—"}；该记录不能证明进程仍存活。`
      : stateFresh === "unknown" ? "当前目录没有阶段心跳。进程存活状态未知；日志存在、连接成功与任务存活分别判断。"
      : `阶段文件记录为“${runtime.phase_label}”，更新于 ${date(runtime.recorded_at)}。进程存活尚未独立核验。`;
    note("wb-state-note", message);
    $("wb-state-note").classList.toggle("wb-warning", stale);
    $("wb-resource-time").textContent = isNumber(runtime.resource_time) ? `${date(runtime.resource_time)} · ${resourceFresh === "stale" ? "历史记录，非实时值" : "最近采样值"}` : "未发现资源采样，保留为缺失";
    const resources = runtime.resources || {}, high = runtime.high_watermarks || {};
    $("wb-resource-cards").replaceChildren(
      fact("RAM", bytes(resources.ram_current_bytes), `采样高水位 ${bytes(high.ram_current_bytes)}`),
      fact("VRAM", bytes(resources.vram_used_bytes), `采样高水位 ${bytes(high.vram_used_bytes)}`),
      fact("GPU 利用率", isNumber(resources.gpu_utilization_percent) ? fmt(resources.gpu_utilization_percent, 0) + "%" : "未记录", resourceFresh === "stale" ? "旧采样，不代表当前负载" : "采样时刻的设备利用率"),
      fact("磁盘余量", bytes(resources.disk_free_bytes), "记录所在文件系统"));
    const power = effective.iteration_weight_power;
    $("wb-effective").replaceChildren(
      fact("可训练 Regret 参数", fmt(effective.regret_trainable_parameters, 0), effective.model?.d_model ? `d=${effective.model.d_model} · ${effective.model.num_layers} 层 · ${effective.model.nhead} heads` : "参数量需要有效配置记录"),
      fact("实际 Reservoir 容量", fmt(effective.effective_capacity, 0), `配置字段 ${fmt(effective.declared_capacity, 0)} · ${effective.capacity_basis || "未记录"}`),
      fact("每轮训练", isNumber(effective.train_steps) ? `${fmt(effective.train_steps, 0)} steps` : "未记录", `Batch ${fmt(effective.minibatch_size, 0)} · ${human(effective.loss)}`),
      fact("迭代权重", isNumber(power) ? `${effective.normalize_iteration_weights ? "(t / T)" : "t"}^${power}` : "未记录", `学习率 ${human(effective.learning_rate_schedule)}`),
      fact("Equity 特征", human(effective.equity_features), `SPR guard ${human(effective.spr_action_guard)}`),
      fact("脚本对手根树", isNumber(effective.scripted_opponent_prob) ? fmt(effective.scripted_opponent_prob * 100, 1) + "%" : "未记录", human(effective.scripted_opponent_pool)));
    if (isStage2) {
      $("wb-effective").replaceChildren(...stage2Facts(data));
      estimate = `${human(effective.matches_per_update)} 场完整 50 手比赛 / update · 配置本次追加 ${human(effective.updates)} 次更新；续训终点需运行元数据确认`;
      $("wb-progress").replaceChildren(progress("训练更新进度", step, null, estimate), progress("独立评测进度", evalStep, null, "训练净收益不会填入独立评测"));
    }
    $("wb-effective").append(fact("实验标识", data.experiment?.label || "未标记", data.experiment?.experiment_id || "旧记录需显式登记"));
    $("wb-config-source").textContent = `${effective.source || "配置来源未记录"} · 源码指纹文件数 ${effective.source_file_count || 0}`;
    drawEvents("wb-home-events", (wb.events || []).slice(0, 4));
    drawEvents("wb-events", wb.events || []);
    const resourceRows = runtime.resources_history || [];
    if (resourceRows.length) {
      const first = resourceRows[0].step;
      const rows = resourceRows.map(r => ({step: (r.step - first) / 60, ram: isNumber(r.ram_current_bytes) ? r.ram_current_bytes / 2**30 : null, vram: isNumber(r.vram_used_bytes) ? r.vram_used_bytes / 2**30 : null}));
      $("wb-resource-chart-scope").textContent = `来源 resources.jsonl · ${date(first)} 至 ${date(resourceRows[resourceRows.length-1].step)} · 横轴：自首条记录起分钟；纵轴：GiB`;
      chart("wb-resource-chart", rows, [{key: "ram", label: "RAM (GiB)"}, {key: "vram", label: "VRAM (GiB)"}], {zero: true});
    } else {
      $("wb-resource-chart-scope").textContent = "没有资源历史；页面不额外调用 GPU 采集";
      chart("wb-resource-chart", [], [], {emptyTitle: "未记录资源历史", emptyText: "接入现有 runner 的 resources.jsonl 后显示。"});
    }
    renderHealth();
  }

  function renderHealth() {
    const runtime = current?.workbench?.runtime || {};
    const success = health?.last_success_at;
    const connected = health?.log_service === "responding" || isNumber(success) && (!health.last_error_at || success >= health.last_error_at);
    $("wb-health").textContent = health ? `本机页面可用 · ${connected ? "最近日志请求成功" : health.last_error ? "数据连接待恢复" : "等待数据连接"}` : "正在检查连接";
    $("wb-connection-details").replaceChildren(
      fact("本机网页", health ? "可用" : "待检查", "常驻入口 127.0.0.1:6009"),
      fact("SSH 子进程", health?.ssh_process === "running" ? "运行中" : health?.ssh_process === "reconnecting" ? "重连中" : "本地日志服务", "子进程运行不等于训练存活"),
      fact("最近日志请求", connected ? "成功" : health?.last_error || "尚无记录", isNumber(success) ? date(success) : "以实际数据请求为准"),
      fact("训练进程", "尚未独立核验", `阶段记录：${runtime.phase_label || "未知"} · ${isNumber(runtime.recorded_at) ? age(runtime.recorded_at) : "无心跳"}`));
  }

  async function loadHealth() {
    try { health = await request("/api/health"); renderHealth(); } catch (_) { note("wb-health", "本机健康检查暂不可用"); }
  }

  async function loadCatalog(force = false) {
    if (catalogBusy || (!force && catalog.length && Date.now() - catalogAt < 30000)) { renderCatalog(); return; }
    catalogBusy = true;
    try {
      const data = await request("/api/catalog"); catalog = data.experiments; catalogAt = Date.now();
      selected = new Set([...selected].filter(id => catalog.some(item => item.id === id)));
      if (!selected.size && current) selected.add(current.id);
      renderCatalog(); note("wb-catalog-error", "", false);

    } catch (e) { note("wb-catalog-error", `${e.message}。保留已读取索引。`); }
    finally { catalogBusy = false; }
  }

  function renderCatalog() {
    const query = $("wb-search").value.trim().toLowerCase(), category = $("wb-category").value;
    const scoped = api.visibleRuns(catalog, api.getScope());
    $("wb-catalog-note").textContent = `${api.getScope() === "formal" ? "正式实验" : "全部记录"} ${scoped.length} 项 / 全部 ${catalog.length} 项 · 索引读取于 ${catalogAt ? new Date(catalogAt).toLocaleTimeString("zh-CN", {hour12:false}) : "等待索引"}`;
    selected = new Set([...selected].filter(id => scoped.some(item => item.id === id)));
    const rows = scoped.filter(item => (category === "all" || item.category === category) && `${item.name} ${JSON.stringify(item.effective)}`.toLowerCase().includes(query));
    $("wb-catalog").replaceChildren(...rows.map(item => {
      const row = el("tr"), chooseCell = el("td"), choose = el("input");
      choose.type = "checkbox"; choose.checked = selected.has(item.id); choose.setAttribute("aria-label", "对比 " + item.name);
      choose.addEventListener("change", () => {
        if (choose.checked && selected.size >= 4) { choose.checked = false; note("wb-catalog-error", "一次最多对比四个实验，请先取消一个选择。"); return; }
        choose.checked ? selected.add(item.id) : selected.delete(item.id); storeSelected(); renderCatalog();
      }); chooseCell.append(choose); row.append(chooseCell);
      const name = el("td", "wb-experiment-name"); name.append(el("strong", "", `[${item.experiment?.label || "未标记"}] ${item.experiment?.name || item.name}`));
      const effective = item.effective || {};
      name.append(el("small", "", `seed ${effective.seed ?? "—"} · ${effective.model?.d_model ? "d=" + effective.model.d_model : "模型未记录"} · 容量 ${fmt(effective.effective_capacity, 0)} · steps ${fmt(effective.train_steps, 0)}`));
      row.append(name, el("td", "", CATEGORIES[item.category] || "其他"), el("td", "", `${item.stage === "stage2" ? "S2 U=" : "T="}${item.last_step ?? "—"} / ${item.eval_step ?? "—"}`));
      const phase = el("td"); phase.append(el("span", "", item.reported_phase || "unknown"));
      phase.append(el("small", "", freshness(item.phase_recorded_at) === "stale" ? "记录已过期" : item.phase_recorded_at ? "最近记录" : "无阶段心跳")); row.append(phase);
      const action = el("td"); action.append(button("查看", async () => { await api.selectRun(item.id); location.hash = "overview"; route(); })); row.append(action); return row;
    }));
    $("wb-selected-count").textContent = `已选择 ${selected.size} / 4 个实验`;
    $("wb-compare").disabled = selected.size < 2;
    if (!rows.length) { const row = el("tr"), cell = el("td", "", "没有匹配的实验；可清空搜索或切换类型。"); cell.colSpan = 6; row.append(cell); $("wb-catalog").append(row); }
  }

  async function compare() {
    $("wb-compare").disabled = true; $("wb-compare").textContent = "正在读取对照…";
    try {
      comparison = await Promise.all([...selected].map(id => request("/api/run?id=" + encodeURIComponent(id))));
      comparisonAt = Date.now(); $("wb-comparison").classList.remove("hidden");
      drawComparison(); note("wb-catalog-error", "", false);
    } catch (e) { note("wb-catalog-error", `对照未更新：${e.message}`); }
    finally { $("wb-compare").textContent = "对比所选实验"; $("wb-compare").disabled = selected.size < 2; }
  }

  function chosenEvaluation(data) {
    const items = data.workbench?.evaluations || [];
    return items.find(e => e.id === evalChoices[data.id]) || latestEvaluation(items);
  }

  function drawComparison() {
    if (!comparison.length) return;
    const labels = comparison.map(data => data.name);
    const controls = comparison.map(data => {
      const label = el("label", "wb-candidate-choice", data.name), select = el("select"); select.setAttribute("aria-label", data.name + " 对照评测");
      const items = data.workbench?.evaluations || [];
      select.replaceChildren(...(items.length ? items.slice().reverse().map(e => option(e.id, api.evaluationLabel(e))) : [option("", "未记录评测")]));
      const chosen = chosenEvaluation(data); if (chosen) select.value = chosen.id;
      select.addEventListener("change", () => { evalChoices[data.id] = select.value; drawComparison(); }); label.append(select); return label;
    });
    $("wb-compare-choices").replaceChildren(...controls);
    const differing = ["rules_version", "numeric_version", "equity_features", "spr_action_guard", "preflop_action_guard", "iteration_weight_power", "normalize_iteration_weights"].filter(key => new Set(comparison.map(d => human(d.workbench?.effective?.[key]))).size > 1);
    const selections = comparison.map(chosenEvaluation);
    const mismatch = new Set(selections.map(e => e ? `${e.selection}:${e.seed}:${e.sampling_power}:${e.guard_override}` : "missing")).size > 1;
    $("wb-compare-notes").textContent = `快照读取于 ${new Date(comparisonAt).toLocaleTimeString("zh-CN", {hour12:false})}。${differing.length ? "配置口径存在差异或缺失：" + differing.join("、") + "。" : "请结合有效配置判断是否属于严格消融。"}${mismatch ? "评测策略或 seed 不同，不能直接作为配对差值。" : "跨实验差值须有匹配牌对证据。"} 不自动选出冠军。`;
    const axis = $("wb-compare-axis").value, metric = $("wb-compare-metric").value;
    const curves = comparison.map(data => coordinates(data.rows, axis));
    const points = new Map();
    curves.forEach((rows, i) => rows.forEach(row => { const merged = points.get(row.step) || {step: row.step}; if (isNumber(row[metric])) merged["candidate" + i] = row[metric]; points.set(row.step, merged); }));
    const axisLabel = axis === "time" ? "累计日志 iteration_s（秒）" : axis === "samples" ? "累计新增 regret 样本（条）" : "迭代轮次";
    $("wb-compare-chart-title").textContent = $("wb-compare-metric").selectedOptions[0].textContent;
    $("wb-compare-chart-scope").textContent = `来源：各实验日志 · 横轴：${axisLabel}；纵轴：${metric.includes("ratio") ? "比值" : "R²"}。缺失计时后不继续推算累计时间；保留训练探针不是独立验证集。`;
    // Retain each candidate's actual points; no interpolation creates a model.
    chart("wb-compare-chart", [...points.values()].sort((a,b) => a.step-b.step), comparison.map((data,i) => ({key:"candidate"+i,label:data.name})), {zero: metric.includes("r2"), connectMissing:true, emptyText:"所选实验没有这项共同诊断记录。"});
    const configFields = [
      ["训练 seed", "seed"], ["实际容量", "effective_capacity"], ["配置容量字段", "declared_capacity"], ["Regret 参数", "regret_trainable_parameters"],
      ["模型", "model"], ["训练 steps", "train_steps"], ["Batch", "minibatch_size"], ["Loss", "loss"], ["目标尺度", "regret_target_scale"],
      ["权重指数", "iteration_weight_power"], ["权重归一", "normalize_iteration_weights"], ["Equity 特征", "equity_features"],
      ["脚本根概率", "scripted_opponent_prob"], ["脚本池", "scripted_opponent_pool"], ["SPR guard", "spr_action_guard"],
      ["翻前 guard", "preflop_action_guard"], ["规则版本", "rules_version"], ["数值特征版本", "numeric_version"]];
    const diffs = configFields.map(([label,key]) => [label,...comparison.map(data => human(data.workbench?.effective?.[key]))]).filter(row => new Set(row.slice(1)).size > 1);
    $("wb-config-diff").replaceChildren(table(["字段（仅显示差异）",...labels], diffs.length ? diffs : [["已记录字段",...labels.map(() => "所选字段一致；缺失值不代表配置已核验")]]));
    const opponents = [...new Set(selections.flatMap(e => (e?.results || []).map(r => r.opponent)))];
    const matrixRows = opponents.map(name => [OPPONENT_LABELS[name] || name,...selections.map(e => {
      const result = e?.results.find(r => r.opponent === name);
      if (!result || !isNumber(result.mbb_per_hand)) return "未采集";
      const node = el("div", result.mbb_per_hand >= 0 ? "positive" : "negative", fmt(result.mbb_per_hand,1));
      node.append(el("small", "", result.ci95 ? `95% CI [${fmt(result.ci95[0],1)}, ${fmt(result.ci95[1],1)}]` : "未提供 CI"));
      node.append(el("small", "", `${fmt(result.pairs,0)} 对发牌${result.training_opponent ? " · 训练对手" : ""}`)); return node;
    })]);
    const commonNames = new Set(["random", "calling_station", "aggressive", "equity", "nit", "overbet", "slumbot"]);
    const matrix = matrixRows.filter((_,i) => commonNames.has(opponents[i]));
    $("wb-opponent-matrix").replaceChildren(table(["对手",...labels], matrix.length ? matrix : [["独立评测",...labels.map(() => "未采集")]]));
    const archiveRows = matrixRows.filter((_,i) => !commonNames.has(opponents[i]));
    if (archiveRows.length) {
      const detail = el("details", "panel wb-space"); detail.append(el("summary", "", "各自归档内部对照（对手随候选变化）"),
        el("p", "wb-description", "这里的首轮、上一检查点或 T=N 来自各候选自己的策略归档，不是同一个固定对手；不能按共同基线计算优劣。"),table(["各自归档中的对手",...labels],archiveRows));
      $("wb-opponent-matrix").append(detail);
    }
    const matched = axis === "time" ? matchedEndpoints(comparison) : null;
    $("wb-time-endpoints").replaceChildren();
    if (matched) $("wb-time-endpoints").append(el("p", "wb-description", `共同记录时间预算 ${fmt(matched.budget,1)} 秒；选取不超过该预算的最后一个完整日志点，不插值。此口径不是 GPU 账单时间，也不替代 Study 的正式同算时评测。`),
      table(["实验", "实际轮次", "累计记录时间（秒）"], comparison.map((d,i) => [d.name,matched.points[i]?.originalStep,matched.points[i]?.step])));
  }

  function drawAssets(data) {
    const assets = data.workbench?.assets || {};
    mini("wb-asset-cards", [
      ["训练恢复点文件", assets.checkpoint ? bytes(assets.checkpoint.bytes) : "未发现", "", assets.checkpoint ? date(assets.checkpoint.updated_at) : "不能从策略快照重建优化器和 Reservoir"],
      ["策略文件 / 清单条目", `${assets.present_snapshots ?? 0} / ${assets.advertised_snapshots ?? 0}`, "", `清单终点 T=${assets.through ?? "—"}`],
      ["1…T 连续覆盖", assets.history_complete ? "已覆盖" : "未确认 / 不连续", "", "仅名称与存在性检查，不是权重哈希验证"],
      ["完整压缩归档", String(assets.archives?.length || 0), "", "只展示已发现的归档文件元数据"]]);
    const gaps = [...new Set([...(assets.missing_files || []),...(assets.history_gaps || [])])];
    note("wb-asset-note", `${assets.verification || "未取得归档检查结果"}。引用清单${assets.references_match_file ? "与当前文件元数据匹配" : "未与当前文件核验匹配"}。${gaps.length ? "未覆盖轮次示例：" + gaps.join("、") + "；缺失历史不能凭当前权重补造。" : "未执行实际恢复测试。"}`);
    $("wb-asset-references").replaceChildren(table(["checkpoint 引用代次", "目录存在性"], (assets.generations || []).length ? assets.generations.map(g => [g.name,g.directory_present ? "目录存在，内容未验证" : "目录缺失 / 越界"]) : [["引用清单", "未记录；不能据文件存在断言可恢复"]]));
    $("wb-archives").replaceChildren(table(["压缩归档", "文件大小", "写入时间", "依赖与验证级别"], assets.archives?.length ? assets.archives.map(a => [a.name,bytes(a.bytes),date(a.updated_at),`${a.manifest_present ? "有 manifest" : "缺少 manifest"} · 对象 ${a.objects_present ?? 0}/${a.objects_checked ?? 0} 存在且大小匹配。${a.verification || "未验证"}${a.restore_requires ? "；需要 " + a.restore_requires : ""}`]) : [["完整归档", "未发现", "—", "未验证"]]));
  }

  function drawTraining(data) {
    const r = data.latest || {}, effective = data.workbench?.effective || {};
    $("wb-training-extra").classList.toggle("hidden", ["PPO", "Stage 2 PPO"].includes(data.kind));
    if (["PPO", "Stage 2 PPO"].includes(data.kind)) return;
    const sum = suffix => [r["p0_"+suffix],r["p1_"+suffix]].every(isNumber) ? r["p0_"+suffix]+r["p1_"+suffix] : null;
    const selfplay = sum("selfplay_roots"), scripted = sum("scripted_roots");
    const total = isNumber(selfplay) && isNumber(scripted) ? selfplay+scripted : null;
    const percent = count => total > 0 && isNumber(count) ? fmt(count/total*100,1) : "—";
    mini("wb-training-extra", [
      ["自博弈根树", percent(selfplay), "%", `${fmt(selfplay,0)} / ${fmt(total,0)} 棵实际根树`],
      ["Equity 根树", percent(sum("scripted_equity_roots")), "%", "按根树计数，不等于保留样本比例"],
      ["Nit 根树", percent(sum("scripted_nit_roots")), "%", `配置脚本根概率 ${isNumber(effective.scripted_opponent_prob) ? fmt(effective.scripted_opponent_prob*100,1)+"%" : "未记录"}`],
      ["学习率：首步 → 末步", `${human(r.learning_rate_first)} → ${human(r.learning_rate_last)}`, "", "当前轮实际记录；缺失时不推算"]]);
  }

  function drawEvaluationDetails(data) {
    const wb = data.workbench || {}, value = $("eval-select").value;
    const chosen = (wb.evaluations || []).find(e => "eval:"+e.id === value) || (wb.evaluations || []).find(e => e.step === data.eval_latest?.step && e.selection === "full" && !e.intervention) || (wb.evaluations || []).find(e => e.step === data.eval_latest?.step && !e.intervention);
    const container = $("wb-evaluation-details"); container.replaceChildren();
    if (chosen) {
      const panel = el("div", "wb-facts panel");
      panel.append(fact("策略选择", chosen.label, `T=${chosen.step}${chosen.intervention ? " · guard 干预" : ""}`),
        fact("评测 seed", human(chosen.seed), "不同 seed 不自动组成配对差值"),
        fact("训练对手", chosen.results.filter(r => r.training_opponent).map(r => OPPONENT_LABELS[r.opponent] || r.opponent).join(" / ") || "原报告未标记", "训练对手收益与额外对手分别解读"),
        fact("证据来源", chosen.source, chosen.format || "日志指标")); container.append(panel);
    }
    for (const [label,delta] of [["burn-in 相对 full 的已记录差值",wb.paired_delta],["Study 已记录配对差值",wb.study_paired_deltas]]) {
      if (!delta || !Object.keys(delta).length) continue;
      const details = el("details", "panel wb-space"), summary = el("summary", "", label), pre = el("pre", "", JSON.stringify(delta,null,2));
      details.append(summary,el("p", "wb-description", "原报告提供的配对统计；页面不从两个边际 CI 推导差值。"),pre); container.append(details);
    }
  }

  function setupTraces(data, changedRun) {
    const sources = data.workbench?.trace_sources || [], previous = changedRun ? "" : $("wb-trace-select").value;
    $("wb-trace-select").replaceChildren(...(sources.length ? sources.map(s => option(s.id, s.source)) : [option("", "当前实验无完整动作轨迹")]));
    if (sources.some(s => s.id === previous)) $("wb-trace-select").value = previous;
    $("wb-load-hands").disabled = !sources.length;
    if (changedRun || !sources.length) {
      tracePage = null; traceOffset = 0; hand = null; traceGeneration++;
      $("wb-hand-list").replaceChildren(); $("wb-hand-replay").replaceChildren(); $("wb-leak-buckets").replaceChildren();
      $("wb-hand-count").textContent = ""; $("wb-hand-title").textContent = "选择左侧一手牌";
      $("wb-bucket").replaceChildren(option("all", "全部线路"));
      $("wb-hands-prev").disabled = true; $("wb-hands-next").disabled = true;
    }
    if (!sources.length) note("wb-replay-note", "当前实验的已发现报告没有 hands.jsonl 动作轨迹。普通牌对汇总只能看收益，不能还原完整动作。本页不会启动补采或重训；可在实验库切换到已有诊断轨迹的实验。");
    else if (!tracePage) {
      const source = sources.find(s => s.id === $("wb-trace-select").value) || sources[0];
      note("wb-replay-note", `点击“读取轨迹”按需建立只读索引。最多索引每个文件前 64 MiB / 25,000 手。${source.scope ? ` 来源 ${source.scope} · ${source.status || "未记录完成状态"} · writer 记录 ${source.recorded_hands ?? "—"} 手${source.truncated ? `，已截断 / 丢弃 ${source.dropped_hands ?? "—"} 手` : ""}；此为采样子集。` : ""}`);
    }
  }

  async function loadHands(reset = false) {
    if (!current || !$("wb-trace-select").value) return;
    if (reset) traceOffset = 0;
    const generation = ++traceGeneration, runId = current.id;
    const query = new URLSearchParams({id:runId,trace:$("wb-trace-select").value,offset:String(traceOffset),limit:"20",seat:$("wb-seat").value,category:$("wb-bucket").value,order:$("wb-hand-order").value});
    $("wb-load-hands").disabled = true;
    try {
      const result = await request("/api/hands?"+query);
      if (generation !== traceGeneration || current.id !== runId) return;
      tracePage = result; drawHands();
    } catch(e) { if (generation === traceGeneration) note("wb-replay-note", `轨迹读取失败：${e.message}`); }
    finally { if (generation === traceGeneration) $("wb-load-hands").disabled = false; }
  }

  function drawHands() {
    const result = tracePage, selectedBucket = $("wb-bucket").value;
    $("wb-bucket").replaceChildren(option("all", "全部线路"),...Object.keys(result.buckets).map(key => option(key, BUCKETS[key] || key)));
    if ([...$("wb-bucket").options].some(o => o.value === selectedBucket)) $("wb-bucket").value = selectedBucket;
    const source = (current?.workbench?.trace_sources || []).find(s => s.id === $("wb-trace-select").value);
    const coverage = source?.scope ? ` 来源 ${source.scope} · ${source.status || "状态未记录"}${source.truncated ? ` · writer 已丢弃 ${source.dropped_hands ?? "—"} 手` : ""} · 抽样子集。` : "";
    note("wb-replay-note", `${result.scope}${coverage}。当前索引 ${fmt(result.indexed_hands,0)} 手${result.truncated ? "（受上限或未完成尾行限制，未覆盖全文件）" : ""}；跳过 ${result.skipped} 条无法使用记录。分桶统计针对该来源的全部已索引手数，不随下方位置筛选改变。`);
    $("wb-leak-buckets").replaceChildren(...Object.entries(result.buckets).map(([key,value]) => {
      const node = button(BUCKETS[key] || key, () => { $("wb-bucket").value = key; loadHands(true); }); node.className = "wb-bucket";
      node.append(el("strong", value.chips >= 0 ? "positive" : "negative", `${fmt(value.contribution_mbb_per_all_hands,1)} mbb / 全部手`),
        el("small", "", `n=${fmt(value.hands,0)} · 桶内均值 ${fmt(value.conditional_mbb_per_hand,1)} mbb/手`)); return node;
    }));
    $("wb-hand-count").textContent = result.filtered_hands ? `${traceOffset+1}–${Math.min(traceOffset+result.items.length,result.filtered_hands)} / ${result.filtered_hands}` : "0 / 0";
    $("wb-hand-list").replaceChildren(...result.items.map(row => {
      const node = button(`${row.hero_seat === 0 ? "BB" : "SB"} · ${fmt(row.payoff_chips / result.big_blind_chips,2)} BB · ${BUCKETS[row.category] || row.category}`, () => loadHand(row.record));
      node.className = "wb-hand-row"; node.append(el("small", "", `记录 #${row.record} · ${row.scope ? `U=${row.update ?? "—"} / match ${row.match ?? "—"}` : `σ${row.snapshot ?? "—"}`} · ${row.events} 个行动`)); return node;
    }));
    if (!result.items.length) $("wb-hand-list").append(el("p", "wb-description", "该筛选没有匹配对局。"));
    $("wb-hands-prev").disabled = traceOffset <= 0; $("wb-hands-next").disabled = traceOffset+20 >= result.filtered_hands;
  }

  async function loadHand(record) {
    const generation = traceGeneration, handRequest = ++handGeneration, runId = current.id;
    try {
      const result = await request("/api/hand?"+new URLSearchParams({id:runId,trace:$("wb-trace-select").value,record:String(record)}));
      if (generation !== traceGeneration || handRequest !== handGeneration || current.id !== runId) return;
      hand = result; handStep = 0; drawHand();
    } catch(e) { note("wb-replay-note", e.message); }
  }

  function drawHand() {
    if (!hand) return;
    const events = hand.events || [], currentEvent = events[handStep];
    $("wb-hand-title").textContent = `#${hand.record} · ${hand.hero_seat === 0 ? "BB" : "SB"} · σ${hand.snapshot ?? "—"} · 最终 ${fmt(hand.payoff_chips / hand.big_blind_chips,2)} BB`;
    const container = $("wb-hand-replay"); container.replaceChildren();
    const board = el("div", "wb-board");
    for (const card of currentEvent?.board || []) board.append(el("span", "wb-playing-card", card));
    if (!board.childNodes.length) board.append(el("span", "wb-description", "此时没有已记录的公共牌"));
    container.append(board);
    if (hand.scope) container.append(el("p", "wb-description", `${hand.scope} · U=${hand.update ?? "—"} · 对手 ${hand.opponent ?? "未记录"} · match ${hand.match ?? "—"} / pair ${hand.pair ?? "—"}`));
    if (handStep === events.length-1 && hand.terminal_board?.length) container.append(el("p", "wb-description", `结算后的公共牌：${hand.terminal_board.join(" ")}（不属于本行动之前的信息）`));
    const ownCards = hand.hero_hole || hand.hole_cards;
    container.append(el("p", "wb-description", ownCards ? `模型底牌：${human(ownCards)}` : "原轨迹未记录模型底牌；页面不从种子补算或编造。"));
    const controls = el("div", "wb-toolbar");
    const prev = button("上一步", () => { handStep--; drawHand(); }), next = button("下一步", () => { handStep++; drawHand(); });
    prev.disabled = handStep <= 0; next.disabled = handStep >= events.length-1;
    const range = el("input"); range.type = "range"; range.min = "0"; range.max = String(Math.max(0,events.length-1)); range.value = String(handStep); range.setAttribute("aria-label", "回放行动步数");
    range.addEventListener("input", () => { handStep = Number(range.value); drawHand(); }); controls.append(prev,range,next); container.append(controls);
    if (currentEvent) {
      const actions = ["Fold", "Check / Call", "½ Pot", "Pot", "All-in"], streets = ["Preflop", "Flop", "Turn", "River"];
      container.append(el("h3", "", `${handStep+1} / ${events.length} · ${streets[currentEvent.street] || currentEvent.street} · P${currentEvent.seat} ${currentEvent.seat === hand.hero_seat ? "模型" : "对手"} · ${actions[currentEvent.action] || currentEvent.action}`));
      const recordedBB = value => isNumber(value) ? `${fmt(value / hand.big_blind_chips,2)} BB` : "未记录";
      const facts = el("div", "wb-facts"); facts.append(fact("行动前底池", recordedBB(currentEvent.pot)), fact("面对跟注额", recordedBB(currentEvent.to_call)), fact("本次投入", recordedBB(currentEvent.chips_added))); container.append(facts);
    }
    if (hand.post_hand_oracle_audit && Object.keys(hand.post_hand_oracle_audit).length) {
      const audit = el("details", "wb-space"); audit.append(el("summary", "", "事后审计（不属于当时可见信息）"), el("p", "wb-description", "对实际底牌的事后 equity 不是对手范围 equity，不是动作错误或诈唬的证明。"),el("pre", "", JSON.stringify(hand.post_hand_oracle_audit,null,2))); container.append(audit);
    }
  }

  function exportComparison() {
    if (!comparison.length) return;
    const lines = ["# Overwatch 实验对照", "", `读取时间：${new Date(comparisonAt).toISOString()}`, "", $("wb-compare-notes").textContent, ""];
    for (const data of comparison) {
      const evaluation = chosenEvaluation(data);
      lines.push(`## ${data.name}`, "", `目录：${data.path}`, `最新训练日志：T=${data.workbench?.runtime?.training_step ?? "未记录"}`, "", "有效配置：", "```json",JSON.stringify(data.workbench?.effective || {},null,2),"```", "");
      if (evaluation) { lines.push(`评测：T=${evaluation.step} / ${evaluation.label} / seed=${evaluation.seed ?? "未记录"}`,`来源：${evaluation.source}`,"");
        for (const result of evaluation.results) lines.push(`- ${result.opponent}: ${result.mbb_per_hand ?? "未采集"} mbb/hand; CI=${JSON.stringify(result.ci95)}; pairs=${result.pairs ?? "未记录"}${result.training_opponent ? "; 训练对手" : ""}`);
      }
      lines.push("", "证据文件：",...data.sources.map(s => `- ${s.name} (${s.bytes} bytes)`), "");
    }
    lines.push("本报告保留原测量，不自动推导配对差值、反事实 EV 或 GTO 结论。");
    const url = URL.createObjectURL(new Blob([lines.join("\n")],{type:"text/markdown;charset=utf-8"}));
    const link = el("a"); link.href = url; link.download = "overwatch-comparison.md"; link.click(); setTimeout(() => URL.revokeObjectURL(url),1000);
  }

  function render(data) {
    const changedRun = current?.id !== data.id; current = data;
    renderRuntime(data); drawStage2(data); drawAssets(data); drawTraining(data); drawEvaluationDetails(data); setupTraces(data, changedRun);
    if (!catalog.length) loadCatalog();
    route();
  }

  $("wb-search").addEventListener("input", renderCatalog);
  $("wb-category").addEventListener("change", renderCatalog);
  $("wb-compare").addEventListener("click", compare);
  $("wb-compare-axis").addEventListener("change", drawComparison);
  $("wb-compare-metric").addEventListener("change", drawComparison);
  $("wb-report-export").addEventListener("click", exportComparison);
  $("wb-load-hands").addEventListener("click", () => loadHands(true));
  $("wb-trace-select").addEventListener("change", () => { tracePage = null; loadHands(true); });
  ["wb-seat","wb-bucket","wb-hand-order"].forEach(id => $(id).addEventListener("change", () => loadHands(true)));
  $("wb-hands-prev").addEventListener("click", () => { traceOffset = Math.max(0,traceOffset-20); loadHands(); });
  $("wb-hands-next").addEventListener("click", () => { traceOffset += 20; loadHands(); });
  window.addEventListener("hashchange", () => { route(); window.scrollTo({top:0}); });
  window.OverwatchWorkbench = {render, scopeChanged: () => {
    comparison = []; $("wb-comparison").classList.add("hidden"); renderCatalog();
  }};
  route(); loadHealth(); if (api.getCurrent()) render(api.getCurrent());
  setInterval(() => {
    if (document.hidden || !$("auto-refresh").checked) return;
    loadHealth(); if (current) renderRuntime(current);
    if (location.hash === "#experiments") loadCatalog();
  }, 15000);
})();
