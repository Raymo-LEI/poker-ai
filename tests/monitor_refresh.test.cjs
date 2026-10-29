const {test} = require('node:test');
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const vm = require('node:vm');
const source = readFileSync(require('node:path').join(__dirname, '../tools/training_monitor/app.js'), 'utf8');

function client(fetch) {
  const timers = new Map();
  let id = 0;
  const context = vm.createContext({fetch, AbortController, DOMException,
    document: {hidden: false}, $: () => ({checked: true}),
    setTimeout: (callback, delay) => {timers.set(++id, {callback, delay}); return id;},
    clearTimeout: key => timers.delete(key)});
  vm.runInContext('const responseCache = new Map(); let refreshTimer; let failures = 0; function refresh() {}\n'
    + source.slice(source.indexOf('async function getJSON'), source.indexOf('function error(')), context);
  return {context, timers, get: (url, signal = new AbortController().signal) => context.getJSON(url, signal)};
}

test('304 reuses parsed data, keeps run caches separate, and accepts changed ETags', async () => {
  const calls = [];
  const replies = [
    {status: 200, etag: 'W/"a"', data: {step: 1}},
    {status: 304},
    {status: 200, etag: 'W/"b"', data: {step: 2}},
    {status: 200, etag: 'W/"c"', data: {step: 3}},
  ];
  const c = client(async (url, options) => {
    calls.push({url, options});
    const reply = replies.shift();
    return {...reply, ok: reply.status === 200, headers: {get: () => reply.etag},
      json: async () => {assert.notEqual(reply.status, 304); return reply.data;}};
  });
  const first = await c.get('/api/run?id=a');
  assert.equal(await c.get('/api/run?id=a'), first);
  assert.equal(calls[1].options.headers['If-None-Match'], 'W/"a"');
  assert.equal((await c.get('/api/run?id=b')).step, 2);
  assert.equal(calls[2].options.headers['If-None-Match'], undefined);
  assert.equal((await c.get('/api/run?id=a')).step, 3);
  assert.equal(c.timers.size, 0);
});

test('deadline covers the response body, and a cancelled refresh aborts the fetch', async () => {
  const c = client(async (_, {signal}) => ({status: 200, ok: true,
    headers: {get: () => null}, json: () => new Promise((_, reject) => {
      signal.addEventListener('abort', () => reject(signal.reason));
    })}));
  const pending = c.get('/api/run?id=a');
  await new Promise(resolve => setImmediate(resolve));
  const timer = [...c.timers.values()][0];
  assert.equal(timer.delay, 45000);
  timer.callback();
  await assert.rejects(pending, /45 秒/);
  const aborter = new AbortController();
  const cancelled = c.get('/api/run?id=b', aborter.signal);
  await new Promise(resolve => setImmediate(resolve));
  aborter.abort();
  await assert.rejects(cancelled, {name: 'AbortError'});
  assert.equal(c.timers.size, 0);
});

test('failed refreshes back off, and background tabs do not schedule polling', () => {
  const c = client(() => {});
  vm.runInContext('failures = 3; scheduleRefresh()', c.context);
  assert.equal([...c.timers.values()][0].delay, 40000);
  vm.runInContext('document.hidden = true; scheduleRefresh()', c.context);
  assert.equal(c.timers.size, 0);
});

test('evaluation history is independent of training window and snapshots do not mix HUD rounds', () => {
  const context = vm.createContext({finite: v => typeof v === 'number' && Number.isFinite(v)});
  vm.runInContext(source.slice(source.indexOf('function evaluationView'), source.indexOf('function render(')), context);
  const rows = Array.from({length: 250}, (_, i) => ({step: i + 1}));
  for (const step of [100, 150, 200, 250]) rows[step - 1]['eval/winrate_vs_random_mbb'] = step;
  rows[99]['hud/eval_hands'] = 4000;
  rows[249]['hud/eval_hands'] = 10000;
  const data = {rows, eval_latest: rows[249], hud_latest: rows[249]};
  const view = context.evaluationView(data, '100');
  assert.deepEqual(Array.from(view.history, row => row.step), [100, 150, 200, 250]);
  assert.equal(view.evaluation.step, 100);
  assert.equal(view.hud['hud/eval_hands'], 4000);
  const missingHud = context.evaluationView(data, '150');
  assert.equal(missingHud.evaluation.step, 150);
  assert.equal(Object.keys(missingHud.hud).length, 0);
  const missingRound = context.evaluationView(data, '500');
  assert.equal(missingRound.selection, 'latest');
  assert.equal(missingRound.evaluation.step, 250);
});

test('Stage 2 independent cards preserve real zero and failed retention without rollout fallback', () => {
  const {stage2EvaluationCards} = require('../tools/training_monitor/workbench.js');
  const format = v => typeof v === 'number' && Number.isFinite(v) ? String(v) : '—';
  const cards = stage2EvaluationCards({latest:{'stage2/hero_net_bb':9999},stage2_eval_latest:{
    'stage2/eval/update':100,'stage2/eval/equity/mbb_per_hand':0,
    'stage2/eval/equity/match_win':.6,'stage2/eval/reference/mbb_per_hand':-21,
    'stage2/eval/retention_pass':0}},format);
  assert.equal(cards[0][1],'0');
  assert.equal(cards[1][1],'60');
  assert.equal(cards[2][1],'-21');
  assert.equal(cards[3][1],'FAIL');
  assert.equal(stage2EvaluationCards({latest:{'stage2/hero_net_bb':9999}},format)[0][1],'—');
});

test('Stage 2 pro evaluation cards and series use recorded independent scores and intervals', () => {
  const {stage2EvaluationCards,stage2EvaluationSeries} = require('../tools/training_monitor/workbench.js');
  const format = v => typeof v === 'number' && Number.isFinite(v) ? String(v) : '—';
  const evaluation = {'stage2/eval/update':100,
    'stage2/eval/TAG_Pro/mbb_per_hand':-441.8,'stage2/eval/TAG_Pro/match_win':.42,
    'stage2/eval/TAG_Pro/hands':5000,
    'eval/winrate_vs_TAG_Pro_mbb_ci_low':-914.3,'eval/winrate_vs_TAG_Pro_mbb_ci_high':30.7,
    'stage2/eval/Trapper_CR/mbb_per_hand':0,'stage2/eval/Trapper_CR/match_win':0,
    'stage2/eval/Trapper_CR/mbb_ci95_low':-100,'stage2/eval/Trapper_CR/mbb_ci95_high':0,
    'stage2/eval/future_pro/mbb_per_hand':12};
  const cards = stage2EvaluationCards({latest:{'stage2/window/TAG_Pro/mbb_per_hand':9999},stage2_eval_latest:evaluation},format);
  const tag = cards.find(row=>row[0].includes('TAG_Pro'));
  const trap = cards.find(row=>row[0].includes('Trapper_CR'));
  assert.equal(tag[1],'-441.8');
  assert.match(tag[3],/95% CI \[-914.3, 30.7\]/);
  assert.match(tag[3],/5000 手/);
  assert.equal(trap[1],'0');
  assert.match(trap[3],/比赛胜率 0%/);
  assert.match(trap[3],/95% CI \[-100, 0\]/);
  const keys = stage2EvaluationSeries([evaluation]).map(row=>row.key);
  assert.ok(keys.includes('stage2/eval/TAG_Pro/mbb_per_hand'));
  assert.ok(keys.includes('stage2/eval/Trapper_CR/mbb_per_hand'));
  assert.ok(keys.includes('stage2/eval/future_pro/mbb_per_hand'));
  assert.ok(!keys.includes('stage2/eval/self/mbb_per_hand'));
  const missing = stage2EvaluationCards({latest:{'stage2/window/TAG_Pro/mbb_per_hand':9999}},format);
  assert.equal(missing.find(row=>row[0].includes('TAG_Pro'))[1],'—');
});

test('formal selector excludes unmarked, smoke and forged truthy flags by default', () => {
  const context = vm.createContext({});
  vm.runInContext(source.slice(source.indexOf('function visibleRuns'),source.indexOf('function experimentScope')),context);
  const runs = [{id:'formal',experiment:{is_formal:true}}, {id:'smoke',experiment:{is_formal:false}},
    {id:'old'}, {id:'forged',experiment:{is_formal:'true'}}];
  assert.equal(context.visibleRuns(runs).map(r=>r.id).join(','),'formal');
  assert.equal(context.visibleRuns(runs,'all').length,4);
});
