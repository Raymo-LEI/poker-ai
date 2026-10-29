const {test} = require('node:test');
const assert = require('node:assert/strict');
const {coordinates, matchedEndpoints, freshness, latestEvaluation} = require('../tools/training_monitor/workbench.js');

test('time/sample axes stop at missing data rather than inventing durations or draws', () => {
  const rows = [{step:1,iteration_s:10,p0_regret_samples:3,p1_regret_samples:7},
    {step:2,iteration_s:20,p0_regret_samples:8,p1_regret_samples:2},
    {step:3}, {step:4,iteration_s:5,p0_regret_samples:1,p1_regret_samples:1}];
  assert.deepEqual(coordinates(rows,'time').map(r=>r.step),[10,30]);
  assert.deepEqual(coordinates(rows,'samples').map(r=>r.step),[10,20]);
  assert.deepEqual(coordinates(rows,'iteration').map(r=>r.step),[1,2,3,4]);
});

test('shared recorded-time budget uses an actual complete endpoint without interpolation', () => {
  const a = {rows:[{step:1,iteration_s:10},{step:2,iteration_s:10},{step:3,iteration_s:10}]};
  const b = {rows:[{step:1,iteration_s:12},{step:2,iteration_s:12},{step:3,iteration_s:12}]};
  const matched = matchedEndpoints([a,b]);
  assert.equal(matched.budget,30);
  assert.equal(matched.points[0].originalStep,3);
  assert.equal(matched.points[1].originalStep,2);
  assert.equal(matchedEndpoints([a,{rows:[]}]),null);
});

test('old phase records are stale and never imply a live trainer', () => {
  assert.equal(freshness(undefined,1000),'unknown');
  assert.equal(freshness(800,1000),'stale');
  assert.equal(freshness(995,1000),'recent');
});

test('a newer logged evaluation takes precedence over an older full report, without choosing an intervention', () => {
  const full = {id:'full-50',step:50,selection:'full',results:[{mbb_per_hand:1}]};
  const logged = {id:'logged-100',step:100,selection:'logged',results:[{mbb_per_hand:2}]};
  const intervention = {id:'guard-200',step:200,selection:'full',intervention:true,results:[{mbb_per_hand:3}]};
  assert.equal(latestEvaluation([full,logged,intervention]),logged);
  assert.equal(latestEvaluation([{step:300,results:[]}]),undefined);
});
