"""Read-only experiment identity, effective metadata and bounded trace adapters."""
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("wb_monitor", Path(__file__).resolve().parents[1] / "scripts/monitor_training.py")
monitor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(monitor)
import monitor_workbench as wb


def write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def run_store(tmp_path):
    run = tmp_path / "candidate"
    run.mkdir()
    (run / "metrics.csv").write_text("iteration,p0_loss,p1_loss\n1,4,3\n2,3,2\n")
    return run, monitor.Store([tmp_path])


def test_effective_config_status_and_resource_freshness_have_separate_sources(tmp_path):
    run, store = run_store(tmp_path)
    write(run, "config.json", {"buffer_capacity": 4, "train_steps": 100})
    write(run, "launch_manifest.json", {"effective_config": {"buffer_capacity": 4, "effective_buffer_capacity": 8,
        "separate_player_models": False, "train_steps": 2400, "regret_trainable_parameters": 1491013}})
    write(run, "status.json", {"phase": "training", "last_completed_fit": {"iteration": "1"},
        "resources": {"time_utc": "2020-01-01T00:00:00Z", "ram_current_bytes": 123}})
    (run / "resources.jsonl").write_text('{"time_utc":"2020-01-01T00:00:00Z","ram_current_bytes":123}\n{"unfinished":')
    data = store.load("0/candidate")["workbench"]
    assert data["effective"]["effective_capacity"] == 8
    assert data["effective"]["train_steps"] == 2400
    assert data["effective"]["capacity_basis"] == "有效配置记录"
    assert data["runtime"]["training_step"] == 2
    assert data["runtime"]["source_iteration"] == 1
    assert data["runtime"]["process_liveness"] == "unknown"
    assert data["runtime"]["resource_time"] == 1577836800
    assert len(data["runtime"]["resources_history"]) == 1
    write(run, "status.json", {"phase": "evaluation_failed", "training_complete": True})
    assert store.load("0/candidate")["workbench"]["runtime"]["reported_phase"] == "evaluation_failed"


def trace_report(selection):
    return {"format": "cfr_equity_loss_trace_v1", "selection": selection, "selected_iterations": [1,2],
        "seed": 17, "sampling_power": 2 if selection == "burnin_quadratic" else 1,
        "summary": {"mbb_per_hand": 25, "ci95": [-1, 51], "pairs": 2}}


def test_same_iteration_variants_coexist_and_published_copy_deduplicates(tmp_path):
    run, store = run_store(tmp_path)
    report = {"format": "sd_cfr_readiness_v1", "hero_iteration": 2, "settings": {"seed": 17, "pairs": 2},
              "results": {"duplicate:2_vs_random": {"mbb_per_hand": 12, "pairs": 2}}}
    write(run, "report.json", report)
    write(run, "evaluations/readiness_periodic/full_budget_t2/report.json", report)
    write(run, "evaluations/full/report.json", trace_report("full"))
    write(run, "evaluations/burnin_quadratic/report.json", trace_report("burnin_quadratic"))
    data = store.load("0/candidate")
    evaluations = data["workbench"]["evaluations"]
    assert len(evaluations) == 3
    assert len({item["id"] for item in evaluations}) == 3
    assert {item["selection"] for item in evaluations} == {"full", "burnin_quadratic"}
    assert data["eval_latest"]["eval/winrate_vs_random_mbb"] == 12
    report["settings"]["seed"] = 18
    write(run, "evaluations/readiness_periodic/full_budget_t2/report.json", report)
    assert len(store.load("0/candidate")["workbench"]["evaluations"]) == 4


def test_trace_filtering_buckets_paging_and_hand_reading(tmp_path, monkeypatch):
    run, store = run_store(tmp_path)
    write(run, "evaluations/full/report.json", trace_report("full"))
    path = run / "evaluations/full/hands.jsonl"
    hands = [{"hero_seat": i%2, "payoff_chips": value, "category": "sb_open_fold" if i == 0 else "other_showdown",
              "snapshot": 2, "pair": i//2, "events": [{"seat": i%2,"street": 0,"action": 0,"pot":150,"to_call":50,"chips_added":0,"board":[]}]} for i,value in enumerate([-50,-100,200,100])]
    path.write_text("".join(json.dumps(h)+"\n" for h in hands) + "damaged\n")
    trace_id = store.load("0/candidate")["workbench"]["trace_sources"][0]["id"]
    page = store.hands("0/candidate", trace_id, limit=1)
    assert page["indexed_hands"] == 4 and page["skipped"] == 1
    assert page["items"][0]["record"] == 1
    assert page["buckets"]["sb_open_fold"]["contribution_mbb_per_all_hands"] == -125
    assert store.hands("0/candidate", trace_id, seat="1")["filtered_hands"] == 2
    assert store.hands("0/candidate", trace_id, category_filter="sb_open_fold")["filtered_hands"] == 1
    hand = store.hand("0/candidate", trace_id, 1)
    assert hand["payoff_chips"] == -100 and len(hand["events"]) == 1
    assert "hole_cards" not in hand
    with pytest.raises(KeyError):
        store.hand("0/candidate", trace_id, 999)
    with pytest.raises(KeyError):
        store.hands("0/candidate", "../../secret")
    monkeypatch.setattr(wb, "MAX_TRACE_ROWS", 2)
    store.workbench.trace_cache.clear()
    assert store.hands("0/candidate", trace_id)["truncated"] is True


def test_trace_cache_is_invalidated_on_replacement_and_symlinks_are_rejected(tmp_path):
    run, store = run_store(tmp_path)
    write(run, "evaluations/full/report.json", trace_report("full"))
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "hands.jsonl").write_text('{"payoff_chips":999,"events":[]}\n')
    (run / "evaluations/full/hands.jsonl").symlink_to(outside / "hands.jsonl")
    assert store.load("0/candidate")["workbench"]["trace_sources"] == []
    (run / "evaluations/full/hands.jsonl").unlink()
    path = run / "evaluations/full/hands.jsonl"
    path.write_text('{"payoff_chips":1,"events":[]}\n')
    trace_id = store.load("0/candidate")["workbench"]["trace_sources"][0]["id"]
    assert store.hand("0/candidate", trace_id, 0)["payoff_chips"] == 1
    path.write_text('{"payoff_chips":1000,"events":[]}\n')
    assert store.hand("0/candidate", trace_id, 0)["payoff_chips"] == 1000


def test_checkpoint_sidecar_does_not_claim_restore_or_read_weight_bytes(tmp_path):
    run, store = run_store(tmp_path)
    checkpoint = run / "train.pt"
    checkpoint.write_bytes(b"invalid pickle: must never deserialize")
    stat = checkpoint.stat()
    generation = "iteration_0000002_" + "a"*32
    (run / "checkpoint_data" / generation).mkdir(parents=True)
    write(run, "train.pt.refs.json", {"fingerprint": [stat.st_size,stat.st_mtime_ns,stat.st_ino],"generations":[generation]})
    write(run, "strategies/index.json", {"iterations":[1,2,"bad",False]})
    (run / "strategies/iteration_0000002.pt").write_bytes(b"also not a pickle")
    result = store.load("0/candidate")["workbench"]["assets"]
    assert result["references_match_file"] is True
    assert result["generations"][0]["directory_present"] is True
    assert result["history_complete"] is False and result["missing_files"] == [1]
    assert "未加载或恢复" in result["verification"]


def test_catalog_includes_initializing_runs_without_metrics_and_limits_path_access(tmp_path):
    run = tmp_path / "new_run"
    run.mkdir()
    write(run,"status.json",{"phase":"initializing"})
    store = monitor.Store([tmp_path])
    item = store.catalog()["experiments"][0]
    assert item["last_step"] is None and item["reported_phase"] == "initializing"
    with pytest.raises(KeyError):
        store.hands("0/../outside", "anything")


def test_compressed_archive_reports_manifest_objects_without_unpacking(tmp_path):
    run, store = run_store(tmp_path)
    path = run / "checkpoint_archives/iteration_0000002.tar.gz"
    path.parent.mkdir()
    path.write_bytes(b"not a gzip file: metadata checks must not unpack")
    obj = "a"*64 + ".pt"
    (path.parent / "objects").mkdir()
    (path.parent / "objects" / obj).write_bytes(b"data")
    write(run, "checkpoint_archives/iteration_0000002.tar.gz.json", {
        "format": "sd_cfr_restart_archive_v1", "strategy_objects": {
            "a": {"object": obj, "bytes": 4}, "b": {"object": "../escape.pt", "bytes": 0}},
        "restore_requires": "archive + manifest + objects/"})
    result = store.load("0/candidate")["workbench"]["assets"]["archives"][0]
    assert result["manifest_present"] is True
    assert result["objects_checked"] == 2 and result["objects_present"] == 1
    assert "未哈希或恢复" in result["verification"]


def test_catalog_uses_newer_evaluation_log_instead_of_old_published_report(tmp_path):
    run, store = run_store(tmp_path)
    write(run,"report.json",{"format":"sd_cfr_readiness_v1","hero_iteration":50,
        "results":{"duplicate:50_vs_random":{"mbb_per_hand":5}}})
    (run/"eval.jsonl").write_text('{"iteration":100,"eval/winrate_vs_random_mbb":10}\n{"iteration":200,"hud/preflop_vpip_sb":0.5}\n')
    assert store.catalog()["experiments"][0]["eval_step"] == 100


def test_stage2_jsonl_catalog_and_training_scope(tmp_path):
    run = tmp_path / 'stage2_candidate'
    run.mkdir()
    write(run, 'config.json', {'stage': 'stage2', 'updates': 10, 'device': 'cuda',
        'reference': 'frozen.pt', 'policy': {'alpha': 1}, 'ppo': {'epochs': 4}})
    rows = [{'update': u, 'stage2': 1, 'policy_loss': .1, 'base_kl': .03,
             'beta_kl': .5, 'kl_penalty': .015, 'seconds': 2,
             'stage2/matches': u, 'stage2/hands': 50*u,
             'stage2/hero_net_bb': v, 'stage2/opponent': name}
            for u,v,name in [(1,10,'equity'),(2,-4,'nit'),(3,2,'equity')]]
    (run / 'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows)+'{"update":4')
    store = monitor.Store([tmp_path])
    data = store.load('0/stage2_candidate')
    assert data['stage'] == 'stage2' and data['kind'] == 'Stage 2 PPO'
    assert data['latest']['stage2/base_kl'] == .03
    assert data['eval_latest'] == {}
    assert data['workbench']['runtime']['training_step'] == 3
    s2 = data['workbench']['stage2']
    assert s2['latest_opponent'] == 'equity'
    assert s2['window_wins'] == 2 and s2['window_losses'] == 1
    assert s2['window_net_bb_mean'] == pytest.approx(8/3)
    assert s2['opponents'][0]['mean_net_bb'] == 6
    assert s2['opponents'][1]['wins'] == 0
    item = store.catalog()['experiments'][0]
    assert item['last_step'] == 3 and item['category'] == 'training'
    assert item['effective']['updates'] == 10


def test_stage2_quota_block_uses_actual_counts_and_rolling_opponents(tmp_path):
    run = tmp_path / 'quota'
    run.mkdir()
    write(run,'config.json',{'stage':'stage2','matches_per_update':20})
    rows = [{'update':u,'stage2':1,'policy_loss':.1,'stage2/hero_net_bb':value,
        'stage2/matches':u*20,'stage2/sampled/equity':7,'stage2/sampled/reference':2,
        'stage2/sampled/history':3,'stage2/sampled/self':2,'stage2/sampled/nit':6,
        'stage2/window/equity/matches':7*u,'stage2/window/equity/mbb_per_hand':0.,
        'stage2/window/equity/match_win':0.} for u,value in [(1,10),(2,-2)]]
    (run/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    data=monitor.Store([tmp_path]).load('0/quota')
    s2=data['workbench']['stage2']
    assert s2['block_updates']
    assert s2['window_updates']==2 and s2['window_matches']==40
    assert s2['window_net_bb_mean']==4
    assert s2['opponents'][0]=={'name':'equity','matches':14,'mbb_per_hand':0,'match_win':0,'sampled':7}
    assert not data['eval_latest']


def test_stage2_block_win_counts_use_matches_not_update_means(tmp_path):
    run = tmp_path / 'block_wins'
    run.mkdir()
    write(run, 'config.json', {'stage':'stage2', 'matches_per_update':20})
    # Both block means are negative, yet 18 of the actual matches won.
    rows = [{'update':u, 'stage2':1, 'stage2/hero_net_bb':-140.7,
             'stage2/sampled/equity':8, 'stage2/sampled/history':12,
             'stage2/match_win':.45, 'stage2/match_draws':1} for u in (1,2)]
    (run/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    store = monitor.Store([tmp_path])
    summary = store.load('0/block_wins')['workbench']['stage2']
    assert summary['window_matches'] == 40
    assert summary['window_wins'] == 18
    assert summary['window_draws'] == 2 and summary['window_losses'] == 20
    assert summary['latest_opponent'] == 'mixed'
    assert summary['latest_sampled'] == {'equity':8, 'history':12}
    assert all(o['name'] != 'unknown' for o in summary['opponents'])
    # An incomplete ledger must never replace exact block counters.
    write(run, 'matches.jsonl', {'format':'stage2_training_match_v1', 'update':2,
          'match':21, 'opponent':'equity', 'hero_net_bb':100})
    assert store.load('0/block_wins')['workbench']['stage2']['window_wins'] == 18


def test_stage2_report_only_preserves_independent_results_and_trace_metadata(tmp_path):
    run,store=run_store(tmp_path)
    write(run,'config.json',{'stage':'stage2','matches_per_update':20})
    report={'format':'stage2_duplicate_matches_v1','update':100,'seed':9,'retention_pass':False,
      'equity_positive':False,'opponents':{'equity':{'candidate':{'mbb_per_hand':0,'mbb_ci95':[-10,10],
        'match_win':.4,'match_pairs':50},'gain_mbb_per_hand':-7}}}
    write(run,'stage2_evaluations/update_0000100.json',report)
    write(run,'evaluations/stage2_train_u0000001/report.json',{'format':'stage2_hand_trace_v1','scope':'training'})
    data=store.load('0/candidate')
    assert data['eval_latest']['eval/winrate_vs_equity_mbb']==0
    assert data['stage2_eval_latest']['stage2/eval/retention_pass']==0
    assert data['workbench']['evaluations'][0]['selection']=='stage2_candidate'
    assert data['workbench']['evaluations'][0]['results'][0]['ci95']==[-10,10]
    assert data['workbench']['evaluations'][0]['step']==100
    # A trace report is metadata, never an evaluation result.
    assert len(data['workbench']['evaluations'])==1
