"""Monitor regressions: real log formats, interrupted writes and read-only paths."""

import importlib.util
import json
import gzip
from http.client import HTTPConnection
from pathlib import Path
import threading

import pytest


SPEC = importlib.util.spec_from_file_location(
    "training_monitor", Path(__file__).resolve().parents[1] / "scripts/monitor_training.py")
monitor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(monitor)


def make_run(tmp_path, metrics="iteration,p0_loss,p1_loss\n1,12,34\n"):
    run = tmp_path / "experiment"
    run.mkdir()
    (run / "metrics.csv").write_text(metrics)
    store = monitor.Store([tmp_path])
    return run, store


def test_fixed_fit_probe_is_not_overwritten_by_checkpoint_probe(tmp_path):
    run, store = make_run(tmp_path, "iteration,p0_probe_r2,p0_probe_loss_ratio,p0_probe_samples\n100,0.72,0.91,1024\n")
    (run / "diagnostics.jsonl").write_text(
        '{"iteration":100,"p0_probe_r2":0.69,"p0_probe_samples":2048,"p0_probe_end_zero_ratio":0.3}\n')
    (run / "ops").mkdir()
    (run / "ops/environment.json").write_text('{"device":"cuda","gpu":"RTX 4090"}')
    data = store.load("0/experiment")
    probe = data["diagnostic_latest"]
    assert probe["train/probe_r2_p0"] == .72
    assert probe["train/probe_loss_ratio_p0"] == .91
    assert probe["train/probe_samples_p0"] == 1024
    assert probe["p0_checkpoint_probe_r2"] == .69
    assert probe["p0_checkpoint_probe_samples"] == 2048
    assert probe["train/probe_end_zero_ratio_p0"] == .3
    assert data["environment"]["gpu"] == "RTX 4090"


def test_csv_join_and_weighted_rates(tmp_path):
    run, store = make_run(tmp_path, "iteration,p0_loss,p0_regret_samples,p1_regret_samples,p0_inference_rows,p1_inference_rows,p0_inference_calls,p1_inference_calls\n1,12,20,80,40,160,10,10\n")
    (run / "performance.csv").write_text("iteration,p0_collect_s,p1_collect_s,iteration_s\n1,2,8,11\n")
    data = store.load("0/experiment")
    assert len(data["rows"]) == 1
    assert data["latest"]["train/loss_p0"] == 12
    assert data["latest"]["system/regret_samples_per_s"] == 10
    assert data["latest"]["system/average_inference_batch"] == 10
    assert data["latest"]["iteration_s"] == 11


def test_partial_csv_and_jsonl_are_ignored_until_completed(tmp_path):
    run, store = make_run(tmp_path, "iteration,p0_loss\n1,12\n2,3")
    (run / "eval.jsonl").write_text('{"iteration":1,"eval/winrate_vs_random_mbb":25}\n{"iteration":2')
    assert store.load("0/experiment")["latest"]["step"] == 1
    with (run / "metrics.csv").open("a") as handle:
        handle.write("0\n")
    data = store.load("0/experiment")
    assert data["latest"]["train/loss_p0"] == 30
    assert data["eval_latest"]["step"] == 1
    assert data["eval_latest"]["eval/winrate_vs_random_mbb"] == 25


def test_bad_records_and_nonfinite_are_not_fake_zero(tmp_path):
    run, store = make_run(tmp_path, "iteration,p0_loss\n1,nan\n2,inf\n3,12,unexpected\n")
    (run / "metrics.jsonl").write_text('invalid\n[1,2]\n{"iteration":2,"train/explained_variance_p0":NaN}\n')
    data = store.load("0/experiment")
    assert data["latest"]["p0_loss"] is None
    assert data["latest"]["train/explained_variance_p0"] is None
    assert any("损坏" in warning for warning in data["warnings"])
    assert any("p0_loss" in warning for warning in data["warnings"])
    json.dumps(data, allow_nan=False)


def test_ppo_zero_update_and_units(tmp_path):
    _, store = make_run(tmp_path, "update,policy_loss,value_loss,eval_random_bb,legal_entropy,seconds\n0,-0.2,2.5,0.125,0.7,3\n1,-0.1,1.5,,0.6,4\n")
    data = store.load("0/experiment")
    assert data["kind"] == "PPO"
    assert data["rows"][0]["step"] == 0
    assert data["latest"]["train/policy_loss"] == -.1
    assert data["eval_latest"]["eval/winrate_vs_random_mbb"] == 125
    assert data["eval_latest"]["step"] == 0
    assert "eval/winrate_vs_random_mbb_ci_low" not in data["eval_latest"]


def test_nested_jsonl_hud_warning_and_interval(tmp_path):
    run, store = make_run(tmp_path)
    (run / "eval.jsonl").write_text(json.dumps({"step": 1, "metrics": {
        "eval": {"winrate_vs_random_mbb": 50, "winrate_vs_random_mbb_ci_low": -15,
                 "winrate_vs_random_mbb_ci_high": 115},
        "hud": {"action_freq_preflop": {"all_in": .2}}}}) + "\n")
    data = store.load("0/experiment")
    assert data["hud_latest"]["hud/action_freq_preflop/all_in"] == .2
    assert data["eval_latest"]["eval/winrate_vs_random_mbb_ci_low"] == -15
    assert any("超过 10%" in warning for warning in data["warnings"])


def test_benchmark_and_checkpoint_metadata_without_loading_weights(tmp_path):
    run, store = make_run(tmp_path)
    (run / "benchmark.json").write_text(json.dumps({"environment": {"device": "cuda"},
        "config": {"iterations": 2}, "iterations": [{"iteration": 2, "p0_collect_s": 5,
        "metrics": {"p0_loss": 9, "p1_loss": 7}}]}))
    (run / "train.pt").write_bytes(b"this is not a pickle")
    (run / "checkpoint_performance.jsonl").write_text('{"iteration":2,"checkpoint_s":0.4}\n')
    data = store.load("0/experiment")
    assert data["latest"]["train/loss_p0"] == 9
    assert data["latest"]["checkpoint_s"] == .4
    assert data["checkpoint"]["bytes"] == 20
    assert data["environment"]["device"] == "cuda"


def test_path_traversal_and_outside_symlinks(tmp_path):
    run, store = make_run(tmp_path)
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    (outside / "metrics.csv").write_text("iteration,p0_loss\n999,999\n")
    (tmp_path / "escape").symlink_to(outside)
    (run / "performance.csv").symlink_to(outside / "metrics.csv")
    assert set(store.discover()) == {"0/experiment"}
    assert store.load("0/experiment")["latest"]["step"] == 1
    with pytest.raises(KeyError):
        store.load("0/../" + outside.name)


def test_direct_run_root_and_replacement_on_resume(tmp_path):
    run, _ = make_run(tmp_path)
    store = monitor.Store([run])
    assert store.load("0/.")["latest"]["step"] == 1
    (run / "metrics.csv").write_text("iteration,p0_loss\n0,99\n")
    assert store.load("0/.")["latest"]["step"] == 0


def test_bounded_tail_preserves_csv_header(tmp_path, monkeypatch):
    run, store = make_run(tmp_path, "iteration,p0_loss\n" + "".join(f"{i},{i}\n" for i in range(100)))
    monkeypatch.setattr(monitor, "MAX_BYTES", 80)
    monkeypatch.setattr(monitor, "MAX_ROWS", 3)
    data = store.load("0/experiment")
    assert [r["step"] for r in data["rows"]] == [97, 98, 99]
    assert any("末尾" in warning for warning in data["warnings"])


def test_empty_run_and_read_only(tmp_path):
    run, store = make_run(tmp_path, "iteration,p0_loss\n")
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in run.iterdir()}
    data = store.load("0/experiment")
    assert data["rows"] == [] and data["latest"] == {}
    assert before == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in run.iterdir()}


def test_http_compression_revalidation_and_changed_logs(tmp_path):
    run, store = make_run(tmp_path, "iteration,p0_loss\n" + "".join(f"{i},{i}\n" for i in range(100)))
    server = monitor.ThreadingHTTPServer(("127.0.0.1", 0), monitor.make_handler(store))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = HTTPConnection(*server.server_address, timeout=5)
    try:
        url = "/api/run?id=0/experiment"
        connection.request("GET", url, headers={"Accept-Encoding": "gzip"})
        response = connection.getresponse()
        encoded = response.read()
        etag = response.getheader("ETag")
        assert response.status == 200 and response.getheader("Content-Encoding") == "gzip"
        decoded = gzip.decompress(encoded)
        assert len(encoded) < len(decoded) / 2
        assert json.loads(decoded)["latest"]["step"] == 99
        connection.request("GET", url, headers={"If-None-Match": etag, "Accept-Encoding": "gzip"})
        response = connection.getresponse()
        assert response.status == 304 and response.read() == b""
        with (run / "metrics.csv").open("a") as handle:
            handle.write("100,25\n")
        connection.request("GET", url, headers={"If-None-Match": etag, "Accept-Encoding": "gzip;q=0"})
        response = connection.getresponse()
        assert response.status == 200 and response.getheader("ETag") != etag
        assert response.getheader("Content-Encoding") is None
        assert json.loads(response.read())["latest"]["step"] == 100
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_readiness_report_adapts_only_matching_metrics(tmp_path):
    run, store = make_run(tmp_path)
    (run / "report.json").write_text(json.dumps({"format": "sd_cfr_readiness_v1", "hero_iteration": 100,
        "results": {"duplicate:100_vs_random": {"mbb_per_hand": 90, "ci95": [10, 170],
                "sb_mbb_per_hand": 130, "bb_mbb_per_hand": 50},
            "duplicate:50_vs_random": {"mbb_per_hand": 999},
            "matches:100_vs_random": {"win_fraction": .7},
            "health": {"hero": {"sb_vpip": .6, "mean_nonforced_entropy_nats": .8,
                "action_counts_by_street": {"0:0": 20, "0:1": 70, "0:4": 10}}}}}))
    data = store.load("0/experiment")
    assert data["eval_latest"]["eval/winrate_vs_random_mbb"] == 90
    assert data["hud_latest"]["hud/action_freq_preflop/all_in"] == .1
    assert data["eval_latest"]["eval/hunl_50hand_match_win_rate_vs_random"] == .7
    assert "eval/botzone_50hand_match_win_rate" not in data["eval_latest"]
    assert data["evaluation_only"] is False


def test_periodic_reports_preserve_history_and_invalidate_cache(tmp_path):
    run, store = make_run(tmp_path)

    def write_report(step, score):
        path = run / f"evaluations/readiness_periodic/full_budget_t{step}/report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"format": "sd_cfr_readiness_v1", "hero_iteration": step,
            "results": {f"duplicate:{step}_vs_random": {"mbb_per_hand": score, "ci95": [score-1, score+1], "pairs": step},
                "duplicate:25_vs_random": {"mbb_per_hand": 999},
                "health": {"hands": step * 10, "hero": {"sb_vpip": .5}}}}))
        return path

    for step in (100, 150, 200, 250):
        last = write_report(step, step * 2)
    (run / "report.json").write_bytes(last.read_bytes())
    data = store.load("0/experiment")
    evaluations = [r for r in data["rows"] if "eval/winrate_vs_random_mbb" in r]
    assert [r["step"] for r in evaluations] == [100, 150, 200, 250]
    assert [r["eval/winrate_vs_random_mbb"] for r in evaluations] == [200, 300, 400, 500]
    assert data["eval_latest"]["step"] == data["hud_latest"]["step"] == 250
    assert len([s for s in data["sources"] if s["name"].startswith("evaluations/")]) == 4
    write_report(100, 123)
    assert store.load("0/experiment")["rows"][1]["eval/winrate_vs_random_mbb"] == 123
    write_report(300, 600)
    assert store.load("0/experiment")["eval_latest"]["step"] == 300
    # Root is an older published copy: it must not suppress newer reports.
    last.unlink()
    (run / "report.json").unlink()
    assert 250 not in [r["step"] for r in store.load("0/experiment")["rows"]]


def test_periodic_discovery_excludes_interventions_and_outside_symlinks(tmp_path):
    run, store = make_run(tmp_path)
    outside = tmp_path.parent / (tmp_path.name + "-foreign-report")
    outside.mkdir()
    (outside / "report.json").write_text('{"format":"sd_cfr_readiness_v1","hero_iteration":999}')
    base = run / "evaluations/readiness_periodic"
    base.mkdir(parents=True)
    (base / "full_budget_t999").symlink_to(outside)
    (base / "guard_override_t100").mkdir()
    (base / "guard_override_t100/report.json").write_bytes((outside / "report.json").read_bytes())
    assert monitor.history_reports(run) == []
    assert store.load("0/experiment")["latest"]["step"] == 1


def test_evaluation_only_shows_actual_budget_and_previous_snapshot(tmp_path):
    run = tmp_path / "evaluation"
    run.mkdir()
    (run / "report.json").write_text(json.dumps({"format": "sd_cfr_readiness_v1", "hero_iteration": 2,
        "settings": {"pairs": 2, "matches": 1, "health_hands": 4},
        "results": {"duplicate:2_vs_1": {"mbb_per_hand": 123, "ci95": [-200, 446], "pairs": 2}}}))
    data = monitor.Store([tmp_path]).load("0/evaluation")
    assert data["evaluation_only"] is True
    assert data["eval_latest"]["eval/winrate_vs_prev_checkpoint_mbb"] == 123
    assert data["eval_latest"]["eval/winrate_vs_iter_1_mbb_pairs"] == 2
    assert any("2 对发牌" in warning for warning in data["warnings"])


def test_server_run_config_and_source_label(tmp_path):
    run, _ = make_run(tmp_path)
    (run / "ops").mkdir()
    (run / "ops/run_config.json").write_text('{"device":"cuda","iterations":50}')
    data = monitor.Store([tmp_path], "4090 服务器").load("0/experiment")
    assert data["source"] == "4090 服务器"
    assert data["config"]["device"] == "cuda"


def test_separate_probe_log_joins_by_iteration_and_keeps_real_zero(tmp_path):
    run, store = make_run(tmp_path)
    (run / "diagnostics.jsonl").write_text('{"iteration":1,"p0_probe_r2":0,"p1_probe_r2":-0.2,"p0_probe_end_zero_ratio":0.98}\n')
    data = store.load("0/experiment")
    assert data["diagnostic_latest"]["train/probe_r2_p0"] == 0
    assert data["diagnostic_latest"]["train/probe_r2_p1"] == -.2
    assert data["latest"]["train/probe_end_zero_ratio_p0"] == .98


def test_stage2_metrics_panel_schema(tmp_path):
    run = tmp_path / "stage2"
    run.mkdir()
    (run / "config.json").write_text('{"stage":"stage2","device":"cpu"}')
    (run / "metrics.jsonl").write_text(json.dumps({"update":1, "stage2":1, "base_kl":.01,
        "beta_kl":1., "kl_penalty":.01, "stage2/pfsp/equity":.35,
        "stage2/context/vpip":.45, "stage2/hero_net_bb":12.}) + "\n")
    data = monitor.Store([tmp_path]).load("0/stage2")
    assert data["kind"] == "Stage 2 PPO"
    assert data["latest"]["stage2/base_kl"] == .01
    assert data["latest"]["stage2/pfsp/equity"] == .35
    assert not data["eval_latest"]  # rollout profit is not external evaluation


def test_stage2_independent_eval_survives_new_training_rows(tmp_path):
    run = tmp_path / "stage2"
    run.mkdir()
    (run / "config.json").write_text('{"stage":"stage2"}')
    rows = [{"update":100, "stage2":1, "stage2/eval/update":100,
             "stage2/eval/equity/mbb_per_hand":0., "stage2/eval/equity/match_win":.6,
             "stage2/eval/reference/mbb_per_hand":-21., "stage2/eval/retention_pass":0,
             "stage2/eval/nit/gain_mbb_per_hand":1200.},
            {"update":101, "stage2":1, "stage2/hero_net_bb":10000.}]
    (run / "metrics.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
    data = monitor.Store([tmp_path]).load("0/stage2")
    assert data["latest"]["step"] == 101
    assert data["stage2_eval_latest"]["stage2/eval/update"] == 100
    assert data["stage2_eval_latest"]["stage2/eval/equity/mbb_per_hand"] == 0
    assert data["stage2_eval_latest"]["stage2/eval/retention_pass"] == 0
    assert not data["eval_latest"]
