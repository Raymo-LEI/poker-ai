"""CPU checks for the teacher on Kuhn poker, plus snapshot/restore.

This does not train Hold'em and does not claim a strength gain. Deep CFR
error is recorded only as a Kuhn function-approximation gap.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from alphanl.botzone.action_spec import A5, A6, SCHEMA_A6, check_head_width  # noqa: E402
from alphanl.cfr.config import CFRConfig  # noqa: E402
from alphanl.cfr.kuhn import (  # noqa: E402
    KuhnState,
    TabularRegretPolicy,
    _AverageStrategy,
    exploitability,
    game_value,
    nash_strategy,
    train_kuhn,
    uniform_strategy,
)
from alphanl.cfr.trainer import SDCFRTrainer  # noqa: E402
from alphanl.transformer.config import ModelConfig  # noqa: E402
from alphanl.transformer.schema import NUM_ACTIONS  # noqa: E402
from alphanl.transformer.tokenizer import BettingHistoryTokenizer  # noqa: E402


def _own_reach_check() -> dict:
    policy = TabularRegretPolicy()
    policy.regrets[policy.index((0, ""))] = torch.tensor([0, 1, 3, 0, 0], dtype=torch.float64)
    policies = {0: policy, 1: TabularRegretPolicy()}
    average = _AverageStrategy()
    average.add(policies, iteration=4)
    reach = float(policies[0].at((0, ""))[1])
    stored = float(average.sums[(0, "cb")].sum())
    expected = 4.0 * reach
    return {
        "check_probability": reach,
        "cb_mass": stored,
        "expected_cb_mass": expected,
        "matches": abs(stored - expected) < 1e-9,
        "weighting": "iteration times own reach, not opponent or chance reach",
    }


def _payoff_check() -> dict:
    folded = KuhnState((2, 0)).child(2).child(0).returns()
    called = KuhnState((2, 0)).child(2).child(1).returns()
    return {
        "units": "ante 1, bet 1, showdown stake 1 or 2",
        "bet_fold_returns": list(folded),
        "bet_call_returns": list(called),
        "zero_sum": folded[0] + folded[1] == 0 and called[0] + called[1] == 0,
        "equilibrium_value": -1 / 18,
        "nash_value": game_value(nash_strategy()),
        "nash_exploitability": exploitability(nash_strategy()),
        "uniform_exploitability": exploitability(uniform_strategy()),
    }


class _TinyGame:
    def __init__(self, actions=()):
        self.actions = actions

    def is_terminal(self):
        return len(self.actions) == 2

    def current_player(self):
        return len(self.actions)

    def legal_actions(self):
        return (1, 2)

    def observation(self):
        batch = BettingHistoryTokenizer(8).blank()
        batch["legal_mask"][0, 2] = 1
        batch["numeric"][0, 0, 0] = len(self.actions)
        return batch

    def child(self, action):
        return _TinyGame(self.actions + (action,))

    def returns(self):
        value = 1.0 if self.actions[0] == self.actions[1] else -1.0
        return (value, -value)


def _snapshot_restore(out_dir: Path) -> dict:
    cfg = CFRConfig(
        iterations=2,
        traversals_per_player=4,
        traversal_batch_size=4,
        buffer_capacity=100,
        train_steps=2,
        minibatch_size=8,
        separate_player_models=True,
        regret_target_scale=1.0,
        loss="mse",
        amp_dtype="none",
        fused_optimizer=False,
        allow_tf32=False,
        device="cpu",
        seed=7,
        model=ModelConfig(d_model=16, nhead=2, num_layers=1, dim_feedforward=32, max_seq_len=8, dropout=0),
        out_dir=str(out_dir / "original"),
    )
    trainer = SDCFRTrainer(cfg, root_factory=lambda seed: _TinyGame())
    first = trainer.run_iteration()
    checkpoint = out_dir / "train.pt"
    trainer.save_checkpoint(checkpoint)
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    snapshot_path = out_dir / "original" / "strategies" / "iteration_0000001.pt"
    snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    continued = trainer.run_iteration()
    restored = SDCFRTrainer(
        CFRConfig(**{**cfg.to_dict(), "out_dir": str(out_dir / "restored")}),
        root_factory=lambda seed: _TinyGame(),
    )
    restored.load_checkpoint(checkpoint)
    restored_iteration = int(restored.iteration)
    resumed = restored.run_iteration()
    finite = all(np.isfinite(float(first[key])) for key in ("p0_loss", "p1_loss"))
    return {
        "game": "two_node_signed_payoff_not_hunl",
        "gpu_hours_used": 0,
        "cuda_used": False,
        "completed_iteration_in_checkpoint": int(blob["completed_iteration"]),
        "snapshot_iteration": int(snapshot["iteration"]),
        "snapshot_weight": int(snapshot["weight"]),
        "iteration_equals_weight": int(snapshot["iteration"]) == int(snapshot["weight"]) == 1,
        "restored_iteration_before_resume": restored_iteration,
        "resume_metrics_match_continuation": resumed == continued,
        "losses_finite": finite,
        "p0_buffer_size": int(first["p0_buffer_size"]),
        "p1_buffer_size": int(first["p1_buffer_size"]),
        "both_seats_sampled": int(first["p0_buffer_size"]) > 0 and int(first["p1_buffer_size"]) > 0,
        "regret_target_scale": 1.0,
        "poststreet_coverage": "not_applicable_this_game_has_no_board",
    }


def _manifest(schema, out_name: str) -> dict:
    document = schema.document()
    quarter = "quarter_pot" in schema.names
    return {
        "format": "route1_r06_teacher_pilot_manifest_v1",
        "track": schema.schema_id,
        "action_schema_id": schema.schema_id,
        "action_schema_sha256": document["sha256"],
        "num_actions": schema.num_actions,
        "action_names": list(schema.names),
        "quarter_pot_required": quarter,
        "isolated_output_dir": f"runs/r06_{out_name}",
        "seed": 101 if schema.schema_id.endswith("a5_v2") or schema is A5 else 202,
        "max_gpu_hours": 0.25,
        "total_gpu_hours_cap": 1000,
        "gpu_hours_used_before_launch": 0,
        "max_concurrent_training_jobs": 1,
        "iterations": 1,
        "device": "cuda",
        "status": "PENDING_AUTODL_PARENT",
        "blocked_reason": (
            "CPU schema head matches this manifest. GPU pilot remains for the parent, "
            "one job at a time, at most 0.25h. Quarter-pot stays on A6."
        ),
        "command": (
            "python3 eval/r06/launch_pilot.py --manifest "
            f"eval/r06/manifests/{out_name}.json --execute"
        ),
        "strength_claim": False,
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(7)
    tabular = train_kuhn(iterations=1000, traversals_per_player=6, seed=1)
    reach = _own_reach_check()
    payoff = _payoff_check()
    if not reach["matches"] or not payoff["zero_sum"]:
        raise SystemExit("Kuhn reach or payoff check failed")
    if tabular.final_exploitability >= tabular.initial_exploitability / 8:
        raise SystemExit("tabular Kuhn exploitability did not fall")
    restore = _snapshot_restore(root / "snapshot_work")
    if not restore["resume_metrics_match_continuation"] or not restore["losses_finite"]:
        raise SystemExit("snapshot restore failed")
    if not restore["both_seats_sampled"] or not restore["iteration_equals_weight"]:
        raise SystemExit("seat mix or iteration semantics failed")
    head_gap = None
    try:
        check_head_width(A6, NUM_ACTIONS)
    except Exception as exc:
        head_gap = str(exc)
    report = {
        "task_id": "R06",
        "gpu_hours_used": 0,
        "cuda_used": False,
        "strength_claim": False,
        "hunl_convergence_claimed": False,
        "kuhn_tabular": {
            "iterations": 1000,
            "seed": 1,
            "traversals_per_player": 6,
            "initial_exploitability": tabular.initial_exploitability,
            "final_exploitability": tabular.final_exploitability,
            "player_zero_value": tabular.value,
            "equilibrium_value": -1 / 18,
            "regret_samples": tabular.regret_samples,
            "payoff": payoff,
            "own_reach": reach,
        },
        "deep_cfr_approximation": {
            "game": "kuhn_versus_two_node_regret_fit",
            "not_a_hunl_result": True,
            "hunl_convergence_claimed": False,
            "tabular_kuhn_exploitability": tabular.final_exploitability,
            "neural_fit_game": "two_node_signed_payoff",
            "neural_iterations": 2,
            "neural_losses_finite": restore["losses_finite"],
            "note": (
                "The neural fit is a two-iteration CPU regret regression on a toy game. "
                "Its loss is not a Hold'em exploitability and is not evidence the teacher converged."
            ),
        },
        "snapshot_restore": restore,
        "a6_head_width_check": head_gap,
        "gpu_pilot_status": "PENDING_AUTODL_PARENT",
    }
    (root / "teacher_pilot_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    manifest_dir = root / "manifests"
    manifest_dir.mkdir(exist_ok=True)
    a5 = _manifest(A5, "a5")
    a5["seed"] = 101
    a6 = _manifest(A6, "a6")
    a6["seed"] = 202
    (manifest_dir / "a5.json").write_text(json.dumps(a5, indent=2) + "\n", encoding="utf-8")
    (manifest_dir / "a6.json").write_text(json.dumps(a6, indent=2) + "\n", encoding="utf-8")
    selection = {
        "selected_candidate": None,
        "strength_claim": False,
        "reason": "A short pilot is not a strength result. Quarter-pot stays in the A6 manifest.",
        "follow_on": "Bind the regret head width to botzone_a6_v2 before any GPU pilot. Do not run the five-wide head and call it A6.",
        "a5_schema_sha256": a5["action_schema_sha256"],
        "a6_schema_sha256": a6["action_schema_sha256"],
        "quarter_pot_retained": True,
    }
    (root / "candidate_spec_selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "kuhn_exploitability": tabular.final_exploitability,
        "kuhn_value": tabular.value,
        "restore_match": restore["resume_metrics_match_continuation"],
        "gpu_pilot_status": report["gpu_pilot_status"],
    }))


if __name__ == "__main__":
    main()
