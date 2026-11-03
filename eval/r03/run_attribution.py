"""Four-layer attribution on public Botzone infosets.

Teacher snapshots and the frozen SuperStudent checkpoint are not in this
tree. Those layers are ``MISSING`` and are not filled with an unweighted
average. PPO raw probabilities and deployment probabilities both come from
the frozen V6 pack: raw is the masked softmax, deploy is that vector after
the pack's guard. PyTorch and NumPy see the same FP32 weights.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "eval"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "v6_runtime"))

from alphanl.botzone.kernel import BranchableKernel, LegacyFiveActionBridge  # noqa: E402
from alphanl.cfr.teacher_window import load_teacher_window, missing_teacher_record  # noqa: E402
from botzone_referee_adapter import JUDGE  # noqa: E402

_NPZ = Path(__file__).resolve().parent / "weights" / "V6_R6_MODEL.npz"
_NPZ_SHA = "dfd57d307c6d4d9eb5d789ba145370792752ea85f5f43ba57b5dd787e3b4faaa"
_JUDGE_SHA = "b70702fdcd28333f91087f1a3dc3ede1ca59737404622c64dc6dbf7e9206e544"
_JUDGE_PATH = _REPO / "tests" / "fixtures" / "botzone_judge.py"
_WINDOW = Path(__file__).resolve().parent / "teacher_window_manifest.json"
_RANKS = "23456789TJQKA"
_SUITS = "hdsc"
_FOLD_BASELINE = 0.01
_PT_NP_LIMIT = 1e-4


def card_id(text: str) -> int:
    if len(text) != 2 or text[0] not in _RANKS or text[1] not in _SUITS:
        raise ValueError(f"card must look like Ks, got {text!r}")
    return _RANKS.index(text[0]) * 4 + _SUITS.index(text[1])


def permute_card(card: int, suit_map: dict) -> int:
    return (card // 4) * 4 + suit_map[card % 4]


def permute_cards(cards, suit_map: dict):
    return [permute_card(card, suit_map) for card in cards]


def deck_from_partial(holes, board) -> list:
    """Judge pop-order deck. Filler cards are the unused ids, lowest first."""

    known = [int(card) for card in (*holes[0], *holes[1], *board)]
    if len(known) != len(set(known)):
        raise ValueError("holes and board overlap")
    unused = [card for card in range(52) if card not in set(known)]
    full = list(board) + unused[: 5 - len(board)]
    rest = unused[5 - len(board) :]
    sequence = [holes[0][0], holes[0][1], holes[1][0], holes[1][1], *full]
    deck = rest + list(reversed(sequence))
    if len(deck) != 52 or len(set(deck)) != 52:
        raise RuntimeError("deck is not 52 distinct cards")
    return deck


def reach(holes, board, actions, *, dealer: int = 0) -> BranchableKernel:
    kernel = BranchableKernel.start_hand(deck_from_partial(holes, board), dealer=dealer)
    for response in actions:
        kernel.step(int(response))
    return kernel


def _fresh_cards(hole, board):
    return [JUDGE.Card.from_int(int(card)) for card in (*hole, *board)]


def absolute_nuts(hole, board) -> bool:
    """True when every other two-card holding loses to this hand on a full board."""

    if len(board) != 5 or len(hole) != 2:
        return False
    used = set(int(card) for card in (*hole, *board))
    hero = _fresh_cards(hole, board)
    for left in range(52):
        if left in used:
            continue
        for right in range(left + 1, 52):
            if right in used:
                continue
            opp = _fresh_cards((left, right), board)
            if JUDGE.compare_full_cards(list(hero), opp) <= 0:
                return False
    return True


def equity_clue(hole, board) -> dict:
    """Win/tie rate against the remaining hands. A clue, not a training label."""

    if len(board) < 3:
        return {"status": "not_computed", "reason": "board shorter than a flop"}
    used = set(int(card) for card in (*hole, *board))
    wins = ties = losses = 0
    hero = _fresh_cards(hole, board)
    for left in range(52):
        if left in used:
            continue
        for right in range(left + 1, 52):
            if right in used:
                continue
            compared = JUDGE.compare_full_cards(list(hero), _fresh_cards((left, right), board))
            if compared > 0:
                wins += 1
            elif compared < 0:
                losses += 1
            else:
                ties += 1
    total = wins + ties + losses
    return {
        "status": "clue_only",
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "win_rate": wins / total,
        "not_a_zero_fold_label": True,
    }


def _load_pack():
    blob = _NPZ.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != _NPZ_SHA:
        raise SystemExit(f"V6 npz sha {digest} != {_NPZ_SHA}")
    archive = np.load(_NPZ, allow_pickle=True)
    try:
        meta = json.loads(bytes(archive["meta_json"]).decode("utf-8"))
        weights = {
            key: np.array(archive[key], dtype=np.float32, copy=True)
            for key in archive.files
            if key != "meta_json"
        }
    finally:
        archive.close()
    runtime = importlib.import_module("stage2_runtime.runtime")
    guard_mod = importlib.import_module("stage2_runtime.deployment_guard")
    agent = runtime.Agent(
        weights=weights,
        config=meta["config"],
        settings=dict(meta["settings"]),
        numeric_version=meta["numeric_version"],
        deployment_guard=meta["deployment_guard"],
    )
    return meta, weights, runtime, guard_mod, agent


def _request(kernel: BranchableKernel) -> dict:
    audit = kernel.referee_audit_state()
    seat = int(kernel.actor)
    return {
        "num_players": 2,
        "dealer_id": int(kernel.dealer),
        "my_id": seat,
        "my_chips": int(kernel.stacks[seat]),
        "my_cards": [int(card) for card in audit["holes"][seat]],
        "public_cards": [int(card) for card in kernel.public_cards],
        "history": [dict(item) for item in audit["history"]],
        "hand": 0,
    }


def _amounts(snap, responses):
    return {
        action: snap.me_stack if response == -2 else min(snap.to_call, snap.me_stack) if action == 1 else max(response, 0)
        for action, response in responses.items()
    }


def _softmax(logits) -> np.ndarray:
    row = np.asarray(logits, dtype=np.float64).reshape(-1)
    row = row - np.max(row)
    probs = np.exp(row)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError("softmax failed")
    return probs / total


def _illegal_mass(probs, legal_ids) -> float:
    legal = set(int(action) for action in legal_ids)
    return float(sum(float(probs[action]) for action in range(5) if action not in legal))


def diagnose(kernel: BranchableKernel, pack) -> dict:
    meta, _weights, runtime, guard_mod, agent = pack
    request = _request(kernel)
    opponent = kernel.referee_audit_state()["holes"][1 - kernel.actor]
    if any(card in request["my_cards"] or card in request["public_cards"] for card in opponent):
        raise RuntimeError("request leaked an opponent hole card")
    snap, responses = runtime.request_snapshot(request)
    context, _data = runtime.context_from_payload({"requests": [request]})
    batch = agent.observation(snap, context)
    logits = agent.logits(batch)[0]
    raw = _softmax(logits)
    guarded, info = guard_mod.apply_guard(
        snap, raw.copy(), agent.deployment_guard, runtime.estimated_equity, _amounts(snap, responses)
    )
    guarded = np.asarray(guarded, dtype=np.float64)
    type_ids = np.asarray(batch["type_ids"][0])
    action_tokens = int(np.sum(type_ids == 8))
    bridge = LegacyFiveActionBridge(kernel)
    return {
        "request_public": request,
        "seat": int(kernel.actor),
        "street": int(kernel.street),
        "dealer": int(kernel.dealer),
        "pot": int(kernel.pot),
        "stacks": list(kernel.stacks),
        "to_call": None if kernel.min_positive_extra() is None and not kernel.is_legal(0) else int(snap.to_call),
        "numeric_features": np.asarray(batch["numeric"][0]).tolist(),
        "match_ctx": np.asarray(batch["match_ctx"][0]).tolist(),
        "feature_names_kernel": list(kernel.policy_view()["feature_names"]),
        "kernel_features": kernel.policy_view()["features"],
        "legal_ids": [int(action) for action in snap.legal_action_ids],
        "deploy_chip_responses": {str(action): int(response) for action, response in responses.items()},
        "legacy_chip_responses": [
            None if response is None else int(response) for response in bridge.responses()
        ],
        "history_events": len(snap.events),
        "history_action_tokens": action_tokens,
        "history_truncated": action_tokens < len(snap.events),
        "max_seq_len": int(meta["config"]["model"]["max_seq_len"]),
        "ppo_raw_probs": raw.tolist(),
        "deploy_probs": guarded.tolist(),
        "guard_changed_argmax": int(np.argmax(raw)) != int(np.argmax(guarded)),
        "guard_info_keys": sorted(str(key) for key in info),
        "illegal_mass_raw": _illegal_mass(raw, snap.legal_action_ids),
        "illegal_mass_deploy": _illegal_mass(guarded, snap.legal_action_ids),
        "batch": batch,
        "logits": np.asarray(logits, dtype=np.float32),
    }


def load_torch_policy(weights, meta):
    import torch
    from alphanl.cfr.config import CFRConfig
    from alphanl.cfr.distillation import PolicyNetwork
    from alphanl.stage2.policy import ResidualExploiterPolicy

    known = set(CFRConfig.__dataclass_fields__)
    config = CFRConfig.from_checkpoint({key: value for key, value in meta["config"].items() if key in known})
    reference = PolicyNetwork(config)
    settings = meta["settings"]
    model = ResidualExploiterPolicy(
        reference,
        alpha=float(settings["alpha"]),
        hidden=int(settings["hidden"]),
        beta_kl=float(settings["beta_kl"]),
        late_beta_kl=float(settings["late_beta_kl"]),
    )
    model.eval()
    state = model.state_dict()
    mapped = {}
    residual = ("context_encoder", "fusion", "delta_head", "value_head")
    for key, array in weights.items():
        prefix = key.split(".", 1)[0]
        torch_key = key if prefix in residual else "reference.backbone." + key
        if torch_key not in state:
            raise KeyError(f"npz key {key} does not match torch parameter {torch_key}")
        tensor = torch.from_numpy(np.array(array, dtype=np.float32))
        if tuple(tensor.shape) != tuple(state[torch_key].shape):
            raise ValueError(f"shape mismatch for {torch_key}: {tuple(tensor.shape)} vs {tuple(state[torch_key].shape)}")
        mapped[torch_key] = tensor
    missing = [key for key in state if key not in mapped]
    if missing:
        raise KeyError(f"torch parameters missing from the npz: {missing}")
    model.load_state_dict(mapped, strict=True)
    model.eval()
    return model


def torch_probs(model, batch) -> np.ndarray:
    import torch

    converted = {}
    for key, value in batch.items():
        array = np.asarray(value)
        if key == "pad_mask":
            converted[key] = torch.from_numpy(array.astype(np.bool_))
        elif key in ("token_ids", "type_ids"):
            converted[key] = torch.from_numpy(array.astype(np.int64))
        else:
            converted[key] = torch.from_numpy(np.array(array, dtype=np.float32))
    with torch.inference_mode():
        logits, _value = model(converted)
    return _softmax(logits[0].detach().cpu().numpy())


def _round_key(probs):
    return [int(round(float(prob) * 1e8)) for prob in probs]


def _layer_for(record) -> str:
    if record["encoding_illegal_responses"]:
        return "encoding"
    if record["proof_grade"] == "hard_dominated":
        raw_fold = record["ppo_raw_probs"][0]
        deploy_fold = record["deploy_probs"][0]
        if raw_fold > _FOLD_BASELINE:
            return "ppo"
        if deploy_fold > _FOLD_BASELINE and deploy_fold > raw_fold + 1e-8:
            return "guard"
        return "none"
    return "unattributed"


def _case(case_id, source, proof_grade, holes, board, actions, *, dealer=0, note=""):
    return {
        "case_id": case_id,
        "source": source,
        "proof_grade": proof_grade,
        "holes": holes,
        "board": board,
        "actions_before_decision": list(actions),
        "dealer": dealer,
        "note": note,
    }


def build_cases():
    """Named public situations plus suit permutations. No opponent-hole labels."""

    junk = ["2c", "3c"]
    specs = [
        _case(
            "river_ks_nut_flush",
            "constructed_absolute_nuts",
            "hard_dominated",
            [["5h", "6h"], ["Ks", "3s"]],
            ["As", "9s", "4s", "2h", "7d"],
            [0, 0, 0, 0, 0, 0],
            note="Ace is on the board. Hero Ks3s is the nut flush and beats every other holding. Checkdown to the river; SB is to act.",
        ),
        _case(
            "river_ks_nut_flush_facing_bet",
            "constructed_absolute_nuts",
            "hard_dominated",
            [["Ks", "3s"], ["5h", "6h"]],
            ["As", "9s", "4s", "2h", "7d"],
            [0, 0, 0, 0, 0, 0, 400],
            note="Same nuts on the BB, facing a 400 river bet. Fold is strictly worse than call.",
        ),
        _case(
            "turn_nut_flush",
            "constructed_turn_nuts",
            "research_pending",
            [["5h", "6h"], ["Ks", "3s"]],
            ["As", "9s", "4s", "2h"],
            [0, 0, 0, 0],
            note="Turn nut flush is not a certified zero-fold state. Equity is a clue only. SB acts.",
        ),
        _case(
            "flop_74_top_pair_gutshot",
            "constructed_top_pair_gutshot",
            "research_pending",
            [["2c", "3c"], ["7d", "4s"]],
            ["7h", "6c", "2d"],
            [0, 0],
            note="Top pair plus a gutshot. Not a dominated-fold proof. SB acts.",
        ),
        _case(
            "river_board_two_pair",
            "constructed_board_two_pair",
            "research_pending",
            [["2c", "4c"], ["7c", "3s"]],
            ["Ah", "Ad", "Kh", "Kd", "2h"],
            [0, 0, 0, 0, 0, 0],
            note="Board is aces and kings. Hero does not beat a full house. Not a zero-fold label. SB acts.",
        ),
        _case(
            "flop_nut_flush_draw",
            "constructed_flush_draw",
            "research_pending",
            [["2c", "3c"], ["Ah", "Kh"]],
            ["Qh", "9h", "2d"],
            [0, 0],
            note="Nut flush draw. Future cards are not a dominated-fold proof. SB acts.",
        ),
        _case(
            "flop_facing_overbet_reraise",
            "constructed_overbet_reraise",
            "research_pending",
            [["8c", "7c"], ["9d", "6d"]],
            ["Kh", "Td", "4c", "2s", "3h"],
            [0, 0, 100, 1000],
            note="SB bets 100 on the flop. BB reraises 1000 into a pot of 300. SB is to act.",
        ),
    ]
    suit_cycles = (
        {0: 1, 1: 2, 2: 3, 3: 0},
        {0: 2, 1: 3, 2: 0, 3: 1},
        {0: 3, 1: 0, 2: 1, 3: 2},
    )
    cases = []
    for spec in specs:
        cases.append(spec)
        holes = [[card_id(card) for card in pair] for pair in spec["holes"]]
        board = [card_id(card) for card in spec["board"]]
        for index, suit_map in enumerate(suit_cycles, start=1):
            perm = dict(spec)
            perm["case_id"] = f"{spec['case_id']}__suit{index}"
            perm["source"] = spec["source"] + "_suit_permutation"
            perm["holes_text"] = spec["holes"]
            perm["board_text"] = spec["board"]
            perm["holes"] = [permute_cards(pair, suit_map) for pair in holes]
            perm["board"] = permute_cards(board, suit_map)
            perm["note"] = spec["note"] + f" Suit cycle {index}."
            cases.append(perm)
    # One rank-shifted copy of the top-pair gutshot. 8-5 on 8-7-3 is the same shape.
    cases.append(
        _case(
            "flop_85_top_pair_gutshot",
            "constructed_top_pair_gutshot_rank_shift",
            "research_pending",
            [["2c", "3c"], ["8d", "5s"]],
            ["8h", "7c", "3d"],
            [0, 0],
            note="Rank shift of 74 top pair plus gutshot. Still research-pending.",
        )
    )
    return cases


def _materialize(spec):
    if spec["holes"] and isinstance(spec["holes"][0][0], str):
        holes = [[card_id(card) for card in pair] for pair in spec["holes"]]
        board = [card_id(card) for card in spec["board"]]
    else:
        holes = spec["holes"]
        board = spec["board"]
    kernel = reach(holes, board, spec["actions_before_decision"], dealer=spec["dealer"])
    return holes, board, kernel


def _public_record(spec, diagnosis, window) -> dict:
    holes, board, kernel = diagnosis["kernel_parts"]
    audit = kernel.referee_audit_state()
    seat = kernel.actor
    hero = list(audit["holes"][seat])
    shown = list(kernel.public_cards)
    nuts = absolute_nuts(hero, shown) if len(shown) == 5 else False
    grade = spec["proof_grade"]
    if grade == "hard_dominated" and not nuts:
        raise RuntimeError(f"{spec['case_id']} was marked hard-dominated but is not absolute nuts")
    if grade != "hard_dominated" and nuts and len(shown) == 5 and spec["case_id"].startswith("river_"):
        raise RuntimeError(f"{spec['case_id']} is absolute nuts but not graded hard_dominated")
    encoding_illegal = []
    for action_text, response in diagnosis["deploy_chip_responses"].items():
        if not kernel.is_legal(int(response)):
            encoding_illegal.append({"action": int(action_text), "response": int(response)})
    record = {
        "case_id": spec["case_id"],
        "source": spec["source"],
        "proof_grade": grade,
        "note": spec["note"],
        "seat": seat,
        "dealer": diagnosis["dealer"],
        "street": diagnosis["street"],
        "pot": diagnosis["pot"],
        "stacks": diagnosis["stacks"],
        "hero_hole": hero,
        "public_cards": shown,
        "absolute_nuts": nuts,
        "equity_clue": None if grade == "hard_dominated" else equity_clue(hero, shown),
        "legal_ids": diagnosis["legal_ids"],
        "legacy_chip_responses": diagnosis["legacy_chip_responses"],
        "deploy_chip_responses": diagnosis["deploy_chip_responses"],
        "encoding_illegal_responses": encoding_illegal,
        "history_events": diagnosis["history_events"],
        "history_action_tokens": diagnosis["history_action_tokens"],
        "history_truncated": diagnosis["history_truncated"],
        "max_seq_len": diagnosis["max_seq_len"],
        "numeric_features": diagnosis["numeric_features"],
        "match_ctx": diagnosis["match_ctx"],
        "kernel_features": {
            key: value for key, value in diagnosis["kernel_features"].items()
        },
        "teacher": missing_teacher_record(
            window,
            "Teacher SuperStudent snapshots are not attached. No own-reach mixture was computed.",
        ),
        "frozen_student": {
            "status": "MISSING",
            "reason": "policy_best.pt is AutoDL-only and was not attached. No student probabilities were invented.",
            "probs": None,
            "unweighted_fill": False,
        },
        "ppo_raw_probs": diagnosis["ppo_raw_probs"],
        "deploy_probs": diagnosis["deploy_probs"],
        "ppo_raw_probs_1e8": _round_key(diagnosis["ppo_raw_probs"]),
        "deploy_probs_1e8": _round_key(diagnosis["deploy_probs"]),
        "guard_changed_argmax": diagnosis["guard_changed_argmax"],
        "illegal_mass_raw": diagnosis["illegal_mass_raw"],
        "illegal_mass_deploy": diagnosis["illegal_mass_deploy"],
        "layers_observed": ["teacher_missing", "student_missing", "ppo_raw", "deploy_guard"],
    }
    record["root_cause_layer"] = _layer_for(record)
    record["baseline_defect"] = (
        record["proof_grade"] == "hard_dominated" and record["ppo_raw_probs"][0] > _FOLD_BASELINE
    )
    record["argmax_raw"] = int(np.argmax(diagnosis["ppo_raw_probs"]))
    record["argmax_deploy"] = int(np.argmax(diagnosis["deploy_probs"]))
    del holes, board
    return record


def _discrete(record) -> dict:
    return {
        "case_id": record["case_id"],
        "source": record["source"],
        "proof_grade": record["proof_grade"],
        "root_cause_layer": record["root_cause_layer"],
        "legal_ids": record["legal_ids"],
        "argmax_raw": record["argmax_raw"],
        "argmax_deploy": record["argmax_deploy"],
        "ppo_raw_probs_1e8": record["ppo_raw_probs_1e8"],
        "deploy_probs_1e8": record["deploy_probs_1e8"],
        "baseline_defect": record["baseline_defect"],
        "history_truncated": record["history_truncated"],
        "encoding_illegal_responses": record["encoding_illegal_responses"],
    }


def run() -> dict:
    judge_sha = hashlib.sha256(_JUDGE_PATH.read_bytes()).hexdigest()
    if judge_sha != _JUDGE_SHA:
        raise SystemExit("botzone_judge.py hash changed")
    window = load_teacher_window(_WINDOW)
    pack = _load_pack()
    meta, weights, _runtime, _guard, _agent = pack
    model = load_torch_policy(weights, meta)
    cases = build_cases()
    records = []
    pt_np = []
    for spec in cases:
        _holes, _board, kernel = _materialize(spec)
        diagnosis = diagnose(kernel, pack)
        diagnosis["kernel_parts"] = (_holes, _board, kernel)
        record = _public_record(spec, diagnosis, window)
        torch_raw = torch_probs(model, diagnosis["batch"])
        gap = float(np.max(np.abs(torch_raw - np.asarray(record["ppo_raw_probs"]))))
        pt_np.append({"case_id": record["case_id"], "max_abs": gap})
        record["pt_np_max_abs"] = gap
        records.append(record)
    second = []
    for spec in cases:
        _holes, _board, kernel = _materialize(spec)
        diagnosis = diagnose(kernel, pack)
        diagnosis["kernel_parts"] = (_holes, _board, kernel)
        second.append(_discrete(_public_record(spec, diagnosis, window)))
    first = [_discrete(record) for record in records]
    if first != second:
        raise SystemExit("shared discrete diagnostics differ across two runs of the same pack")
    max_gap = max(item["max_abs"] for item in pt_np)
    illegal = max(record["illegal_mass_deploy"] for record in records)
    illegal_raw = max(record["illegal_mass_raw"] for record in records)
    defects = [
        {
            "case_id": record["case_id"],
            "layer": "ppo",
            "raw_fold": record["ppo_raw_probs"][0],
            "deploy_fold": record["deploy_probs"][0],
            "note": "Old V6 raw fold on a certified river nuts state exceeds 1%. Baseline defect, not an R03 failure.",
        }
        for record in records
        if record["baseline_defect"]
    ]
    unattributed = [record["case_id"] for record in records if record["root_cause_layer"] == "unattributed"]
    status = "PASS"
    if max_gap > _PT_NP_LIMIT or illegal > 1e-12 or illegal_raw > 1e-12:
        status = "FAIL"
    if any(record["encoding_illegal_responses"] for record in records):
        status = "FAIL"
    summary = {
        "task_id": "R03",
        "status": status,
        "gpu_hours_used": 0,
        "judge_sha256": judge_sha,
        "npz_sha256": _NPZ_SHA,
        "cases": len(records),
        "pt_np_max_abs": max_gap,
        "pt_np_limit": _PT_NP_LIMIT,
        "illegal_mass_deploy_max": illegal,
        "illegal_mass_raw_max": illegal_raw,
        "discrete_diagnostics_identical": True,
        "baseline_defects": defects,
        "unattributed": unattributed,
        "teacher_status": "MISSING",
        "frozen_student_status": "MISSING",
        "hard_dominated": sum(record["proof_grade"] == "hard_dominated" for record in records),
        "research_pending": sum(record["proof_grade"] == "research_pending" for record in records),
        "pt_np_by_case": pt_np,
    }
    return {"summary": summary, "records": records}


def _write(payload) -> None:
    root = Path(__file__).resolve().parent
    suite = root / "counterexample_suite"
    suite.mkdir(parents=True, exist_ok=True)
    for stale in suite.glob("*.json"):
        stale.unlink()
    lines = []
    index = []
    for record in payload["records"]:
        public = {key: value for key, value in record.items() if key != "kernel_parts"}
        lines.append(json.dumps(public, ensure_ascii=False, sort_keys=True))
        (suite / f"{record['case_id']}.json").write_text(
            json.dumps(public, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        index.append(_discrete(record))
    (root / "attribution.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (suite / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "attribution_summary.json").write_text(
        json.dumps(payload["summary"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_report(root, payload)


def _write_report(root: Path, payload) -> None:
    summary = payload["summary"]
    defects = summary["baseline_defects"]
    lines = [
        "# R03 四层归因",
        "",
        f"状态：**{summary['status']}**。GPU 小时 = 0。裁判 SHA256 `{summary['judge_sha256']}`。V6 npz SHA256 `{summary['npz_sha256']}`。",
        "",
        "同一公开信息集上的四层：",
        "",
        "1. Teacher 混合：`MISSING`。SuperStudent / 150..350 快照没有附在这台机器上，没有计算 own-reach，也没有用无权平均补标签。窗口在 `teacher_window_manifest.json`（after_exclusive=149，through=350）。",
        "2. 冻结学生：`MISSING`。`policy_best.pt` 未附带。",
        "3. PPO raw：V6 权重在部署护栏之前的掩码 softmax。",
        "4. 最终部署：同一向量经过 V6 包里的 v10.2 护栏。护栏和 raw 分开记录。",
        "",
        f"PyTorch 与 NumPy 在同一套 FP32 权重、同一批输入上的最大概率绝对误差是 `{summary['pt_np_max_abs']}`（上限 `{summary['pt_np_limit']}`）。",
        f"部署概率的非法动作质量最大值是 `{summary['illegal_mass_deploy_max']}`，raw 是 `{summary['illegal_mass_raw_max']}`。",
        "同一部署包连跑两遍，离散诊断（合法动作、argmax、1e-8 概率刻度、证明等级）一致。",
        "",
        f"反例 {summary['cases']} 个：硬证明河牌绝对坚果 {summary['hard_dominated']} 个，待研究 {summary['research_pending']} 个。每个文件在 `counterexample_suite/`，带 `source` 和 `proof_grade`。",
        "",
        "河牌绝对坚果用官方 `compare_full_cards` 对所有对手底牌穷举，Hero 严格赢下每一手时，弃牌严格劣于跟注或过牌。转牌坚果、顶对、听牌、公共双对和超池再加注只记公平率线索，不写成零弃牌真理。",
        "",
        "## 基线缺陷",
        "",
    ]
    if not defects:
        lines.append("已认证的河牌绝对坚果上，V6 raw 弃牌概率没有超过 1%。没有把旧模型记成基线缺陷。")
    else:
        lines.append("下面这些旧 V6 raw 弃牌超过 1%。这是基线缺陷，不因此判 R03 失败。新模型的 1% 门槛留到 R08/R09/R13。过牌到河牌的同一坚果上 raw 弃牌是 0；超标出现在面对下注的信息集。有的花色置换上护栏把弃牌清掉，有的没有。raw 已经超过 1%，所以层级记在 PPO，不把护栏当成这批缺陷的来源。")
        lines.append("")
        for defect in defects:
            lines.append(
                f"- `{defect['case_id']}`：raw 弃牌 {defect['raw_fold']:.6f}，部署弃牌 {defect['deploy_fold']:.6f}"
            )
    lines.extend(["", "## 尚未归因", ""])
    if not summary["unattributed"]:
        lines.append("没有未归因的反例。")
    else:
        lines.append("这些局面没有可证明的支配动作，Teacher 和冻结学生又都缺失，所以不能把偏差归到蒸馏或 Teacher。")
        lines.append("")
        for case_id in summary["unattributed"]:
            lines.append(f"- `{case_id}`")
    lines.extend(
        [
            "",
            "## 层级",
            "",
            "硬证明局面上，raw 弃牌超过 1% 归到 PPO；raw 不超过、护栏把弃牌抬过 1% 归到护栏。部署发出的筹码响应若被裁判拒绝，归到编码。这次没有发现要先修内核才能继续的系统性规则错误。",
            "",
        ]
    )
    (root / "root_cause_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    payload = run()
    _write(payload)
    print(json.dumps({key: payload["summary"][key] for key in (
        "status", "cases", "pt_np_max_abs", "illegal_mass_deploy_max",
        "baseline_defects", "unattributed",
    )}, ensure_ascii=False))
    if payload["summary"]["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
