"""NumPy export, forward parity, observation parity, and model fallback.

The development weights are a random initialisation of
``configs/transformer/botzone_laptop_8gb_maskfold.json``. The repository
does not contain ``update_0000999.pt``.
"""

import copy
import gzip
import hashlib
import importlib.util
import json
import os
import random
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "botzone_bots" / "model"
FIXTURE = ROOT / "tests" / "fixtures" / "botzone_real.jsonl.gz"
LEGAL_PY = ROOT / "src" / "alphanl" / "botzone" / "legal.py"
CONFIG = ROOT / "configs" / "transformer" / "botzone_laptop_8gb_maskfold.json"
PUBLIC_AT = {0: 0, 1: 3, 2: 4, 3: 5}
_MODEL_NAMES = ("betting", "forward", "observe", "weights", "reconstruct", "decide")

sys.path.insert(0, str(ROOT / "eval"))

from botzone_referee_adapter import (  # noqa: E402
    new_dealt_holdem,
    referee_is_legal,
    state_from_holdem,
)


def _load_export():
    path = ROOT / "tools" / "export_numpy_weights.py"
    spec = importlib.util.spec_from_file_location("export_numpy_weights", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model_api():
    """Import the zip entry without leaving its names over the rule bot."""

    saved = {}
    for name in _MODEL_NAMES:
        if name in sys.modules:
            saved[name] = sys.modules.pop(name)
    sys.path.insert(0, str(MODEL_DIR))
    try:
        import decide
        import forward
        import observe
        import weights

        api = {
            "decide": decide,
            "forward": forward,
            "observe": observe,
            "weights": weights,
        }
        for name in _MODEL_NAMES:
            sys.modules.pop(name, None)
        return api
    finally:
        sys.path.remove(str(MODEL_DIR))
        sys.modules.update(saved)


def _request_from_game(game, hole, public):
    return {
        "num_players": int(game.num_players),
        "dealer_id": int(game.dealer_idx),
        "my_id": int(game.round_idx),
        "my_chips": int(game.player_chips[game.round_idx]),
        "my_cards": list(hole),
        "public_cards": list(public),
        "history": copy.deepcopy(game.history),
    }


def _load_hands():
    hands = []
    with gzip.open(str(FIXTURE), "rt") as handle:
        for line in handle:
            hands.append(json.loads(line))
    return hands


def _spec_action_id(response, to_call_before, chips_added, stack_before):
    if response == -1:
        return 0
    if chips_added > 0 and stack_before - chips_added <= 0:
        return 4
    if response == -2:
        return 4
    if response == 0:
        return 1
    if response > 0:
        return 2
    raise AssertionError(response)


def _chips_added(state, response, to_call):
    if response == -1:
        return 0
    if response == -2:
        return int(state.stacks[state.actor])
    if response == 0:
        return int(to_call(state))
    return int(response)


def _training_batch(request, tokenizer):
    """Observation the training tokenizer emits for this judge request."""

    from alphanl.botzone import actions as botzone_actions
    from alphanl.botzone.legal import apply, new_hand, to_call
    from alphanl.transformer.botzone_table import botzone_id_to_card
    from alphanl.transformer.history import make_action_event
    from alphanl.transformer.schema import HandSnapshot

    dealer = int(request["dealer_id"])
    my_id = int(request["my_id"])
    state = new_hand(dealer)
    events = []
    sb_seat = (dealer + 1) % 2
    for entry in request["history"]:
        response = int(entry["action"])
        owed = int(to_call(state))
        stack_before = int(state.stacks[state.actor])
        chips = _chips_added(state, response, to_call)
        action_id = _spec_action_id(response, owed, chips, stack_before)
        events.append(
            make_action_event(
                street=int(state.street),
                actor_seat=int(state.actor),
                me_seat=my_id,
                sb_seat=sb_seat,
                action_id=action_id,
                action_type=None,
                pot_before=int(state.pot),
                chips_added=chips,
                to_call_before=owed,
                stack_before=stack_before,
            )
        )
        state = apply(state, response)
    hole = tuple(botzone_id_to_card(card) for card in request["my_cards"])
    board = tuple(botzone_id_to_card(card) for card in request["public_cards"])
    snap = HandSnapshot(
        events=tuple(events),
        hole_cards=hole,
        board_cards=board,
        me_seat=my_id,
        sb_seat=sb_seat,
        street=int(state.street),
        pot=float(state.pot),
        stacks=(float(state.stacks[0]), float(state.stacks[1])),
        to_call=float(to_call(state)),
        legal_action_ids=tuple(botzone_actions.legal_action_ids(state)),
        big_blind=100.0,
        starting_stack=20000.0,
    )
    return state, tokenizer.encode(snap)


def _arrays_equal(left, right):
    return left.shape == right.shape and np.array_equal(left, right)


def _state_problems(view, train_state, bot_state):
    problems = []
    if tuple(view.chips) != tuple(train_state.stacks):
        problems.append("chips")
    if int(view.pot) != int(train_state.pot):
        problems.append("pot")
    if int(view.street) != int(train_state.street):
        problems.append("street")
    if int(view.round_raise) != int(train_state.bound_b) // 2:
        problems.append("round_raise")
    if tuple(view.street_put) != tuple(train_state.street_commit):
        problems.append("street_commit")
    if tuple(bot_state.stacks) != tuple(train_state.stacks):
        problems.append("bot_stacks")
    if int(bot_state.pot) != int(train_state.pot):
        problems.append("bot_pot")
    if int(bot_state.street) != int(train_state.street):
        problems.append("bot_street")
    if int(bot_state.bound_b) != int(train_state.bound_b):
        problems.append("bot_bound")
    if int(bot_state.actor) != int(train_state.actor):
        problems.append("bot_actor")
    if tuple(bot_state.all_in) != tuple(train_state.all_in):
        problems.append("bot_all_in")
    if tuple(bot_state.folded) != tuple(train_state.folded):
        problems.append("bot_folded")
    if tuple(bot_state.street_commit) != tuple(train_state.street_commit):
        problems.append("bot_commit")
    return problems


def _obs_problems(bot_batch, train_batch):
    problems = []
    train_np = {key: value.detach().cpu().numpy() for key, value in train_batch.items()}
    for key in ("token_ids", "type_ids", "numeric", "pad_mask", "legal_mask"):
        if not _arrays_equal(np.asarray(bot_batch[key]), train_np[key]):
            problems.append(key)
    return problems


def replay_report():
    """41 real matches: state, observation, and a legal model action."""

    api = load_model_api()
    api["weights"].clear_cache()
    from alphanl.transformer.tokenizer import BettingHistoryTokenizer

    tokenizer = BettingHistoryTokenizer(64)
    weights, meta = api["weights"].load_weights()
    state_mismatches = []
    observation_mismatches = []
    illegal = []
    decisions = 0
    ready = []
    for hand in _load_hands():
        game = new_dealt_holdem(int(hand["d"]))
        for pid, response, street in hand["a"]:
            decisions += 1
            if int(game.round_idx) != int(pid) or int(game.round) != int(street):
                state_mismatches.append({"kind": "order", "index": decisions})
                break
            need = PUBLIC_AT[int(game.round)]
            request = _request_from_game(game, hand["c"][pid], list(hand["b"][:need]))
            view = None
            judge_state = state_from_holdem(game)
            try:
                view = api["decide"].reconstruct(request)
                obs = api["observe"].build_observation(request, max_seq_len=64)
                train_state, train_batch = _training_batch(request, tokenizer)
            except Exception as exc:
                state_mismatches.append({"kind": "raise", "index": decisions, "error": str(exc)})
                action = api["decide"]._fallback(view, request)
                if not referee_is_legal(judge_state, action):
                    illegal.append({"index": decisions, "action": action, "kind": "fallback"})
            else:
                problems = _state_problems(view, train_state, obs.state)
                if problems:
                    state_mismatches.append({"kind": "state", "fields": problems, "index": decisions})
                obs_problems = _obs_problems(obs.batch, train_batch)
                if obs_problems:
                    observation_mismatches.append(
                        {"kind": "obs", "fields": obs_problems, "index": decisions}
                    )
                ready.append((request, view, obs, judge_state))
            result = game.player_action(int(response))
            if result:
                break
    _fill_actions(api, weights, meta, ready, illegal)
    return {
        "decisions": decisions,
        "state_mismatches": state_mismatches,
        "observation_mismatches": observation_mismatches,
        "illegal": illegal,
    }


def _fill_actions(api, weights, meta, ready, illegal):
    decide = api["decide"]
    forward = api["forward"]
    checked = 0
    chunk = 32
    for start in range(0, len(ready), chunk):
        block = ready[start : start + chunk]
        batch = {
            key: np.concatenate([item[2].batch[key] for item in block], axis=0)
            for key in ("token_ids", "type_ids", "numeric", "pad_mask", "legal_mask")
        }
        owed = np.asarray([item[2].to_call for item in block], dtype=np.float32)
        logits = forward.forward_logits(weights, meta, batch, owed)
        for row, (request, view, obs, judge_state) in enumerate(block):
            action = decide.response_from_logits(logits[row], obs, view, True, None)
            if checked < 41:
                alone = decide.choose_response(request, deterministic=True)
                if int(alone) != int(action):
                    illegal.append({"index": "spot", "action": [int(alone), int(action)]})
                checked += 1
            if not referee_is_legal(judge_state, int(action)):
                illegal.append({"index": start + row, "action": int(action)})


def _export_random(tmp_path):
    export = _load_export()
    dest = tmp_path / "model.npz"
    ckpt = tmp_path / "random_init.pt"
    export.export_random(dest, CONFIG, seed=1, write_checkpoint=ckpt)
    return dest, ckpt


def test_export_records_architecture_mask_and_legal_hash(tmp_path):
    dest, ckpt = _export_random(tmp_path)
    size = dest.stat().st_size
    assert 2_000_000 < size < 3_000_000
    archive = np.load(dest)
    try:
        meta = json.loads(archive["meta_json"].tobytes().decode("utf-8"))
        for key in archive.files:
            if key == "meta_json":
                continue
            assert archive[key].dtype == np.float32
        count = sum(int(archive[key].size) for key in archive.files if key != "meta_json")
    finally:
        archive.close()
    assert meta["mask_free_fold"] is True
    assert meta["sample"] is False
    assert meta["source"]["random_init"] is True
    assert meta["arch"]["d_model"] == 128
    assert meta["arch"]["nhead"] == 4
    assert meta["arch"]["num_layers"] == 4
    assert meta["arch"]["dim_feedforward"] == 256
    assert meta["arch"]["max_seq_len"] == 64
    assert meta["environment"]["version"] == "botzone-simple-50-100-20000"
    assert meta["environment"]["backend"] == "botzone"
    assert meta["environment"]["big_blind"] == 100
    assert meta["legal_py_sha256"] == hashlib.sha256(LEGAL_PY.read_bytes()).hexdigest()
    assert count == meta["parameter_count"]
    assert meta["weight_bytes_fp32"] == count * 4

    export = _load_export()
    again = tmp_path / "from_ckpt.npz"
    export.export_checkpoint(ckpt, again, CONFIG)
    left = np.load(dest)
    right = np.load(again)
    try:
        assert np.array_equal(left["policy_head.weight"], right["policy_head.weight"])
        assert np.array_equal(left["token_emb.weight"], right["token_emb.weight"])
    finally:
        left.close()
        right.close()


def test_numpy_forward_matches_torch_on_1000_states(tmp_path):
    import torch
    from alphanl.transformer.decision_stats import policy_action_mask
    from alphanl.transformer.model import PokerTransformer
    from alphanl.transformer.tokenizer import BettingHistoryTokenizer
    from alphanl.transformer.train import build_model, set_seed
    from alphanl.transformer.config import ExperimentConfig
    from alphanl.transformer.botzone_table import BotzoneEnvTable

    dest, _ckpt = _export_random(tmp_path)
    os.environ["POKER_WEIGHTS"] = str(dest)
    api = load_model_api()
    api["weights"].clear_cache()
    weights, meta = api["weights"].load_weights()
    cfg = ExperimentConfig.from_json(CONFIG)
    set_seed(1)
    model = build_model(cfg)
    model.load_state_dict(torch.load(_ckpt, map_location="cpu", weights_only=False)["model"])
    model.eval()
    assert isinstance(model, PokerTransformer)

    tokenizer = BettingHistoryTokenizer(64)
    rng = random.Random(0)
    seen = set()
    states = []
    table = BotzoneEnvTable(seed=2, dealer=0)
    hands = 0
    while len(states) < 1000 and hands < 4000:
        hands += 1
        if hands % 2 == 0:
            table = BotzoneEnvTable(seed=hands, dealer=hands % 2)
        table.reset()
        guard = 0
        while not table.is_over() and guard < 40 and len(states) < 1000:
            guard += 1
            snap = table.snapshot(table.current_player())
            batch = tokenizer.encode(snap)
            key = batch["token_ids"].numpy().tobytes() + batch["numeric"].numpy().tobytes()
            if key not in seen:
                seen.add(key)
                states.append((batch, float(snap.to_call)))
            legal = [i for i, flag in enumerate(batch["legal_mask"][0].tolist()) if flag > 0.5]
            table.step(int(rng.choice(legal)))
    assert len(states) >= 1000
    assert any(owed == 0.0 for _batch, owed in states)
    assert any(owed > 0.0 for _batch, owed in states)

    raw_err = 0.0
    masked_err = 0.0
    chunk = 64
    for start in range(0, 1000, chunk):
        block = states[start : start + chunk]
        token_ids = torch.cat([item[0]["token_ids"] for item in block], dim=0)
        type_ids = torch.cat([item[0]["type_ids"] for item in block], dim=0)
        numeric = torch.cat([item[0]["numeric"] for item in block], dim=0)
        pad_mask = torch.cat([item[0]["pad_mask"] for item in block], dim=0)
        legal_mask = torch.cat([item[0]["legal_mask"] for item in block], dim=0)
        owed = torch.tensor([item[1] for item in block], dtype=torch.float32)
        torch_batch = {
            "token_ids": token_ids,
            "type_ids": type_ids,
            "numeric": numeric,
            "pad_mask": pad_mask,
            "legal_mask": legal_mask,
        }
        with torch.no_grad():
            torch_raw, _value = model(torch_batch, mask_illegal=False)
            policy = dict(torch_batch)
            policy["legal_mask"] = policy_action_mask(legal_mask, owed, True)
            torch_masked, _value = model(policy, mask_illegal=True)
        np_batch = {key: value.numpy() for key, value in torch_batch.items()}
        np_raw, _np_value = api["forward"].raw_forward(weights, meta, np_batch)
        np_masked = api["forward"].forward_logits(weights, meta, np_batch, owed.numpy())
        raw_err = max(raw_err, float(np.max(np.abs(np_raw - torch_raw.numpy()))))
        masked_err = max(masked_err, float(np.max(np.abs(np_masked - torch_masked.numpy()))))
    assert raw_err < 1e-4
    assert masked_err < 1e-4


def test_replay_observation_is_bit_identical_and_actions_are_legal(tmp_path):
    dest, _ckpt = _export_random(tmp_path)
    os.environ["POKER_WEIGHTS"] = str(dest)
    report = replay_report()
    assert report["decisions"] > 7000
    assert report["state_mismatches"] == []
    assert report["observation_mismatches"] == []
    assert report["illegal"] == []


def test_ported_betting_matches_legal_on_random_hands():
    from alphanl.botzone.actions import legal_action_ids, response_for_action
    from alphanl.botzone.legal import apply, new_hand

    saved = {name: sys.modules.get(name) for name in _MODEL_NAMES}
    for name in _MODEL_NAMES:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(MODEL_DIR))
    try:
        import betting as port
    finally:
        sys.path.remove(str(MODEL_DIR))
        for name in _MODEL_NAMES:
            sys.modules.pop(name, None)
        for name, module in saved.items():
            if module is not None:
                sys.modules[name] = module

    rng = random.Random(3)
    for dealer in (0, 1):
        for _hand in range(15):
            left = new_hand(dealer)
            right = port.new_hand(dealer)
            for _step in range(12):
                assert left.stacks == right.stacks
                assert left.street_commit == right.street_commit
                assert left.pot == right.pot
                assert left.bound_b == right.bound_b
                assert left.street == right.street
                assert left.actor == right.actor
                assert left.all_in == right.all_in
                assert left.hand_over == right.hand_over
                assert legal_action_ids(left) == port.legal_action_ids(right)
                if left.hand_over:
                    break
                action = rng.choice(legal_action_ids(left))
                response = response_for_action(left, action)
                left = apply(left, response)
                right = port.apply(right, response)


def test_fallback_on_missing_weights_nan_and_bad_meta(tmp_path):
    api = load_model_api()
    game = new_dealt_holdem(0)
    hole = [card.to_int() for card in game.player_cards[game.round_idx]]
    request = _request_from_game(game, hole, [])
    api["weights"].clear_cache()
    os.environ["POKER_WEIGHTS"] = str(tmp_path / "missing.npz")
    assert api["decide"].choose_response(request) == 0
    assert api["decide"].judge_accepts(api["decide"].reconstruct(request), 0)

    dest, _ckpt = _export_random(tmp_path)
    nan_path = tmp_path / "nan.npz"
    _rewrite(dest, nan_path, bias_nan=True, vocab=None)
    api["weights"].clear_cache()
    os.environ["POKER_WEIGHTS"] = str(nan_path)
    assert api["decide"].choose_response(request) == 0

    bad_path = tmp_path / "badmeta.npz"
    _rewrite(dest, bad_path, bias_nan=False, vocab=1)
    api["weights"].clear_cache()
    os.environ["POKER_WEIGHTS"] = str(bad_path)
    assert api["decide"].choose_response(request) == 0

    broken = dict(request)
    broken["my_chips"] = 1
    assert api["decide"].choose_response(broken) == -1
    assert api["decide"]._fallback(None, None) == -1
    import io

    old_in, old_out = sys.stdin, sys.stdout
    sys.stdin = io.StringIO("{}\n")
    sys.stdout = io.StringIO()
    try:
        api["decide"].run()
        text = sys.stdout.getvalue()
    finally:
        sys.stdin, sys.stdout = old_in, old_out
    assert json.loads(text)["response"] == -1


def test_sampling_stays_on_legal_ids(tmp_path):
    dest, _ckpt = _export_random(tmp_path)
    os.environ["POKER_WEIGHTS"] = str(dest)
    os.environ["POKER_SAMPLE"] = "1"
    api = load_model_api()
    api["weights"].clear_cache()
    try:
        game = new_dealt_holdem(0)
        # Walk to a street where several ids are legal by calling once if needed.
        request = _request_from_game(
            game,
            [card.to_int() for card in game.player_cards[game.round_idx]],
            [],
        )
        seen = set()
        rng = random.Random(4)
        for _draw in range(40):
            action = api["decide"].choose_response(request, deterministic=False, rng=rng)
            assert api["decide"].judge_accepts(api["decide"].reconstruct(request), action)
            seen.add(action)
        assert len(seen) >= 2
    finally:
        os.environ.pop("POKER_SAMPLE", None)


def test_mask_free_fold_removes_fold_when_nothing_is_owed():
    api = load_model_api()
    logits = np.array([[5.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    legal = np.array([[1.0, 1.0, 1.0, 0.0, 1.0]], dtype=np.float32)
    masked = api["forward"].mask_logits(logits, legal, np.array([0.0], dtype=np.float32), True)
    assert float(masked[0, 0]) < -1.0e8
    assert int(np.argmax(masked[0])) == 2
    facing = api["forward"].mask_logits(logits, legal, np.array([50.0], dtype=np.float32), True)
    assert float(facing[0, 0]) == pytest.approx(5.0)
    assert int(np.argmax(facing[0])) == 0


def test_pack_zip_is_torch_free(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "pack_botzone_model", ROOT / "tools" / "pack_botzone_model.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dest = module.pack(tmp_path / "model.zip")
    with zipfile.ZipFile(dest) as archive:
        names = set(archive.namelist())
        assert "__main__.py" in names
        assert "decide.py" in names
        assert "reconstruct.py" in names
        assert "forward.py" in names
        assert not any(name.endswith(".npz") or name.endswith(".pt") for name in names)
        for name in names:
            text = archive.read(name).decode("utf-8")
            assert "import torch" not in text
            assert "from torch" not in text


def _rewrite(src, dest, bias_nan, vocab):
    archive = np.load(src)
    payload = {key: np.array(archive[key]) for key in archive.files}
    archive.close()
    if bias_nan:
        bias = payload["policy_head.bias"].copy()
        bias[:] = np.nan
        payload["policy_head.bias"] = bias
    if vocab is not None:
        meta = json.loads(payload["meta_json"].tobytes().decode("utf-8"))
        meta["arch"]["vocab_size"] = vocab
        raw = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8")
        payload["meta_json"] = np.frombuffer(raw, dtype=np.uint8).copy()
    np.savez(str(dest), **payload)
