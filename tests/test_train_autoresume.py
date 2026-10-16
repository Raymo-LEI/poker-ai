"""T-R1: --resume auto, atomic checkpoint recovery, metrics continuity."""

import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

HAS_RLCARD = importlib.util.find_spec("rlcard") is not None
REPO = Path(__file__).resolve().parents[1]


def _tiny_cfg(tmp_path: Path, num_updates: int, *, checkpoint_every: int = 3):
    from alphanl.transformer.config import ExperimentConfig

    cfg = ExperimentConfig()
    cfg.model.d_model = 32
    cfg.model.nhead = 4
    cfg.model.num_layers = 1
    cfg.model.dim_feedforward = 64
    cfg.model.max_seq_len = 32
    cfg.model.dropout = 0.0
    cfg.device = "cpu"
    cfg.rollout_device = "cpu"
    cfg.seed = 3
    cfg.ppo.num_envs = 2
    cfg.ppo.rollout_steps = 4
    cfg.ppo.epochs = 1
    cfg.ppo.minibatch_size = 4
    cfg.train.num_updates = num_updates
    cfg.train.eval_every = 0
    cfg.train.checkpoint_every = checkpoint_every
    cfg.train.keep_last = 20
    cfg.train.log_every = 1
    cfg.train.amp = False
    cfg.pool.snapshot_every = 1
    cfg.pool.max_size = 3
    cfg.train.out_dir = str(tmp_path / "out")
    cfg.train.checkpoint_dir = str(tmp_path / "ckpt")
    return cfg


def _load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _metric_updates(path: Path):
    with path.open(newline="") as handle:
        return [int(row["update"]) for row in csv.DictReader(handle)]


def _equal_model(left, right) -> None:
    assert left.keys() == right.keys()
    for key in left:
        assert torch.equal(left[key].cpu(), right[key].cpu()), key


def _equal_optimizer(left, right) -> None:
    assert left["state"].keys() == right["state"].keys()
    assert len(left["state"]) > 0
    for pid in left["state"]:
        assert left["state"][pid].keys() == right["state"][pid].keys()
        for key in left["state"][pid]:
            assert torch.equal(
                torch.as_tensor(left["state"][pid][key]).cpu(),
                torch.as_tensor(right["state"][pid][key]).cpu(),
            ), (pid, key)
    assert left["param_groups"] == right["param_groups"]


def test_config_hash_ignores_key_order_and_reports_a_diff():
    from alphanl.transformer.checkpoint import ConfigMismatchError, assert_resume_config, config_hash

    saved = {"seed": 1, "train": {"num_updates": 10, "keep_last": 3}}
    current = {"train": {"keep_last": 3, "num_updates": 10}, "seed": 1}
    assert config_hash(saved) == config_hash(current)
    assert_resume_config(saved, config_hash(saved), current, allow=False, path="ckpt.pt")
    changed = {"seed": 2, "train": {"num_updates": 10, "keep_last": 3}}
    with pytest.raises(ConfigMismatchError, match="seed"):
        assert_resume_config(saved, config_hash(saved), changed, allow=False, path="ckpt.pt")
    assert_resume_config(saved, config_hash(saved), changed, allow=True, path="ckpt.pt")
    assert_resume_config(saved, None, changed, allow=False, path="old.pt")


def test_truncate_metrics_keeps_earlier_rows_byte_for_byte(tmp_path: Path):
    from alphanl.transformer.train import truncate_metrics_csv

    path = tmp_path / "metrics.csv"
    original = "update,loss\n0,1.5\n1,2.5\n2,9\n3,9\n3,8\n"
    path.write_text(original, encoding="utf-8")
    truncate_metrics_csv(path, 2)
    assert path.read_text(encoding="utf-8") == "update,loss\n0,1.5\n1,2.5\n"
    truncate_metrics_csv(path, 2)
    assert path.read_text(encoding="utf-8") == "update,loss\n0,1.5\n1,2.5\n"


def test_keep_last_20_fallback_skips_corrupt_newest(tmp_path: Path):
    from alphanl.transformer.checkpoint import (
        checkpoint_path,
        latest_complete_checkpoint,
        list_checkpoints,
        save_checkpoint,
    )

    model = torch.nn.Linear(2, 2)
    for update in range(25):
        save_checkpoint(
            checkpoint_path(tmp_path, update),
            model=model,
            optimizer=None,
            update=update,
            config={"update": update},
            keep_last=20,
        )
    files = list_checkpoints(tmp_path)
    assert len(files) == 20
    assert files[0].name == "update_0000005.pt"
    assert files[-1].name == "update_0000024.pt"
    files[-1].write_bytes(b"this is not a torch checkpoint")
    found = latest_complete_checkpoint(tmp_path)
    assert found == files[-2]
    assert _load(found)["update"] == 23
    assert _load(found)["config_hash"]


def test_kill_during_write_leaves_the_previous_checkpoint(tmp_path: Path):
    from alphanl.transformer.checkpoint import checkpoint_path, latest_complete_checkpoint, save_checkpoint

    model = torch.nn.Linear(4, 4)
    torch.manual_seed(0)
    with torch.no_grad():
        model.weight.fill_(1.25)
    good = checkpoint_path(tmp_path, 0)
    save_checkpoint(good, model=model, optimizer=None, update=0, config={"seed": 1}, keep_last=20)
    before = _load(good)["model"]["weight"].clone()

    env = os.environ.copy()
    env["ALPHANL_CKPT_FAULT"] = "kill-during-write"
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["CKPT_DIR"] = str(tmp_path)
    code = (
        "import os, torch\n"
        "from alphanl.transformer.checkpoint import checkpoint_path, save_checkpoint\n"
        "save_checkpoint(checkpoint_path(os.environ['CKPT_DIR'], 1), "
        "model=torch.nn.Linear(8, 8), optimizer=None, update=1, config={'seed': 1})\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0, proc.returncode
    assert not (tmp_path / "update_0000001.pt").exists()
    assert (tmp_path / "update_0000001.pt.tmp").is_file()
    found = latest_complete_checkpoint(tmp_path)
    assert found == good
    assert torch.equal(_load(found)["model"]["weight"], before)


def test_auto_with_no_checkpoint_resolves_fresh(tmp_path: Path):
    from alphanl.transformer.train import _resolve_resume

    mode, path = _resolve_resume("auto", tmp_path / "missing")
    assert mode == "auto" and path is None


@pytest.mark.skipif(not HAS_RLCARD, reason="rlcard is not installed")
def test_rlcard_and_pokerkit_rng_roundtrip_deals_the_same_hand(tmp_path: Path):
    from alphanl.transformer.tables import PokerkitTable, RlcardTable

    live = RlcardTable(seed=7, starting_stack=100, small_blind=1, big_blind=2)
    for _ in range(4):
        live.reset()
    blob = tmp_path / "rl.pt"
    torch.save({"s": live.rng_state()}, blob)
    loaded = _load(blob)["s"]
    live.reset()
    expected = (live.hole_cards(0), live.hole_cards(1))
    fresh = RlcardTable(seed=99, starting_stack=100, small_blind=1, big_blind=2)
    fresh.set_rng_state(loaded)
    fresh.reset()
    assert (fresh.hole_cards(0), fresh.hole_cards(1)) == expected

    poker = PokerkitTable(seed=1, starting_stack=100, small_blind=1, big_blind=2)
    for _ in range(5):
        poker.reset()
    torch.save({"s": poker.rng_state()}, tmp_path / "pk.pt")
    poker_state = _load(tmp_path / "pk.pt")["s"]
    poker.reset()
    dealt = (poker.hole_cards(0), poker.hole_cards(1))
    other = PokerkitTable(seed=4, starting_stack=100, small_blind=1, big_blind=2)
    other.set_rng_state(poker_state)
    other.reset()
    assert (other.hole_cards(0), other.hole_cards(1)) == dealt


def test_botzone_rng_roundtrip_keeps_the_next_dealer(tmp_path: Path):
    import random

    from alphanl.botzone.table import BotzoneTable

    live = BotzoneTable(rng=random.Random(1), dealer=0)
    for _ in range(3):
        live.start_hand()
    torch.save({"s": live.rng_state()}, tmp_path / "bz.pt")
    state = _load(tmp_path / "bz.pt")["s"]
    live.start_hand()
    expected = live.holes
    fresh = BotzoneTable(rng=random.Random(9), dealer=1)
    fresh.set_rng_state(state)
    fresh.start_hand()
    assert fresh.holes == expected


def test_resume_auto_refuses_a_different_config(tmp_path: Path):
    from alphanl.transformer.checkpoint import checkpoint_path, save_checkpoint
    from alphanl.transformer.train import ConfigMismatchError, build_model, run_training

    cfg = _tiny_cfg(tmp_path, 4)
    save_checkpoint(
        checkpoint_path(cfg.train.checkpoint_dir, 1),
        model=build_model(cfg),
        optimizer=None,
        update=1,
        config=cfg.to_dict(),
        keep_last=20,
    )
    cfg.seed = 99
    with pytest.raises(ConfigMismatchError, match="allow-config-mismatch"):
        run_training(cfg, resume="auto")


def test_cli_resume_auto_mismatch_exits_2(tmp_path: Path):
    from alphanl.transformer.checkpoint import checkpoint_path, save_checkpoint
    from alphanl.transformer.train import build_model

    cfg = _tiny_cfg(tmp_path, 4)
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
    save_checkpoint(
        checkpoint_path(cfg.train.checkpoint_dir, 1),
        model=build_model(cfg),
        optimizer=None,
        update=1,
        config=cfg.to_dict(),
        keep_last=20,
    )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["seed"] = 99
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "train_transformer.py"),
            "--config",
            str(config_path),
            "--resume",
            "auto",
            "--updates",
            "4",
        ],
        env=env,
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2, proc.stderr
    assert "config hash" in proc.stderr


@pytest.mark.skipif(not HAS_RLCARD, reason="rlcard is not installed")
def test_resume_auto_matches_control_that_resets_at_the_boundary(tmp_path: Path):
    """6 updates with a reset after update 2 == 3 updates, kill, --resume auto for 3 more.

    Mid-hand state is not restored. The control calls the same RNG-restore and
    ``reset`` after the checkpoint at update 2 (``reset_envs_after_update``).
    The split run stops after that checkpoint (``stop_after_update``), which is
    the kill between updates, then starts a new ``run_training(..., resume='auto')``.
    """

    import alphanl.transformer.botzone_table  # noqa: F401
    import alphanl.transformer.train as train_mod

    try:
        from torch.utils.tensorboard import SummaryWriter  # noqa: F401
    except ImportError:
        pass

    control = _tiny_cfg(tmp_path / "control", 6)
    train_mod.run_training(control, reset_envs_after_update=2)
    control_blob = _load(Path(control.train.checkpoint_dir) / "update_0000005.pt")

    split = _tiny_cfg(tmp_path / "split", 6)
    train_mod.run_training(split, stop_after_update=2)
    metrics = Path(split.train.out_dir) / "metrics.csv"
    assert _metric_updates(metrics) == [0, 1, 2]
    lines = metrics.read_text(encoding="utf-8").splitlines()
    gap = lines[-1].split(",")
    gap[0] = "3"
    extra_three = ",".join(gap)
    gap[0] = "7"
    extra_seven = ",".join(gap)
    with metrics.open("a", encoding="utf-8", newline="") as handle:
        handle.write(extra_three + "\n" + extra_seven + "\n")
    assert _metric_updates(metrics) == [0, 1, 2, 3, 7]

    train_mod.run_training(split, resume="auto")
    resumed = _load(Path(split.train.checkpoint_dir) / "update_0000005.pt")
    assert control_blob["update"] == resumed["update"] == 5
    _equal_model(control_blob["model"], resumed["model"])
    _equal_optimizer(control_blob["optimizer"], resumed["optimizer"])
    assert _metric_updates(metrics) == [0, 1, 2, 3, 4, 5]


@pytest.mark.skipif(not HAS_RLCARD, reason="rlcard is not installed")
def test_corrupt_newest_checkpoint_falls_back_during_resume(tmp_path: Path):
    import alphanl.transformer.botzone_table  # noqa: F401
    import alphanl.transformer.train as train_mod

    try:
        from torch.utils.tensorboard import SummaryWriter  # noqa: F401
    except ImportError:
        pass

    cfg = _tiny_cfg(tmp_path, 3, checkpoint_every=1)
    train_mod.run_training(cfg, stop_after_update=1)
    ckpt = Path(cfg.train.checkpoint_dir)
    newest = ckpt / "update_0000001.pt"
    previous = ckpt / "update_0000000.pt"
    assert newest.is_file() and previous.is_file()
    previous_update = _load(previous)["update"]
    newest.write_bytes(b"torn-zip")
    metrics = Path(cfg.train.out_dir) / "metrics.csv"
    assert _metric_updates(metrics) == [0, 1]
    train_mod.run_training(cfg, resume="auto")
    assert _metric_updates(metrics) == [0, 1, 2]
    final = _load(ckpt / "update_0000002.pt")
    assert previous_update == 0
    assert final["update"] == 2
    assert not (ckpt / "update_0000001.pt").read_bytes().startswith(b"torn-zip")


PPO_LR = 1e-4
BC_LR = 1e-3


def _import_training():
    import alphanl.transformer.botzone_table  # noqa: F401
    import alphanl.transformer.train as train_mod

    try:
        from torch.utils.tensorboard import SummaryWriter  # noqa: F401
    except ImportError:
        pass
    return train_mod


def _write_bc(path: Path, cfg) -> None:
    """BC file with Adam lr 1e-3. PPO must not load that optimizer."""

    from alphanl.transformer.checkpoint import save_checkpoint
    from alphanl.transformer.train import build_model

    model = build_model(cfg)
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(0.02)
    optimizer = torch.optim.Adam(model.parameters(), lr=BC_LR, eps=cfg.ppo.adam_eps)
    first = next(model.parameters())
    first.grad = torch.ones_like(first)
    optimizer.step()
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        update=-1,
        config={"stage": "bc"},
    )


def _launch(cfg, bc: Path, **kwargs):
    """Call run_training the way train_autoresume.ps1 would for this directory."""

    from alphanl.transformer.checkpoint import choose_launch_mode, ppo_launch_args
    from alphanl.transformer.train import run_training

    mode = choose_launch_mode(cfg.train.checkpoint_dir)
    args = ppo_launch_args(cfg.train.checkpoint_dir, bc)
    if mode == "fresh":
        assert args == ["--init-weights", str(bc), "--allow-cross-backend-init"]
        run_training(cfg, init_weights=bc, allow_cross_backend_init=True, **kwargs)
    else:
        assert mode == "resume"
        assert args == ["--resume", "auto"]
        run_training(cfg, resume="auto", **kwargs)
    return mode


def test_empty_or_unreadable_dir_chooses_fresh_init_weights(tmp_path: Path):
    from alphanl.transformer.checkpoint import choose_launch_mode, ppo_launch_args

    bc = tmp_path / "bc.pt"
    assert choose_launch_mode(tmp_path / "missing") == "fresh"
    empty = tmp_path / "empty"
    empty.mkdir()
    assert choose_launch_mode(empty) == "fresh"
    assert ppo_launch_args(empty, bc) == [
        "--init-weights",
        str(bc),
        "--allow-cross-backend-init",
    ]
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "update_0000004.pt").write_bytes(b"not-a-checkpoint")
    (bad / "update_0000004.pt.tmp").write_bytes(b"partial")
    assert choose_launch_mode(bad) == "fresh"


def test_corrupt_newest_with_older_complete_chooses_resume(tmp_path: Path):
    from alphanl.transformer.checkpoint import (
        checkpoint_path,
        choose_launch_mode,
        ppo_launch_args,
        save_checkpoint,
    )

    model = torch.nn.Linear(2, 2)
    for update in (3, 4):
        save_checkpoint(
            checkpoint_path(tmp_path, update),
            model=model,
            optimizer=None,
            update=update,
            config={"update": update},
            keep_last=20,
        )
    newest = checkpoint_path(tmp_path, 4)
    newest.write_bytes(b"torn")
    bc = tmp_path / "bc.pt"
    assert choose_launch_mode(tmp_path) == "resume"
    assert ppo_launch_args(tmp_path, bc) == ["--resume", "auto"]


def test_launch_mode_cli_prints_one_word(tmp_path: Path):
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "train_launch_mode.py"),
            "--checkpoint-dir",
            str(tmp_path),
        ],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "fresh"


@pytest.mark.skipif(not HAS_RLCARD, reason="rlcard is not installed")
def test_relaunch_keeps_ppo_adam_not_bc_init(tmp_path: Path):
    """3 PPO updates, stop, relaunch via the script's mode choice.

    The second launch must resume PPO weights and Adam (lr 1e-4). It must
    not pass --init-weights, which would put the BC weights back and drop
    the PPO Adam (BC was saved at lr 1e-3). The control is one process that
    also resets tables after update 2, because --resume auto deals a new
    hand instead of restoring the mid-hand.
    """

    train_mod = _import_training()
    bc = tmp_path / "bc.pt"
    _write_bc(bc, _tiny_cfg(tmp_path / "shape", 1))
    bc_blob = _load(bc)
    assert bc_blob["optimizer"]["param_groups"][0]["lr"] == BC_LR

    control = _tiny_cfg(tmp_path / "control", 6)
    control.ppo.lr = PPO_LR
    train_mod.run_training(
        control,
        init_weights=bc,
        allow_cross_backend_init=True,
        reset_envs_after_update=2,
    )

    split = _tiny_cfg(tmp_path / "split", 6)
    split.ppo.lr = PPO_LR
    assert _launch(split, bc, stop_after_update=2) == "fresh"
    assert _launch(split, bc) == "resume"

    control_blob = _load(Path(control.train.checkpoint_dir) / "update_0000005.pt")
    resumed = _load(Path(split.train.checkpoint_dir) / "update_0000005.pt")
    _equal_model(control_blob["model"], resumed["model"])
    _equal_optimizer(control_blob["optimizer"], resumed["optimizer"])
    assert resumed["optimizer"]["param_groups"][0]["lr"] == PPO_LR
    assert len(resumed["optimizer"]["state"]) > 0
    weight_key = next(iter(resumed["model"]))
    assert not torch.equal(resumed["model"][weight_key].cpu(), bc_blob["model"][weight_key].cpu())


@pytest.mark.skipif(not HAS_RLCARD, reason="rlcard is not installed")
def test_corrupt_newest_relaunch_uses_older_ppo_not_bc(tmp_path: Path):
    """Newest file torn, older file complete: launcher stays on --resume auto.

    The continued weights match a run that had only that older file, and
    they do not match starting over from the BC checkpoint.
    """

    train_mod = _import_training()
    bc = tmp_path / "bc.pt"
    _write_bc(bc, _tiny_cfg(tmp_path / "shape", 1))

    def make(path: Path):
        cfg = _tiny_cfg(path, 3, checkpoint_every=1)
        cfg.ppo.lr = PPO_LR
        return cfg

    ref = make(tmp_path / "ref")
    split = make(tmp_path / "split")
    assert _launch(ref, bc, stop_after_update=1) == "fresh"
    assert _launch(split, bc, stop_after_update=1) == "fresh"
    ref_old = _load(Path(ref.train.checkpoint_dir) / "update_0000000.pt")
    split_old = _load(Path(split.train.checkpoint_dir) / "update_0000000.pt")
    _equal_model(ref_old["model"], split_old["model"])
    _equal_optimizer(ref_old["optimizer"], split_old["optimizer"])
    assert ref_old["optimizer"]["param_groups"][0]["lr"] == PPO_LR

    (Path(ref.train.checkpoint_dir) / "update_0000001.pt").unlink()
    (Path(split.train.checkpoint_dir) / "update_0000001.pt").write_bytes(b"torn-zip")
    train_mod.run_training(ref, resume="auto")
    assert _launch(split, bc) == "resume"

    ref_final = _load(Path(ref.train.checkpoint_dir) / "update_0000002.pt")
    split_final = _load(Path(split.train.checkpoint_dir) / "update_0000002.pt")
    _equal_model(ref_final["model"], split_final["model"])
    _equal_optimizer(ref_final["optimizer"], split_final["optimizer"])
    assert split_final["optimizer"]["param_groups"][0]["lr"] == PPO_LR
    assert split_final["optimizer"]["param_groups"][0]["lr"] != BC_LR

    fresh = make(tmp_path / "fresh")
    assert _launch(fresh, bc) == "fresh"
    fresh_final = _load(Path(fresh.train.checkpoint_dir) / "update_0000002.pt")
    assert any(
        not torch.equal(split_final["model"][key].cpu(), fresh_final["model"][key].cpu())
        for key in split_final["model"]
    )
