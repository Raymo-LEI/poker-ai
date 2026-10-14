"""Control-group conv net, config switch, and behavior-cloning warm start. CPU only."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from alphanl.nl_obs import encode_observation, masked_softmax_tf
from alphanl.transformer.bc import (
    bc_losses,
    bc_update,
    dataset_for_encoder,
    legalize_teacher,
    save_bc_checkpoint,
    transformer_tensors_from_npz,
)
from alphanl.transformer.checkpoint import load_checkpoint
from alphanl.transformer.config import ExperimentConfig
from alphanl.transformer.conv_net import (
    AlphaNLConvNet,
    export_tf_weights,
    load_tf_weights,
    tf_same_pad,
)
from alphanl.transformer.conv_obs import AlphaNLObserver
from alphanl.transformer.model import PokerTransformer
from alphanl.transformer.ppo import ppo_update
from alphanl.transformer.train import build_model, build_optimizer, collect_rollout, describe_run, flatten_rollout


ROOT = Path(__file__).resolve().parents[1]


def _conv_batch(batch_size=4, legal=None):
    card = torch.zeros(batch_size, 4, 13, 6)
    action = torch.zeros(batch_size, 4, 5, 25)
    card[:, 0, 12, 0] = 1
    card[:, 1, 11, 0] = 1
    if legal is None:
        legal_mask = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0]).expand(batch_size, -1).clone()
    else:
        legal_mask = legal.expand(batch_size, -1).clone()
    return {
        "card_info": card,
        "action_info": action,
        "extra_info": torch.tensor([[99.0, 98.0]]).expand(batch_size, -1).clone(),
        "legal_mask": legal_mask,
    }


def _token_batch(batch_size=8, seq=16):
    return {
        "token_ids": torch.randint(1, 40, (batch_size, seq)),
        "type_ids": torch.randint(1, 8, (batch_size, seq)),
        "numeric": torch.randn(batch_size, seq, 4),
        "pad_mask": torch.zeros(batch_size, seq, dtype=torch.bool),
        "legal_mask": torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0]).expand(batch_size, -1).clone(),
    }


def test_config_switch_selects_encoder_and_shapes():
    laptop = ExperimentConfig.from_json(ROOT / "configs" / "transformer" / "laptop_8gb.json")
    conv_cfg = ExperimentConfig.from_json(ROOT / "configs" / "transformer" / "laptop_8gb_conv.json")
    assert laptop.encoder == "transformer"
    assert conv_cfg.encoder == "alphanl_conv"
    assert "encoder=transformer" in describe_run(laptop)
    assert "encoder=alphanl_conv" in describe_run(conv_cfg)

    transformer = build_model(laptop)
    conv = build_model(conv_cfg)
    assert isinstance(transformer, PokerTransformer)
    assert isinstance(conv, AlphaNLConvNet)

    logits_t, value_t = transformer(_token_batch(seq=laptop.model.max_seq_len))
    logits_c, value_c = conv(_conv_batch())
    assert logits_t.shape == (8, 5) and value_t.shape == (8,)
    assert logits_c.shape == (4, 5) and value_c.shape == (4,)
    assert torch.isfinite(logits_c).all() and torch.isfinite(value_c).all()
    probs = torch.softmax(logits_c, dim=-1)
    assert float(probs[:, 2:4].max().detach()) < 1e-6
    assert torch.allclose(probs.sum(-1), torch.ones(4), atol=1e-5)

    switched = ExperimentConfig()
    switched.apply_overrides(encoder="alphanl_conv")
    assert isinstance(build_model(switched), AlphaNLConvNet)


def test_tf_same_pad_and_flatten_and_extra_info_ignored():
    x = torch.zeros(1, 1, 4, 13)
    padded = tf_same_pad(x, (3, 3), (2, 2))
    # H: out=2, pad=1, top=0, bottom=1. W: out=7, pad=2, left=1, right=1.
    assert tuple(padded.shape) == (1, 1, 5, 15)
    assert float(padded[0, 0, 0, 0]) == 0.0

    model = AlphaNLConvNet().eval()
    batch = _conv_batch(2)
    logits, value = model(batch)
    mutated = {key: tensor.clone() for key, tensor in batch.items()}
    mutated["extra_info"] = mutated["extra_info"] + 40
    logits2, value2 = model(mutated)
    assert torch.allclose(logits, logits2)
    assert torch.allclose(value, value2)

    raw = {
        "hand": ["SA", "HK"],
        "public_cards": ["D2", "C3", "H4"],
        "legal_actions": [0, 1, 4],
        "stakes": [90, 80],
    }
    history = [[[0, 1, [1, 2, 3]], [1, 2, [0, 1, 4]]], [], [], []]
    arrays = encode_observation(raw, history, 1)
    assert arrays["card_info"].shape == (4, 13, 6)
    assert arrays["action_info"].shape == (4, 5, 25)
    assert arrays["action_info"][1, 2, 1] == 1
    assert arrays["action_info"][2, 2, 1] == 1
    assert float(arrays["action_info"][0, 0, -1]) == 1.0
    assert arrays["legal_mask"].tolist() == [1, 1, 0, 0, 1]
    observed = AlphaNLObserver().observe(raw, history, 1)
    assert observed["card_info"].shape == (1, 4, 13, 6)


def test_c1048_weight_layout_roundtrip():
    torch.manual_seed(0)
    source = AlphaNLConvNet()
    exported = export_tf_weights(source)
    # TF conv kernels are HWIO; dense kernels are (in, out).
    kernel = exported["oppo_policy/card_conv0b0/kernel"]
    assert kernel.ndim == 4 and kernel.shape[-1] == 16 and kernel.shape[-2] == 6
    dense = exported["oppo_policy/fc_1/kernel"]
    assert dense.shape[1] == 256
    restored = AlphaNLConvNet()
    missing = load_tf_weights(restored, exported)
    assert missing == []
    batch = _conv_batch(3)
    source.eval()
    restored.eval()
    with torch.no_grad():
        logits_a, value_a = source(batch)
        logits_b, value_b = restored(batch)
    assert torch.allclose(logits_a, logits_b, atol=1e-6)
    assert torch.allclose(value_a, value_b, atol=1e-6)
    # Masked softmax vs the TF fill value stays inside the comparison target
    # when the raw network matches (illegal mass is numerically zero).
    legal = batch["legal_mask"].numpy()
    tf_probs = masked_softmax_tf(logits_a.detach().numpy(), legal)
    pt_probs = torch.softmax(logits_a, dim=-1).numpy()
    assert float(np.max(np.abs(tf_probs - pt_probs))) < 1e-4


def test_bc_loss_masks_illegal_actions_and_decreases():
    torch.manual_seed(0)
    model = PokerTransformer(d_model=32, nhead=4, num_layers=1, dim_feedforward=64, dropout=0.0, max_seq_len=8)
    batch = _token_batch(16, seq=8)
    legal = batch["legal_mask"] > 0.5
    teacher = torch.tensor([0.5, 0.5, 0.0, 0.0, 0.0]).expand(16, -1).clone()
    leaked = teacher.clone()
    leaked[:, 2] = 0.5
    leaked = leaked / leaked.sum(-1, keepdim=True)
    model.eval()
    loss_clean, _ = bc_losses(model, batch, teacher_probs=teacher, kind="kl")
    loss_leaked, _metrics = bc_losses(model, batch, teacher_probs=leaked, kind="kl")
    assert torch.allclose(loss_clean, loss_leaked, atol=1e-5)
    from alphanl.transformer.model import masked_log_probs

    with torch.no_grad():
        logits, _ = model(batch)
        _, _, probs, legal_out = masked_log_probs(logits, batch["legal_mask"])
    assert float(probs[:, 2:].max()) < 1e-6
    cleaned = legalize_teacher(leaked, legal_out)
    assert float(cleaned[:, 2:].max()) == 0.0
    assert torch.allclose(cleaned.sum(-1), torch.ones(16), atol=1e-5)

    actions = torch.zeros(16, dtype=torch.long)
    loss_ce, ce_metrics = bc_losses(model, batch, actions=actions, kind="ce")
    assert ce_metrics["illegal_prob_max"] < 1e-6
    assert torch.isfinite(loss_ce)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    last = None
    peaked = torch.tensor([0.95, 0.05, 0.0, 0.0, 0.0]).expand(16, -1).contiguous()
    for _ in range(25):
        metrics = bc_update(
            model,
            optimizer,
            batch,
            teacher_probs=peaked,
            actions=actions,
            kind="kl",
            epochs=1,
            minibatch_size=16,
        )
        if first is None:
            first = metrics["loss"]
        last = metrics["loss"]
    assert last < first

    torch.manual_seed(1)
    conv = AlphaNLConvNet()
    conv_batch = _conv_batch(8)
    conv_teacher = peaked[:8].clone()
    conv_actions = torch.zeros(8, dtype=torch.long)
    opt = torch.optim.Adam(conv.parameters(), lr=1e-2)
    first_c = bc_update(
        conv, opt, conv_batch, teacher_probs=conv_teacher, actions=conv_actions, kind="kl", epochs=1, minibatch_size=8
    )["loss"]
    last_c = first_c
    for _ in range(8):
        last_c = bc_update(
            conv,
            opt,
            conv_batch,
            teacher_probs=conv_teacher,
            actions=conv_actions,
            kind="kl",
            epochs=1,
            minibatch_size=8,
        )["loss"]
    assert last_c < first_c


def test_bc_checkpoint_loads_into_ppo(tmp_path: Path):
    torch.manual_seed(2)
    cfg = ExperimentConfig()
    cfg.encoder = "transformer"
    cfg.model.d_model = 32
    cfg.model.nhead = 4
    cfg.model.num_layers = 1
    cfg.model.dim_feedforward = 64
    cfg.model.dropout = 0.0
    cfg.model.max_seq_len = 8
    cfg.device = "cpu"
    model = build_model(cfg)
    batch = _token_batch(8, seq=8)
    teacher = torch.tensor([0.8, 0.2, 0, 0, 0]).expand(8, -1).contiguous()
    actions = torch.zeros(8, dtype=torch.long)
    optimizer = build_optimizer(model, cfg)
    bc_update(
        model, optimizer, batch, teacher_probs=teacher, actions=actions, kind="ce", epochs=1, minibatch_size=8
    )
    path = save_bc_checkpoint(tmp_path / "bc.pt", model=model, optimizer=optimizer, config=cfg.to_dict())
    fresh = build_model(cfg)
    fresh_opt = build_optimizer(fresh, cfg)
    before = [p.detach().clone() for p in fresh.parameters()]
    blob = load_checkpoint(path, fresh, fresh_opt, restore_rng=False)
    assert blob["update"] == -1
    assert int(blob["update"]) + 1 == 0
    moved = any(not torch.allclose(a, b) for a, b in zip(before, fresh.parameters()))
    assert moved
    with torch.no_grad():
        logits_a, _ = model(batch)
        logits_b, _ = fresh(batch)
    assert torch.allclose(logits_a, logits_b)

    old_logits, old_values = fresh(batch)
    from alphanl.transformer.model import masked_log_probs

    _, log_probs, _, _ = masked_log_probs(old_logits.detach(), batch["legal_mask"])
    taken = torch.zeros(8, dtype=torch.long)
    old_logp = log_probs.gather(1, taken.unsqueeze(1)).squeeze(1).detach()
    returns = old_values.detach() + 0.25
    advantages = torch.ones(8)
    metrics = ppo_update(
        fresh,
        fresh_opt,
        batch,
        actions=taken,
        old_logp=old_logp,
        old_values=old_values.detach(),
        advantages=advantages,
        returns=returns,
        clip_eps=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        epochs=1,
        minibatch_size=4,
        max_grad_norm=0.5,
    )
    assert all(np.isfinite(value) for value in metrics.values())

    # Conv observations travel through the same rollout storage as tokens.
    class _Toy:
        def __init__(self):
            self.n = 0
            self.observer = AlphaNLObserver()

        def reset(self):
            self.n = 0
            return self._obs()

        def _obs(self):
            obs = self.observer.blank()
            obs["legal_mask"][0] = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])
            return obs

        def step(self, action):
            assert action in (0, 1)
            self.n += 1
            done = self.n >= 1
            return self._obs(), 0.0, done, {}

        def drain_passive_rewards(self):
            return []

    tiny = AlphaNLConvNet()
    envs = [_Toy(), _Toy()]
    obs = [env.reset() for env in envs]
    storage, _stats = collect_rollout(envs, obs, tiny, torch.device("cpu"), 2)
    assert storage["card_info"].shape == (2, 2, 4, 13, 6)
    assert storage["legal_mask"].shape == (2, 2, 5)
    flat = flatten_rollout(storage, gamma=1.0, gae_lambda=1.0)
    assert flat["card_info"].shape[0] == 4
    assert flat["actions"].shape == (4,)


def test_conv_observer_pokerkit_shapes():
    import importlib.util

    if importlib.util.find_spec("pokerkit") is None:
        pytest.skip("pokerkit is not installed")
    from alphanl.transformer.conv_obs import AlphaNLObserver, encode_table_conv
    from alphanl.transformer.tables import PokerkitTable

    table = PokerkitTable(seed=0, starting_stack=100)
    table.reset()
    arrays = encode_table_conv(table, int(table.current_player()))
    assert arrays["card_info"].shape == (4, 13, 6)
    assert arrays["action_info"].shape == (4, 5, 25)
    batch = AlphaNLObserver().observe_table(table, int(table.current_player()))
    assert batch["legal_mask"].shape == (1, 5)
    table.step(int(table.legal_action_ids()[0]))
    if not table.is_over():
        batch2 = AlphaNLObserver().observe_table(table, int(table.current_player()))
        assert batch2["card_info"].shape == (1, 4, 13, 6)


def test_one_ppo_update_pokerkit_conv(tmp_path: Path):
    import importlib.util

    if importlib.util.find_spec("pokerkit") is None:
        pytest.skip("pokerkit is not installed")
    from alphanl.transformer.config import ExperimentConfig
    from alphanl.transformer.train import run_training

    cfg = ExperimentConfig()
    cfg.encoder = "alphanl_conv"
    cfg.env.backend = "pokerkit"
    cfg.device = "cpu"
    cfg.rollout_device = "cpu"
    cfg.seed = 0
    cfg.ppo.num_envs = 1
    cfg.ppo.rollout_steps = 2
    cfg.ppo.epochs = 1
    cfg.ppo.minibatch_size = 2
    cfg.train.num_updates = 1
    cfg.train.eval_every = 0
    cfg.train.checkpoint_every = 1
    cfg.train.log_every = 1
    cfg.pool.snapshot_every = 0
    cfg.pool.max_size = 2
    cfg.train.out_dir = str(tmp_path / "out")
    cfg.train.checkpoint_dir = str(tmp_path / "ckpt")
    run_training(cfg)
    assert (tmp_path / "out" / "metrics.csv").is_file()
    assert (tmp_path / "ckpt" / "update_0000000.pt").is_file()


def test_transformer_bc_reads_dump_meta(tmp_path: Path):
    meta = {
        "hand": ["SA", "HK"],
        "public_cards": [],
        "all_chips": [1.0, 2.0],
        "stakes": [99.0, 98.0],
        "pot": 3.0,
        "stage": 0,
        "legal_actions": [0, 1, 2],
        "current_player": 0,
        "history": [[], [], [], []],
        "sb_seat": 0,
        "big_blind": 2,
        "small_blind": 1,
        "starting_stack": 100,
    }
    arrays = {
        "card_info": np.zeros((1, 4, 13, 6), np.float32),
        "action_info": np.zeros((1, 4, 5, 25), np.float32),
        "extra_info": np.zeros((1, 2), np.float32),
        "legal_mask": np.array([[1, 1, 1, 0, 0]], np.float32),
        "probs": np.array([[0.2, 0.5, 0.3, 0, 0]], np.float32),
        "actions": np.array([1], np.int64),
        "meta_json": np.array([json.dumps(meta)], dtype=object),
    }
    path = tmp_path / "tiny.npz"
    np.savez_compressed(path, **arrays)
    loaded = dict(np.load(path, allow_pickle=True))
    cfg = ExperimentConfig()
    cfg.encoder = "transformer"
    cfg.model.max_seq_len = 32
    batch = dataset_for_encoder(cfg, loaded)
    assert batch["token_ids"].shape[0] == 1
    assert batch["legal_mask"].shape == (1, 5)
    assert int(batch["legal_mask"].sum()) == 3
    direct = transformer_tensors_from_npz(loaded, 32)
    assert torch.equal(direct["actions"], batch["actions"])
