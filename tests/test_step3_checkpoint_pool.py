"""Checkpoint round-trip, opponent pool, and config presets."""

from pathlib import Path

import torch

from alphanl.transformer.checkpoint import load_checkpoint, save_checkpoint
from alphanl.transformer.config import ExperimentConfig
from alphanl.transformer.model import PokerTransformer, count_parameters
from alphanl.transformer.pool import OpponentPool
from alphanl.transformer.tokenizer import NUM_NUMERIC, NUM_TYPES, VOCAB_SIZE
from alphanl.transformer.train import describe_run


def _small() -> PokerTransformer:
    return PokerTransformer(
        vocab_size=VOCAB_SIZE,
        num_types=NUM_TYPES,
        num_numeric=NUM_NUMERIC,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        max_seq_len=8,
    )


def test_save_load_restores_logits_and_pool(tmp_path: Path):
    torch.manual_seed(0)
    model = _small()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    pool = OpponentPool(max_size=2, selfplay_prob=0.0, random_prob=0.0, sampling="uniform", seed=1)
    policy_id = pool.add(model.state_dict())
    pool.record(policy_id, 1.0)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        update=7,
        config={"name": "test"},
        pool=pool,
    )
    restored = _small()
    restored_opt = torch.optim.Adam(restored.parameters(), lr=1e-3)
    restored_pool = OpponentPool(max_size=2, selfplay_prob=0.0, random_prob=0.0, seed=1)
    blob = load_checkpoint(path, restored, restored_opt, restored_pool)
    assert blob["update"] == 7
    assert len(restored_pool) == 1
    assert restored_pool.win_rate(policy_id) == 1.0
    batch = {
        "token_ids": torch.randint(1, 20, (4, 8)),
        "type_ids": torch.ones(4, 8, dtype=torch.long),
        "numeric": torch.zeros(4, 8, NUM_NUMERIC),
        "pad_mask": torch.zeros(4, 8, dtype=torch.bool),
        "legal_mask": torch.ones(4, 5),
    }
    model.eval()
    restored.eval()
    with torch.no_grad():
        logits_a, value_a = model(batch)
        logits_b, value_b = restored(batch)
    assert torch.allclose(logits_a, logits_b)
    assert torch.allclose(value_a, value_b)


def test_pool_pfsp_prefers_losses_and_evicts_oldest():
    pool = OpponentPool(max_size=2, selfplay_prob=0.0, random_prob=0.0, sampling="pfsp", seed=0)
    weak = pool.add({"w": torch.zeros(1)})
    strong = pool.add({"w": torch.ones(1)})
    for _ in range(20):
        pool.record(weak, -1.0)
        pool.record(strong, 1.0)
    counts = {weak: 0, strong: 0}
    for _ in range(200):
        counts[pool.sample()["policy_id"]] += 1
    assert counts[weak] > counts[strong] * 10
    newest = pool.add({"w": torch.full((1,), 2.0)})
    assert len(pool) == 2
    pool.record(weak, 1.0)
    assert newest in pool._order
    assert weak not in pool._entries


def test_presets_fit_the_two_hardware_targets():
    root = Path(__file__).resolve().parents[1]
    laptop = ExperimentConfig.from_json(root / "configs" / "transformer" / "laptop_8gb.json")
    colab = ExperimentConfig.from_json(root / "configs" / "transformer" / "colab_l4.json")
    assert laptop.ppo.entropy_coef > 0
    assert colab.model.d_model > laptop.model.d_model
    assert colab.model.num_layers > laptop.model.num_layers
    assert colab.ppo.minibatch_size > laptop.ppo.minibatch_size
    assert colab.ppo.num_envs * colab.ppo.rollout_steps > laptop.ppo.num_envs * laptop.ppo.rollout_steps
    assert laptop.rollout_device == "cpu"
    assert "entropy_coef=0.01" in describe_run(laptop)
    laptop.apply_overrides(entropy_coef=0.2, num_updates=3)
    assert laptop.ppo.entropy_coef == 0.2
    assert laptop.train.num_updates == 3
    assert count_parameters(PokerTransformer()) > 100_000
