"""Retained archives survive generation GC and restore exact training state."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from alphanl.cfr import SDCFRTrainer
from alphanl.cfr.retention import archive_checkpoint, archive_due, restore_archive
from alphanl.cfr.study import sha256
from test_cfr_trainer import TinyGame, config


def test_schedule_starts_at_100_and_retains_every_tenth_round(tmp_path):
    cfg = config(tmp_path)
    cfg.checkpoint_archive_from = 100
    cfg.checkpoint_archive_every = 10
    assert [i for i in range(1, 201) if archive_due(cfg, i)] == list(range(100, 201, 10))


@pytest.mark.parametrize("disk", [False, True])
def test_full_archive_survives_gc_and_restores_exact_next_iteration(tmp_path, disk):
    cfg = config(tmp_path / "run", checkpoint_archive_from=1, checkpoint_archive_every=1,
                 checkpoint_archive_min_free_bytes=0,
                 buffer_directory=str(tmp_path / "live") if disk else None)
    with SDCFRTrainer(cfg, root_factory=lambda seed: TinyGame()) as trainer:
        trainer.run_iteration()
        trainer.save_checkpoint(tmp_path / "run/train.pt")
        expected = trainer.run_iteration()
        weights = {p: {k: v.clone() for k, v in m.state_dict().items()} for p, m in trainer.models.items()}
        trainer.save_checkpoint(tmp_path / "run/train.pt")
    archives = sorted((tmp_path / "run/checkpoint_archives").glob("*.tar.gz"))
    assert len(archives) == 2
    assert len(list((tmp_path / "run/checkpoint_archives/objects").glob("*.pt"))) == 2
    assert len(list((tmp_path / "run/checkpoint_data").iterdir())) == 1
    checkpoint = restore_archive(archives[0], tmp_path / "materialized", min_free_bytes=0)
    restored_cfg = replace(cfg, out_dir=str(tmp_path / "replay"), checkpoint_archive_from=None,
                           buffer_directory=str(tmp_path / "new_live") if disk else None)
    with SDCFRTrainer(restored_cfg, root_factory=lambda seed: TinyGame()) as restored:
        restored.load_checkpoint(checkpoint)
        assert restored.iteration == 1
        assert restored.run_iteration() == expected
        for player, state in weights.items():
            for key, value in state.items():
                assert torch.equal(value, restored.models[player].state_dict()[key])
    with pytest.raises(FileExistsError):
        restore_archive(archives[0], tmp_path / "materialized", min_free_bytes=0)


@pytest.mark.parametrize("fail_during_stream", [False, True])
def test_low_space_preserves_restart_and_removes_incomplete_archive(tmp_path, monkeypatch, fail_during_stream):
    import alphanl.cfr.retention as retention
    checkpoint = tmp_path / "run/train.pt"
    cfg = config(checkpoint.parent)
    with SDCFRTrainer(cfg, root_factory=lambda seed: TinyGame()) as trainer:
        trainer.run_iteration()
        trainer.save_checkpoint(checkpoint)
    before = sha256(checkpoint)
    payload = torch.load(checkpoint, weights_only=False)
    spaces = iter([10**9, 10**9, 0] if fail_during_stream else [0])
    monkeypatch.setattr(retention.shutil, "disk_usage", lambda p: SimpleNamespace(free=next(spaces, 0)))
    with pytest.raises(OSError, match="reserve"):
        archive_checkpoint(checkpoint, payload, iteration=1, remaining=0, min_free_bytes=0)
    assert sha256(checkpoint) == before
    assert not list((checkpoint.parent / "checkpoint_archives").glob("*.tmp"))
    assert not list((checkpoint.parent / "checkpoint_archives").glob("*.tar.gz"))


def test_projection_stops_without_discarding_completed_archive(tmp_path, monkeypatch):
    import alphanl.cfr.retention as retention
    checkpoint = tmp_path / "run/train.pt"
    with SDCFRTrainer(config(checkpoint.parent), root_factory=lambda seed: TinyGame()) as trainer:
        trainer.run_iteration()
        trainer.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, weights_only=False)
    spaces = iter([10**9, 10**9, 10**9, 0])
    monkeypatch.setattr(retention.shutil, "disk_usage", lambda p: SimpleNamespace(free=next(spaces, 0)))
    with pytest.raises(OSError, match="projected retention"):
        archive_checkpoint(checkpoint, payload, iteration=1, remaining=100, min_free_bytes=0)
    archive = checkpoint.parent / "checkpoint_archives/iteration_0000001.tar.gz"
    assert archive.is_file()
    report = json.loads(archive.with_suffix(archive.suffix + ".json").read_text())
    assert report["archive_sha256"] == sha256(archive)
    assert report["full_restart_state"]


def test_restore_rejects_changed_archive_and_path_escape(tmp_path):
    checkpoint = tmp_path / "run/train.pt"
    cfg = config(checkpoint.parent, checkpoint_archive_from=1, checkpoint_archive_min_free_bytes=0)
    with SDCFRTrainer(cfg, root_factory=lambda seed: TinyGame()) as trainer:
        trainer.run_iteration()
        trainer.save_checkpoint(checkpoint)
    archive = next((checkpoint.parent / "checkpoint_archives").glob("*.tar.gz"))
    metadata = archive.with_suffix(archive.suffix + ".json")
    report = json.loads(metadata.read_text())
    report["files"]["../escape"] = 1
    metadata.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="unsafe"):
        restore_archive(archive, tmp_path / "restore", min_free_bytes=0)
    assert not (tmp_path / "escape").exists()
    archive.write_bytes(archive.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="fingerprint"):
        restore_archive(archive, tmp_path / "restore", min_free_bytes=0)
