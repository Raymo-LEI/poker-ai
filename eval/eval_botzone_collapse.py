#!/usr/bin/env python3
"""M9 collapse gate on a botzone checkpoint (design §5.4.3, milestones M9/M10).

Mean legal-action entropy was logged during the 200-update run. The other
half of the gate was not: among hero decisions where a check is free
(``to_call == 0`` and fold is legal), how often the policy still folds.

This script loads the newest ``update_*.pt`` in a checkpoint directory.
The checkpoint records ``env.backend`` and the blinds; a botzone checkpoint
must load on the botzone backend (50/100, stacks of 20000). It then plays
hands against a pool-free opponent until ``--decisions`` non-forced hero
decisions have been seen.

Two passes, so the measured states are not biased by the other mode:

- ``sample`` draws from the masked policy and advances with that draw.
- ``argmax`` is a separate pass and advances with the greedy legal action.

A hero decision with only one legal id is played and then left out of every
fraction, entropy, and action share. The verdict uses the sample pass:
COLLAPSED when mean legal entropy < ``--entropy-thr`` (default 0.2) and the
sampled free-check fold fraction > ``--fold-thr`` (default 0.5).

The button alternates every hand. The hero seat swaps every two hands, so
the hero is not pinned to one blind (flipping both every hand would make
the hero the big blind forever). Over each block of four hands the hero
blind is BB, SB, SB, BB.

    PYTHONPATH=src python eval/eval_botzone_collapse.py --ckpt-dir runs/smoke_botzone_m9/ckpt --decisions 2000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from alphanl.botzone.actions import ACTION_NAMES, FOLD_ACTION, NUM_ACTIONS  # noqa: E402
from alphanl.botzone.rule_bot import BotzoneRuleBot  # noqa: E402
from alphanl.botzone.rule_eval import uniform_legal_action  # noqa: E402
from alphanl.transformer.checkpoint import latest_checkpoint, load_checkpoint  # noqa: E402
from alphanl.transformer.config import ExperimentConfig, resolve_device  # noqa: E402
from alphanl.transformer.model import entropy_from_probs, masked_log_probs  # noqa: E402
from alphanl.transformer.tokenizer import BettingHistoryTokenizer, batch_to_device  # noqa: E402
from alphanl.transformer.train import build_model, set_seed  # noqa: E402

# Same 95% normal quantile the bb/100 eval uses. Wilson, not the normal interval.
Z_95 = 1.96
BOTZONE_BLINDS = (50.0, 100.0, 20000.0)


@dataclass(frozen=True)
class HeroDecision:
    """One non-forced hero decision after the legal mask has been applied."""

    to_call: float
    fold_legal: bool
    free_check: bool
    action: int
    entropy: float
    n_legal: int


def seats_for_hand(hand_index: int) -> Tuple[int, int]:
    """``(hero_seat, dealer)`` for hand ``hand_index``.

    Dealer (button) flips every hand. Hero seat flips every two hands.
    Hero is the big blind exactly when ``hero_seat == dealer``.
    """

    if hand_index < 0:
        raise ValueError(f"hand index must be >= 0, got {hand_index}")
    return (hand_index // 2) % 2, hand_index % 2


def is_free_check(to_call: float, legal_mask: torch.Tensor) -> bool:
    """Check is free: nothing is owed, and the mask the policy sees allows fold."""

    mask = legal_mask.detach().reshape(-1)
    fold_legal = float(mask[FOLD_ACTION]) > 0.5
    return float(to_call) == 0.0 and fold_legal


def wilson_interval(successes: int, total: int, z: float = Z_95) -> Tuple[float, float]:
    """95% Wilson score interval for ``successes / total``."""

    if total <= 0:
        raise ValueError(f"Wilson interval needs at least one trial, got {total}")
    if successes < 0 or successes > total:
        raise ValueError(f"successes {successes} is outside 0..{total}")
    proportion = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (proportion + z2 / (2.0 * total)) / denom
    margin = z * math.sqrt(proportion * (1.0 - proportion) / total + z2 / (4.0 * total * total)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _as_row(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 2 or tensor.shape[0] != 1:
        raise ValueError(f"{name} must be one decision, got shape {tuple(tensor.shape)}")
    return tensor


@torch.no_grad()
def evaluate_policy_on_mask(
    logits: torch.Tensor,
    legal_mask: torch.Tensor,
    to_call: float,
    *,
    mode: str,
) -> HeroDecision:
    """Mask ``logits`` with :func:`masked_log_probs` and take one action.

    ``mode`` is ``sample`` (draw from the masked distribution) or ``argmax``.
    Entropy is the Shannon entropy of that masked distribution, in nats,
    the same quantity the PPO logger records.
    """

    if mode not in ("sample", "argmax"):
        raise ValueError(f"mode must be 'sample' or 'argmax', got {mode!r}")
    logits = _as_row(logits, "logits").float()
    legal_mask = _as_row(legal_mask, "legal_mask").to(device=logits.device, dtype=torch.float32)
    _masked, log_probs, probs, legal = masked_log_probs(logits, legal_mask)
    entropy = entropy_from_probs(probs, log_probs, legal)
    if mode == "argmax":
        action = int(torch.argmax(probs, dim=-1).item())
    else:
        action = int(torch.distributions.Categorical(probs=probs).sample().item())
    mask_row = legal_mask.reshape(-1)
    legal_ids = [index for index, flag in enumerate(mask_row.tolist()) if float(flag) > 0.5]
    if action not in legal_ids:
        raise RuntimeError(f"policy chose {action} outside legal ids {legal_ids}")
    fold_legal = FOLD_ACTION in legal_ids
    return HeroDecision(
        to_call=float(to_call),
        fold_legal=fold_legal,
        free_check=is_free_check(to_call, mask_row),
        action=action,
        entropy=float(entropy.reshape(-1)[0].item()),
        n_legal=len(legal_ids),
    )


def summarize_decisions(decisions: Sequence[HeroDecision]) -> dict:
    """Fractions, Wilson interval, entropy, and action shares.

    Every decision in ``decisions`` is counted. The caller leaves forced
    (single legal id) decisions out of this list.
    """

    if not decisions:
        raise ValueError("no hero decisions to summarize")
    counts = [0 for _ in range(NUM_ACTIONS)]
    entropies: List[float] = []
    free_entropies: List[float] = []
    free_folds = 0
    for decision in decisions:
        if not 0 <= decision.action < NUM_ACTIONS:
            raise ValueError(f"action id out of range: {decision.action}")
        counts[decision.action] += 1
        entropies.append(decision.entropy)
        if decision.free_check:
            free_entropies.append(decision.entropy)
            if decision.action == FOLD_ACTION:
                free_folds += 1
    total = len(decisions)
    free_total = len(free_entropies)
    if free_total:
        fold_fraction: Optional[float] = free_folds / free_total
        wilson: Optional[List[float]] = list(wilson_interval(free_folds, free_total))
        free_entropy: Optional[float] = sum(free_entropies) / free_total
    else:
        fold_fraction = None
        wilson = None
        free_entropy = None
    return {
        "n_decisions": total,
        "mean_legal_entropy": sum(entropies) / total,
        "mean_legal_entropy_free_check": free_entropy,
        "free_check_count": free_total,
        "free_check_fold_count": free_folds,
        "free_check_fold_fraction": fold_fraction,
        "free_check_fold_wilson95": wilson,
        "action_count": {name: counts[index] for index, name in enumerate(ACTION_NAMES)},
        "action_share": {name: counts[index] / total for index, name in enumerate(ACTION_NAMES)},
    }


def collapse_verdict(sample_summary: dict, entropy_thr: float, fold_thr: float) -> str:
    """COLLAPSED only when both sample-pass inequalities hold."""

    fold_fraction = sample_summary["free_check_fold_fraction"]
    entropy = sample_summary["mean_legal_entropy"]
    if fold_fraction is None:
        return "NOT COLLAPSED"
    if float(entropy) < float(entropy_thr) and float(fold_fraction) > float(fold_thr):
        return "COLLAPSED"
    return "NOT COLLAPSED"


def require_botzone_checkpoint(cfg: ExperimentConfig) -> None:
    """A botzone checkpoint loads on the botzone backend, with its blinds."""

    backend = str(cfg.env.backend)
    if backend != "botzone":
        raise ValueError(
            "a botzone checkpoint must load on the botzone backend, "
            f"got env.backend={backend!r}"
        )
    blinds = (
        float(cfg.env.small_blind),
        float(cfg.env.big_blind),
        float(cfg.env.starting_stack),
    )
    if blinds != BOTZONE_BLINDS:
        raise ValueError(
            "botzone checkpoint must record blinds 50/100 and starting stack 20000, "
            f"got small_blind={blinds[0]}, big_blind={blinds[1]}, starting_stack={blinds[2]}"
        )


def load_botzone_model(
    ckpt_dir: Path,
    device: torch.device,
    *,
    ckpt_path: Optional[Path] = None,
) -> tuple:
    """Newest checkpoint in ``ckpt_dir``, built with the training helper.

    Pass ``ckpt_path`` to load that file instead of the newest
    ``update_*.pt``. The collapse CLI does not pass it, so it still loads
    the newest file. Model construction is the same path as ``--dry-run``
    (``build_model`` on the checkpoint's own config). Weights then come from
    :func:`load_checkpoint`. RNG stored in the checkpoint is not restored,
    so ``--seed`` alone decides the eval.
    """

    if ckpt_path is None:
        path = latest_checkpoint(ckpt_dir)
        if path is None:
            raise FileNotFoundError(f"no update_*.pt checkpoint in {ckpt_dir}")
    else:
        path = Path(ckpt_path)
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint file not found: {path}")
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location="cpu")
    if "config" not in blob:
        raise ValueError(f"checkpoint {path} has no config")
    cfg = ExperimentConfig.from_dict(blob["config"])
    require_botzone_checkpoint(cfg)
    model = build_model(cfg)
    load_checkpoint(path, model, map_location="cpu", restore_rng=False)
    model.to(device)
    model.eval()
    tokenizer = BettingHistoryTokenizer(max_seq_len=cfg.model.max_seq_len)
    return model, tokenizer, cfg, path, int(blob.get("update", -1))


def _opponent_action(table, kind: str, bot: BotzoneRuleBot, rng: random.Random) -> int:
    """Rule bot, or a uniform draw over the five legal ids. No opponent pool.

    ``BotzoneRuleBot`` reads a ``BotzoneTable``. The env adapter keeps that
    table on ``_table`` and records history only in ``step``, so the opponent
    is chosen here and applied through ``step``.
    """

    state = table._table.state
    if state is None:
        raise RuntimeError("opponent acted with no hand")
    if kind == "random":
        return int(uniform_legal_action(state, rng))
    if kind == "rule":
        return int(bot.action_id(table._table))
    raise ValueError(f"opponent must be 'rule' or 'random', got {kind!r}")


def _dealer(table) -> int:
    """Button seat. ``sb_seat`` is public; dealer posts the big blind."""

    return (int(table.sb_seat) + 1) % 2


def play_pass(
    model: torch.nn.Module,
    tokenizer: BettingHistoryTokenizer,
    *,
    mode: str,
    n_decisions: int,
    seed: int,
    opponent: str,
    device: torch.device,
) -> dict:
    """Play until ``n_decisions`` non-forced hero decisions, then summarize.

    ``seed`` is applied at the start of the pass. The deck uses
    ``random.Random(seed)`` inside the botzone table. A random opponent
    uses ``seed + 1``. Policy samples use the torch RNG from ``set_seed``.
    """

    from alphanl.transformer.botzone_table import BotzoneEnvTable

    if mode not in ("sample", "argmax"):
        raise ValueError(f"mode must be 'sample' or 'argmax', got {mode!r}")
    if n_decisions < 1:
        raise ValueError(f"n_decisions must be >= 1, got {n_decisions}")
    set_seed(seed)
    table = BotzoneEnvTable(seed=seed, dealer=0)
    bot = BotzoneRuleBot()
    opp_rng = random.Random(seed + 1)
    decisions: List[HeroDecision] = []
    skipped = 0
    hand_index = 0
    hand_cap = max(n_decisions * 10, 100)
    while len(decisions) < n_decisions:
        if hand_index >= hand_cap:
            raise RuntimeError(
                f"stopped after {hand_index} hands with only {len(decisions)} "
                f"hero decisions (asked for {n_decisions}); skipped {skipped} forced"
            )
        table.reset()
        hero, dealer = seats_for_hand(hand_index)
        dealt = _dealer(table)
        if dealt != dealer:
            raise RuntimeError(f"button did not alternate: hand {hand_index} dealer {dealt}, expected {dealer}")
        while not table.is_over() and len(decisions) < n_decisions:
            if table.current_player() != hero:
                table.step(_opponent_action(table, opponent, bot, opp_rng))
                continue
            snapshot = table.snapshot(hero)
            batch = tokenizer.encode(snapshot)
            legal_ids = [
                index
                for index, flag in enumerate(batch["legal_mask"].reshape(-1).tolist())
                if float(flag) > 0.5
            ]
            if len(legal_ids) <= 1:
                if not legal_ids:
                    raise RuntimeError("hero saw an empty legal mask")
                skipped += 1
                table.step(int(legal_ids[0]))
                continue
            moved = batch_to_device(batch, device)
            with torch.no_grad():
                logits, _value = model(moved, mask_illegal=True)
            decision = evaluate_policy_on_mask(
                logits,
                moved["legal_mask"],
                float(snapshot.to_call),
                mode=mode,
            )
            decisions.append(decision)
            table.step(decision.action)
        hand_index += 1
    summary = summarize_decisions(decisions)
    summary["mode"] = mode
    summary["n_hands"] = hand_index
    summary["n_skipped_forced"] = skipped
    return summary


def _fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def render_report(payload: dict) -> str:
    """Human-readable table. The last line is the collapse verdict."""

    lines = [
        f"checkpoint: {payload['ckpt']}",
        (
            f"update: {payload['update']}  backend: {payload['backend']}  "
            f"blinds: {payload['small_blind']:g}/{payload['big_blind']:g}  "
            f"stack: {payload['starting_stack']:g}"
        ),
        (
            f"opponent: {payload['opponent']}  seed: {payload['seed']}  "
            f"device: {payload['device']}  decisions: {payload['decisions']}"
        ),
        payload["seat_schedule"],
        f"--entropy-thr {payload['entropy_thr']:g} --fold-thr {payload['fold_thr']:g}",
        "",
        (
            f"{'mode':<8} {'decisions':>9} {'skipped':>7} {'hands':>5} "
            f"{'free':>5} {'folds':>5} {'fold_frac':>9} {'wilson_lo':>9} "
            f"{'wilson_hi':>9} {'entropy':>8} {'ent_free':>8}"
        ),
    ]
    for mode in ("sample", "argmax"):
        block = payload[mode]
        wilson = block["free_check_fold_wilson95"]
        if wilson is None:
            low, high = "n/a", "n/a"
        else:
            low, high = f"{wilson[0]:.4f}", f"{wilson[1]:.4f}"
        lines.append(
            f"{block['mode']:<8} {block['n_decisions']:9d} {block['n_skipped_forced']:7d} "
            f"{block['n_hands']:5d} {block['free_check_count']:5d} "
            f"{block['free_check_fold_count']:5d} {_fmt(block['free_check_fold_fraction']):>9} "
            f"{low:>9} {high:>9} {block['mean_legal_entropy']:8.4f} "
            f"{_fmt(block['mean_legal_entropy_free_check']):>8}"
        )
    lines.append("skipped = forced hero decisions (only one legal action); excluded from every stat")
    lines.append("")
    lines.append(f"{'action':<12} {'sample':>10} {'argmax':>10}")
    for name in ACTION_NAMES:
        lines.append(
            f"{name:<12} {payload['sample']['action_share'][name]:10.4f} "
            f"{payload['argmax']['action_share'][name]:10.4f}"
        )
    lines.append("")
    lines.append(payload["verdict"])
    return "\n".join(lines)


def evaluate(args: argparse.Namespace) -> dict:
    if args.decisions < 1:
        raise ValueError(f"--decisions must be at least 1, got {args.decisions}")
    device_name = resolve_device(args.device)
    device = torch.device(device_name)
    model, tokenizer, cfg, path, update = load_botzone_model(Path(args.ckpt_dir), device)
    passes = {}
    for mode in ("sample", "argmax"):
        passes[mode] = play_pass(
            model,
            tokenizer,
            mode=mode,
            n_decisions=args.decisions,
            seed=args.seed,
            opponent=args.opponent,
            device=device,
        )
    verdict = collapse_verdict(passes["sample"], args.entropy_thr, args.fold_thr)
    return {
        "ckpt": str(path),
        "update": update,
        "backend": cfg.env.backend,
        "small_blind": float(cfg.env.small_blind),
        "big_blind": float(cfg.env.big_blind),
        "starting_stack": float(cfg.env.starting_stack),
        "opponent": args.opponent,
        "seed": int(args.seed),
        "decisions": int(args.decisions),
        "device": device_name,
        "entropy_thr": float(args.entropy_thr),
        "fold_thr": float(args.fold_thr),
        "seat_schedule": (
            "button alternates each hand; hero seat swaps every two hands; "
            "hero blind over four hands is BB, SB, SB, BB"
        ),
        "sample": passes["sample"],
        "argmax": passes["argmax"],
        "verdict": verdict,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=Path("runs/smoke_botzone_m9/ckpt"),
        help="Directory of update_*.pt checkpoints. The newest one is loaded.",
    )
    parser.add_argument(
        "--decisions",
        type=int,
        default=2000,
        help="Non-forced hero decisions to collect in each pass.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deck, opponent, and policy-sample seed.")
    parser.add_argument(
        "--opponent",
        choices=("rule", "random"),
        default="rule",
        help="rule = M7 botzone rule bot (default). random = uniform over legal action ids.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/collapse_eval.json"),
        help="JSON report path.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
        help="Policy device. cpu is enough for a few thousand decisions.",
    )
    parser.add_argument("--entropy-thr", type=float, default=0.2, help="COLLAPSED if sample entropy is below this.")
    parser.add_argument(
        "--fold-thr",
        type=float,
        default=0.5,
        help="COLLAPSED if the sampled free-check fold fraction is above this.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> dict:
    args = build_parser().parse_args(argv)
    payload = evaluate(args)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(render_report(payload))
    print(f"wrote {args.out}")
    return payload


if __name__ == "__main__":
    main()
