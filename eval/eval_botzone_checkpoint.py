#!/usr/bin/env python3
"""M11 checkpoint eval on the botzone backend (design §5.4.4, milestones M11).

Loads a trained botzone checkpoint and plays the hero policy against both
weak opponents from M7: the botzone rule bot, and a uniform draw over legal
action ids. Each opponent is a separate pass of ``--hands`` hands (default
5000). The button and the hero seat rotate the same way as
``eval_botzone_collapse.py`` (button every hand, hero seat every two hands),
so the hero is not pinned to one blind.

``--mode`` is ``argmax`` (default), ``sample``, or ``both``. ``both`` plays
two passes. Design §5.4.4 scores the sampled mixed strategy, not the greedy
one; the PASS/FAIL lines name the mode they were computed from. A FAIL is
printed and does not change the exit code.

bb/100 and the 95% interval are :func:`alphanl.botzone.rule_eval.bb_per_100_interval`:
mean of (hero chip delta / big blind) per hand, times 100, interval
mean ± 1.96 × sample std / √n with the ``n - 1`` std. The 50-hand match
score is :func:`alphanl.botzone.rule_eval.match_scores`: dealer alternating,
cumulative chips / 100, and a tail shorter than 50 hands is dropped. A match
win is a positive score. A zero score is a tie, not a win.

The M7 yardstick R = 540.95 bb/100 (rule vs random, 10000 hands, seed 42,
hero seat 0) is printed for comparison only. This script does not recompute it.
No AMP. ``--device auto`` uses CUDA when it is present, otherwise CPU.

    PYTHONPATH=src python eval/eval_botzone_checkpoint.py --ckpt-dir runs/ckpt_botzone --hands 5000 --seed 42 --mode both
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from alphanl.botzone.legal import BIG_BLIND  # noqa: E402
from alphanl.botzone.rule_eval import (  # noqa: E402
    MATCH_LENGTH,
    Z_95,
    bb_per_100_interval,
    match_scores,
)
from alphanl.transformer.config import resolve_device  # noqa: E402
from alphanl.transformer.tokenizer import batch_to_device  # noqa: E402
from alphanl.transformer.train import set_seed  # noqa: E402

# M7 rule-vs-random yardstick. Comparison only; not recomputed here.
M7_REFERENCE_R = 540.95
M7_REFERENCE_R_HANDS = 10000
M7_REFERENCE_R_SEED = 42
M7_REFERENCE_R_HERO_SEAT = 0
M7_REFERENCE_R_MATCHUP = "rule_vs_random"
M7_REFERENCE_R_NOTE = (
    "M7 rule vs random, 10000 hands, seed 42, hero seat 0. "
    "Printed for comparison only; this script does not recompute R."
)
MIN_MATCHES = 100
OPPONENT_NAMES = {
    "rule": "botzone_rule_bot",
    "random": "uniform_legal_action_ids",
}
SEAT_SCHEDULE = (
    "button alternates each hand; hero seat swaps every two hands; "
    "hero blind over four hands is BB, SB, SB, BB"
)


def _load_collapse():
    """The collapse script owns the botzone checkpoint loader and seat rotation."""

    path = Path(__file__).resolve().with_name("eval_botzone_collapse.py")
    name = "_m11b_eval_botzone_collapse"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_COLLAPSE = _load_collapse()
load_botzone_model = _COLLAPSE.load_botzone_model
seats_for_hand = _COLLAPSE.seats_for_hand
evaluate_policy_on_mask = _COLLAPSE.evaluate_policy_on_mask
_opponent_action = _COLLAPSE._opponent_action
_dealer = _COLLAPSE._dealer


def modes_for(mode: str) -> tuple:
    """``both`` plays sample, then argmax. One name plays only that pass."""

    if mode == "both":
        return ("sample", "argmax")
    if mode in ("sample", "argmax"):
        return (mode,)
    raise ValueError(f"mode must be 'sample', 'argmax', or 'both', got {mode!r}")


def summarize_chip_deltas(
    chip_deltas: Sequence[int],
    *,
    match_length: int = MATCH_LENGTH,
    big_blind: int = BIG_BLIND,
) -> dict:
    """bb/100, the M7 normal interval, and 50-hand match scores.

    The interval is exactly :func:`bb_per_100_interval`. Match blocks are
    exactly :func:`match_scores`. A win is a block whose cumulative chips
    are strictly positive.
    """

    mean, std, bb100, low, high = bb_per_100_interval(chip_deltas, big_blind)
    scores = match_scores(chip_deltas, match_length, per=big_blind)
    n_matches = len(scores)
    wins = sum(1 for score in scores if score > 0.0)
    ties = sum(1 for score in scores if score == 0.0)
    losses = sum(1 for score in scores if score < 0.0)
    if wins + ties + losses != n_matches:
        raise RuntimeError("match outcomes do not add up to the number of matches")
    return {
        "n_hands": len(chip_deltas),
        "big_blind": int(big_blind),
        "mean_bb_per_hand": mean,
        "std_bb_per_hand": std,
        "std_divisor": "n-1",
        "bb_per_100": bb100,
        "ci95_low": low,
        "ci95_high": high,
        "ci95": [low, high],
        "match_length": int(match_length),
        "n_matches": n_matches,
        "mean_match_score": (sum(scores) / n_matches) if n_matches else None,
        "match_scores": scores,
        "n_match_wins": wins,
        "n_match_ties": ties,
        "n_match_losses": losses,
        "match_win_fraction": (wins / n_matches) if n_matches else None,
        "match_score_unit": "cumulative_chips_per_100",
        "tail_hands_dropped": len(chip_deltas) % int(match_length),
    }


def pass_line(opponents: dict, mode: str, reference_r: float = M7_REFERENCE_R) -> dict:
    """§5.4.4 / M11 checks for one action-selection pass.

    Versus the rule bot the 95% CI lower bound must be strictly above 0.
    Versus uniform random the bb/100 point estimate must be at least R.
    At least 100 complete matches is reported per opponent, separately
    from those two strength checks.
    """

    rule = opponents["rule"][mode]
    random_block = opponents["random"][mode]
    rule_ok = float(rule["ci95_low"]) > 0.0
    random_ok = float(random_block["bb_per_100"]) >= float(reference_r)
    matches = {}
    for name in ("rule", "random"):
        n_matches = int(opponents[name][mode]["n_matches"])
        ok = n_matches >= MIN_MATCHES
        matches[name] = {
            "n_matches": n_matches,
            "required": MIN_MATCHES,
            "pass": ok,
            "result": "PASS" if ok else "FAIL",
        }
    return {
        "vs_rule_ci_low_gt_0": {
            "ci95_low": float(rule["ci95_low"]),
            "threshold": 0.0,
            "pass": rule_ok,
            "result": "PASS" if rule_ok else "FAIL",
        },
        "vs_random_bb_per_100_ge_R": {
            "bb_per_100": float(random_block["bb_per_100"]),
            "R": float(reference_r),
            "pass": random_ok,
            "result": "PASS" if random_ok else "FAIL",
        },
        "n_matches_ge_100": matches,
    }


def play_hands(
    model: torch.nn.Module,
    tokenizer,
    *,
    mode: str,
    opponent: str,
    n_hands: int,
    seed: int,
    device: torch.device,
) -> List[int]:
    """Hero chip deltas for ``n_hands`` complete hands.

    ``seed`` is applied at the start of the pass. The deck uses
    ``BotzoneEnvTable(seed=seed)``. A random opponent uses ``seed + 1``.
    Policy samples use the torch RNG from :func:`set_seed`. Forced hero
    actions (one legal id) are played and their chip result still counts.
    """

    from alphanl.botzone.rule_bot import BotzoneRuleBot
    from alphanl.transformer.botzone_table import BotzoneEnvTable

    if mode not in ("sample", "argmax"):
        raise ValueError(f"mode must be 'sample' or 'argmax', got {mode!r}")
    if opponent not in OPPONENT_NAMES:
        raise ValueError(f"opponent must be 'rule' or 'random', got {opponent!r}")
    if n_hands < 2:
        raise ValueError(f"n_hands must be at least 2, got {n_hands}")
    set_seed(seed)
    table = BotzoneEnvTable(seed=seed, dealer=0)
    bot = BotzoneRuleBot()
    opp_rng = random.Random(seed + 1)
    model.eval()
    chips: List[int] = []
    for hand_index in range(n_hands):
        if hand_index and hand_index % 1000 == 0:
            print(
                f"playing {opponent}/{mode} {hand_index}/{n_hands}",
                file=sys.stderr,
                flush=True,
            )
        table.reset()
        hero, dealer = seats_for_hand(hand_index)
        dealt = _dealer(table)
        if dealt != dealer:
            raise RuntimeError(
                f"button did not alternate: hand {hand_index} dealer {dealt}, expected {dealer}"
            )
        while not table.is_over():
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
                table.step(int(legal_ids[0]))
                continue
            moved = batch_to_device(batch, device)
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
                logits, _value = model(moved, mask_illegal=True)
            decision = evaluate_policy_on_mask(
                logits,
                moved["legal_mask"],
                float(snapshot.to_call),
                mode=mode,
            )
            table.step(decision.action)
        payoffs = table.payoffs()
        if abs(float(payoffs[0]) + float(payoffs[1])) > 1e-6:
            raise RuntimeError(f"hand {hand_index} payoffs do not sum to 0: {payoffs}")
        hero_delta = float(payoffs[hero])
        if abs(hero_delta - round(hero_delta)) > 1e-6:
            raise RuntimeError(f"hand {hand_index} hero delta is not whole chips: {hero_delta}")
        chips.append(int(round(hero_delta)))
    return chips


def _block(chip_deltas: Sequence[int], opponent: str, mode: str) -> dict:
    block = summarize_chip_deltas(chip_deltas)
    block["opponent"] = opponent
    block["mode"] = mode
    block["chip_deltas"] = [int(delta) for delta in chip_deltas]
    return block


def _fmt_match(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _fmt_frac(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def render_report(payload: dict) -> str:
    """Stdout table. PASS/FAIL lines are informational; the process still exits 0."""

    lines = [
        f"checkpoint: {payload['ckpt']}",
        (
            f"update: {payload['update']}  backend: {payload['backend']}  "
            f"blinds: {payload['small_blind']:g}/{payload['big_blind']:g}  "
            f"stack: {payload['starting_stack']:g}"
        ),
        (
            f"seed: {payload['seed']}  hands: {payload['hands']}  "
            f"device: {payload['device']}  mode: {payload['mode']}  amp: off"
        ),
        payload["seat_schedule"],
        f"M7 reference R = {payload['reference_R']:.2f} bb/100 ({payload['reference_R_note']})",
        "",
        (
            f"{'opponent':<8} {'mode':<8} {'hands':>6} {'bb/100':>10} "
            f"{'ci95_low':>10} {'ci95_high':>10} {'matches':>8} "
            f"{'match_mean':>10} {'win_frac':>8} {'wins':>12}"
        ),
    ]
    for opponent in ("rule", "random"):
        for mode in payload["modes"]:
            block = payload["opponents"][opponent][mode]
            wins = f"{block['n_match_wins']}/{block['n_matches']}"
            lines.append(
                f"{opponent:<8} {mode:<8} {block['n_hands']:6d} {block['bb_per_100']:10.2f} "
                f"{block['ci95_low']:10.2f} {block['ci95_high']:10.2f} {block['n_matches']:8d} "
                f"{_fmt_match(block['mean_match_score']):>10} {_fmt_frac(block['match_win_fraction']):>8} "
                f"{wins:>12}"
            )
    lines.append("")
    lines.append(
        "match score = 50 hands, dealer alternating, cumulative chips/100; "
        "a tail shorter than 50 hands is dropped"
    )
    lines.append("match win = score > 0 (a zero score is a tie, not a win)")
    lines.append("bb/100 CI = mean ± 1.96 * sample std / sqrt(n), std divisor n-1, then × 100")
    lines.append("PASS/FAIL does not change the exit code")
    if "sample" in payload["modes"]:
        lines.append("M11 §5.4.4 pass line is the sample pass (mixed strategy, not greedy)")
    else:
        lines.append(
            "M11 §5.4.4 pass line is the sample pass; this run did not play sample. "
            "Lines below are the modes that ran. Use --mode sample or --mode both for the milestone line."
        )
    for mode in payload["modes"]:
        check = payload["pass_lines"][mode]
        rule = check["vs_rule_ci_low_gt_0"]
        random_check = check["vs_random_bb_per_100_ge_R"]
        lines.append(
            f"vs rule bot [{mode}] 95% CI lower bound > 0: {rule['ci95_low']:.2f}  {rule['result']}"
        )
        lines.append(
            f"vs random [{mode}] bb/100 >= R ({payload['reference_R']:.2f}): "
            f"{random_check['bb_per_100']:.2f}  {random_check['result']}"
        )
        for opponent in ("rule", "random"):
            item = check["n_matches_ge_100"][opponent]
            lines.append(
                f"matches >= 100 [{opponent} {mode}]: {item['n_matches']}  {item['result']}"
            )
    return "\n".join(lines)


def evaluate(args: argparse.Namespace) -> dict:
    if args.hands < 2:
        raise ValueError(f"--hands must be at least 2, got {args.hands}")
    modes = modes_for(args.mode)
    device_name = resolve_device(args.device)
    device = torch.device(device_name)
    model, tokenizer, cfg, path, update = load_botzone_model(
        Path(args.ckpt_dir),
        device,
        ckpt_path=args.ckpt_file,
    )
    opponents = {}
    for opponent in ("rule", "random"):
        opponents[opponent] = {"name": OPPONENT_NAMES[opponent]}
        for mode in modes:
            chips = play_hands(
                model,
                tokenizer,
                mode=mode,
                opponent=opponent,
                n_hands=args.hands,
                seed=args.seed,
                device=device,
            )
            opponents[opponent][mode] = _block(chips, opponent, mode)
    return {
        "ckpt": str(path),
        "ckpt_dir": str(args.ckpt_dir),
        "ckpt_file": None if args.ckpt_file is None else str(args.ckpt_file),
        "update": update,
        "backend": cfg.env.backend,
        "small_blind": float(cfg.env.small_blind),
        "big_blind": float(cfg.env.big_blind),
        "starting_stack": float(cfg.env.starting_stack),
        "seed": int(args.seed),
        "hands": int(args.hands),
        "device": device_name,
        "mode": args.mode,
        "modes": list(modes),
        "amp": False,
        "reference_R": M7_REFERENCE_R,
        "reference_R_hands": M7_REFERENCE_R_HANDS,
        "reference_R_seed": M7_REFERENCE_R_SEED,
        "reference_R_hero_seat": M7_REFERENCE_R_HERO_SEAT,
        "reference_R_matchup": M7_REFERENCE_R_MATCHUP,
        "reference_R_note": M7_REFERENCE_R_NOTE,
        "match_length": MATCH_LENGTH,
        "min_matches": MIN_MATCHES,
        "ci_z": Z_95,
        "seat_schedule": SEAT_SCHEDULE,
        "opponents": opponents,
        "pass_lines": {mode: pass_line(opponents, mode) for mode in modes},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=Path("runs/ckpt_botzone"),
        help="Directory of update_*.pt checkpoints. The newest one is loaded "
        "unless --ckpt-file is set.",
    )
    parser.add_argument(
        "--ckpt-file",
        type=Path,
        default=None,
        help="Load this update file instead of the newest checkpoint in --ckpt-dir.",
    )
    parser.add_argument(
        "--hands",
        type=int,
        default=5000,
        help="Hands per opponent per mode. Default 5000. 20000 and above are allowed. "
        "At least 2 are required for the sample std.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deck, opponent, and policy-sample seed.")
    parser.add_argument(
        "--mode",
        choices=("sample", "argmax", "both"),
        default="argmax",
        help="Policy action selection. both reports sample and argmax as separate passes. "
        "§5.4.4 scores the sample pass.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/eval_checkpoint.json"),
        help="JSON report path.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Policy device. auto picks cuda when it is present. No AMP.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> dict:
    args = build_parser().parse_args(argv)
    payload = evaluate(args)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(render_report(payload))
    print(f"wrote {args.out}")
    return payload


if __name__ == "__main__":
    main()
