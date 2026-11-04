"""50-hand match table with a W>B stop, 100 pairs by 2 seats."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from alphanl.cfr.evaluation import RuleActor, mean_interval
from alphanl.stage2.env import BotzoneMatchEnv
from alphanl.stage2.opponents import NeuralMatchActor
from alphanl.transformer.tables import standard_deck
from ladder_actors import SlayerActor
from alphanl.stage2.train import load_checkpoint

def load_model(path, device):
    return load_checkpoint(path, device=device)

OUT = Path("/root/autodl-tmp/sprint24/tournament")
SEED = 20261030
PAIRS = 100
SCRIPTS = ("Bot17_Probe3Bet", "TAG_Pro", "Trapper_CR", "equity", "NearShove_19949")


def decks():
    rng = np.random.default_rng(SEED)
    return [[rng.permutation(standard_deck()).tolist() for _ in range(50)] for _ in range(PAIRS)]


def remaining_posts(hands_played, first_seat):
    total = 0
    for hand in range(hands_played, 50):
        seat = first_seat ^ (hand % 2)
        total += 100 if seat == 0 else 50
    return total


def locked(env, first_seat):
    lead = env.ctx.hero_net_bb * 100.0
    snap = env._hero_decision_snapshot()
    invested = max(0., snap.starting_stack - snap.me_stack)
    return lead > remaining_posts(env.ctx.hands_played, first_seat) + invested


def summarize(pairs):
    from alphanl.stage2.evaluation import summarize_match_pairs
    stats = summarize_match_pairs(pairs)
    stats["win_rate"] = stats["match_win"]
    return stats


class TorchHero:
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device

    def act(self, env):
        batch = {key: value.to(self.device) for key, value in env.observe().items()}
        action, *_ = self.model.act(batch, deterministic=True)
        return int(action)


class WizardHero:
    def __init__(self, npz, template):
        import json as _json
        import eval_wizard
        agent = eval_wizard.load_agent(npz)
        agent.force_argmax = False
        eval_wizard.attach_config(agent, template)
        self.agent = agent

    def act(self, env):
        return int(self.agent.act(env._hero_decision_snapshot(), env.ctx))


def play(hero, opponent, pair_decks, first, seed):
    env = BotzoneMatchEnv(
        hero.model.config if isinstance(hero, TorchHero) else hero.agent._config_obj,
        opponent, decks=pair_decks, first_hero_seat=first, seed=seed,
        context_fields=hero.model.context_fields if isinstance(hero, TorchHero) else hero.agent._context_fields,
        allin_mask=(hero.model.settings.get("botzone_native_allin_mask") if isinstance(hero, TorchHero) else hero.agent._mask),
    )
    env.reset()
    while not env.done:
        env.step(hero.act(env))
    return env.ctx.hero_net_bb


def script_opponent(name, seed):
    if name in ("Bot17_Probe3Bet", "NearShove_19949", "CheckShove_Trapper"):
        return SlayerActor(name, seed=seed, equity_samples=8)
    return RuleActor(name, seed=seed, equity_samples=8)


def versus_scripts(name, hero, all_decks):
    scores = {}
    for opp in SCRIPTS:
        rows = []
        for pair, pair_decks in enumerate(all_decks):
            legs = []
            for seat in (1, 0):
                seed = SEED + pair * 2 + seat
                torch.manual_seed(seed)
                legs.append(play(hero, script_opponent(opp, seed), pair_decks, seat, seed))
            rows.append(legs)
            if pair % 20 == 19:
                print("TOUR", name, opp, pair + 1, flush=True)
        scores[opp] = summarize(rows)
        print("TOUR_SCORE", name, opp, json.dumps(scores[opp]), flush=True)
    return scores


def versus_hero(left_name, left, right, all_decks):
    rows = []
    for pair, pair_decks in enumerate(all_decks):
        legs = []
        for seat in (1, 0):
            seed = SEED + 50000 + pair * 2 + seat
            villain = NeuralMatchActor(right.model, kind="residual", snapshot_id="villain", seed=seed)
            legs.append(play(left, villain, pair_decks, seat, seed))
        rows.append(legs)
    return summarize(rows)


def load_torch(path):
    model, _ = load_model(path, "cuda")
    model.eval()
    return TorchHero(model)


def main():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    catalog = {
        "C200": Path("/root/autodl-tmp/stage2_champion_v2_cand200_20261029/checkpoints/champion_stage2_v2_cand200.pt"),
        "V8_R8": Path("/root/autodl-tmp/stage2_botzone_native_v2_20261030/update_0000250.pt"),
        "V9_R9_GEN2": Path("/root/autodl-tmp/stage2_u250_gen2_20261103/u250_gen2_best.pt"),
        "V9.5_SURGICAL": Path("/root/autodl-tmp/stage2_u250_gen2_20261103/u250_gen2_surgical.pt"),
        "V11_REBEL": Path("/root/autodl-tmp/sprint24/rebel/u250_rebel_blueprint.pt"),
        "V12_LADDER_SLAYER": Path("/root/autodl-tmp/sprint24/ladder/u250_ladder_slayer.pt"),
    }
    heroes = {}
    for name, path in catalog.items():
        if path.is_file():
            heroes[name] = load_torch(path)
            print("TOUR_LOADED", name, flush=True)
        else:
            print("TOUR_MISSING", name, path, flush=True)
    v10 = Path("/root/autodl-tmp/sprint24/deliver/V10_WIZARD/V10_WIZARD_MODEL.npz")
    if v10.is_file() and "V9.5_SURGICAL" in heroes:
        import eval_wizard
        tree = Path("/root/autodl-tmp/sprint24/wizard/src")
        if str(tree) not in sys.path:
            sys.path.insert(0, str(tree))
        heroes["V10_WIZARD"] = WizardHero(v10, heroes["V9.5_SURGICAL"].model)
        print("TOUR_LOADED V10_WIZARD", flush=True)
    all_decks = decks()
    table = {}
    for name, hero in heroes.items():
        table[name] = versus_scripts(name, hero, all_decks)
        (OUT / "scripts.json").write_text(json.dumps(table, indent=2) + "\n")
    matrix = {}
    names = [name for name, hero in heroes.items() if isinstance(hero, TorchHero)]
    for i, left in enumerate(names):
        matrix[left] = {}
        for right in names[i + 1:]:
            matrix[left][right] = versus_hero(left, heroes[left], heroes[right], all_decks)
            print("H2H", left, right, json.dumps(matrix[left][right]), flush=True)
            (OUT / "h2h.json").write_text(json.dumps(matrix, indent=2) + "\n")
    print("TOUR_DONE", flush=True)


if __name__ == "__main__":
    main()
