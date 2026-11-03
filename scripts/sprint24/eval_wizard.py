"""5,000-hand exam of the blueprint plus online search, then pack if it passes."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/root/autodl-tmp/sprint24")
sys.path.insert(0, "/root/autodl-tmp/stage2_botzone_native_code_20261030/src")
sys.path.insert(0, "/root/autodl-tmp/stage2_botzone_native_code_20261030/scripts")

import pack_wizard
import wizard
from alphanl.cfr.evaluation import RuleActor
from alphanl.stage2.env import BotzoneMatchEnv
from alphanl.stage2.evaluation import summarize_match_pairs
from alphanl.stage2.export import export_botzone_weights
from alphanl.transformer.tables import standard_deck
from phevaluator import evaluate_cards
from surgery import load_model

OUT = Path("/root/autodl-tmp/sprint24/wizard")
SEED = 20261030
OPPONENTS = ("equity", "TAG_Pro", "Trapper_CR")
C200 = Path("/root/autodl-tmp/stage2_champion_v2_cand200_20261029/checkpoints/champion_stage2_v2_cand200.pt")


def load_agent(path):
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(archive["meta_json"].tobytes().decode("utf-8"))
        weights = {key: archive[key] for key in archive.files if key != "meta_json"}
    from stage2_runtime.runtime import Agent
    return Agent(weights, meta["config"], meta["settings"], meta["numeric_version"], meta.get("deployment_guard"))


def blueprint_npz(choice):
    """Copy the V9 archive and replace blueprint tensors. The guard JSON stays untouched."""
    dest = OUT / "surgical_blueprint.npz"
    model, _blob = load_model(Path(choice["pt"]), "cpu")
    fresh = {}
    for key, value in model.state_dict().items():
        if key.startswith("reference.backbone."):
            key = key[len("reference.backbone."):]
        fresh[key] = np.ascontiguousarray(value.detach().cpu().numpy())
    with np.load(pack_wizard.V9_NPZ, allow_pickle=False) as archive:
        meta = np.array(archive["meta_json"])
        base = {key: np.array(archive[key]) for key in archive.files if key != "meta_json"}
    replaced = 0
    for key, value in fresh.items():
        if key not in base:
            continue
        if base[key].shape != value.shape:
            raise RuntimeError("shape mismatch for %s" % key)
        base[key] = value
        replaced += 1
    if replaced < 10:
        raise RuntimeError("surgical export replaced only %d tensors" % replaced)
    np.savez(dest, meta_json=meta, **{key: np.ascontiguousarray(value) for key, value in base.items()})
    print("SURGICAL_NPZ_DONE", dest, dest.stat().st_size, replaced, flush=True)
    return dest


def decks():
    rng = np.random.default_rng(SEED)
    return [[rng.permutation(standard_deck()).tolist() for _ in range(50)] for _ in range(50)]


def play_wizard(agent, opponent, pair_decks, first, seed, times):
    env = BotzoneMatchEnv(
        agent._config_obj, opponent, decks=pair_decks, first_hero_seat=first, seed=seed,
        context_fields=agent._context_fields, allin_mask=agent._mask,
    )
    env.reset()
    rng = np.random.default_rng(seed)
    while not env.done:
        started = time.perf_counter()
        action = agent.act(env._hero_decision_snapshot(), env.ctx, rng=rng)
        times.append(time.perf_counter() - started)
        env.step(int(action))
    return env.ctx.hero_net_bb


def play_torch(model, opponent, pair_decks, first, seed, device):
    env = BotzoneMatchEnv(
        model.config, opponent, decks=pair_decks, first_hero_seat=first, seed=seed,
        context_fields=model.context_fields, allin_mask=model.settings.get("botzone_native_allin_mask"),
    )
    env.reset()
    while not env.done:
        batch = {key: value.to(device) for key, value in env.observe().items()}
        action, *_ = model.act(batch, deterministic=True)
        env.step(int(action))
    return env.ctx.hero_net_bb


class NumpyVillain:
    def __init__(self, agent):
        from alphanl.stage2.context import MatchCtx
        self.agent = agent
        self.agent.force_argmax = True
        self.ctx = MatchCtx()
        self.name = "wizard"

    def begin_match(self):
        from alphanl.stage2.context import MatchCtx
        self.ctx = MatchCtx()

    def begin_hand(self, seat):
        self.ctx.begin_hand()

    def observe_action(self, event):
        self.ctx.observe_action(event)

    def finish_hand(self, net_bb, *, showdown):
        self.ctx.finish_hand(net_bb, showdown=showdown)

    def act(self, snapshot):
        return int(self.agent.act(snapshot, self.ctx, rng=np.random.default_rng(1)))


def attach_config(agent, model):
    agent._config_obj = model.config
    agent._context_fields = model.context_fields
    agent._mask = model.settings.get("botzone_native_allin_mask")


def exam(agent, all_decks, times):
    scores = {}
    for name in OPPONENTS:
        totals = []
        for pair, pair_decks in enumerate(all_decks):
            legs = []
            for seat in (1, 0):
                seed = SEED + pair * 2 + seat
                torch.manual_seed(seed)
                opponent = RuleActor(name, seed=seed, equity_samples=32)
                legs.append(play_wizard(agent, opponent, pair_decks, seat, seed, times))
            totals.append(legs)
            if pair % 10 == 9:
                print("EXAM", name, pair + 1, flush=True)
        scores[name] = summarize_match_pairs(totals)
        print("SCORE", name, round(scores[name]["mbb_per_hand"], 1), flush=True)
    return scores


def h2h(wizard_agent, torch_model, all_decks, device, times):
    """Same-deck both-seat H2H. Each entry is the wizard's 50-hand net."""
    wizard_agent.force_argmax = True
    villain = NumpyVillain(wizard_agent)
    hero = _greedy(torch_model)
    paired = []
    for pair, pair_decks in enumerate(all_decks):
        legs = []
        for seat in (1, 0):
            seed = SEED + pair * 2 + seat
            as_hero = play_wizard(wizard_agent, hero, pair_decks, seat, seed, times)
            as_villain = -play_torch(torch_model, villain, pair_decks, seat, seed, device)
            legs.append(0.5 * (as_hero + as_villain))
        paired.append(legs)
        if pair % 10 == 9:
            print("H2H", pair + 1, flush=True)
    return summarize_match_pairs(paired)


def _greedy(model):
    from alphanl.stage2.opponents import NeuralMatchActor

    class Greedy:
        def __init__(self):
            self.inner = NeuralMatchActor(model, kind="residual", snapshot_id="h2h", seed=1)
            self.name = "greedy"
            self.kind = "residual"

        def begin_match(self):
            self.inner.begin_match()

        def begin_hand(self, seat):
            self.inner.begin_hand(seat)

        def observe_action(self, event):
            self.inner.observe_action(event)

        def finish_hand(self, net_bb, *, showdown):
            self.inner.finish_hand(net_bb, showdown=showdown)

        def act(self, snapshot):
            device = next(model.parameters()).device
            batch = {key: value.to(device) for key, value in self.inner.encode(snapshot).items()}
            action, *_ = model.act(batch, deterministic=True)
            return int(action)

    return Greedy()


def calibrate(agent, live, all_decks):
    """Pick beta on two match pairs. Higher worst-style wins, then the three-way sum."""
    best = None
    for beta in (0.1, 0.25, 0.5, 1.0):
        live.BETA = float(beta)
        agent.weights.pop("wizard_beta", None)
        scores = exam(agent, all_decks[:2], [])
        equity = scores["equity"]["mbb_per_hand"]
        tag = scores["TAG_Pro"]["mbb_per_hand"]
        trap = scores["Trapper_CR"]["mbb_per_hand"]
        rank = (min(tag, trap), equity + tag + trap)
        print("BETA", beta, round(equity, 1), round(tag, 1), round(trap, 1), flush=True)
        if best is None or rank > best[0]:
            best = (rank, beta)
    live.BETA = float(best[1])
    agent.weights["wizard_beta"] = np.float32(best[1])
    print("BETA_STAR", best[1], flush=True)
    return float(best[1])


def percentile(samples, q):
    if not samples:
        return 0.0
    return float(np.percentile(np.asarray(samples) * 1000.0, q))


def main():
    torch.set_num_threads(1)
    wizard.bind(evaluate_cards)
    choice = json.loads((OUT / "blueprint.json").read_text())
    base = blueprint_npz(choice)
    # Release string here is temporary; pack_wizard assigns the real Vn.
    model_path = OUT / "wizard_model.npz"
    pack_wizard.write_model(base, OUT / "vleaf.npz", model_path, "pending")
    tree = pack_wizard.prepare_tree(10, "pending", "pending_model.npz")
    sys.path.insert(0, str(tree))
    agent = load_agent(model_path)
    agent.force_argmax = False
    import stage2_runtime.wizard as live
    blueprint, _blob = load_model(Path(choice["pt"]), "cuda")
    blueprint.eval()
    attach_config(agent, blueprint)
    all_decks = decks()
    probe = []
    play_wizard(agent, RuleActor("equity", seed=1, equity_samples=8), all_decks[0], 1, 1, probe)
    if percentile(probe, 99) > 120:
        live.DCFR_ITERS = 20
        print("SEARCH_THROTTLE", live.DCFR_ITERS, round(percentile(probe, 99), 1), flush=True)
    beta = calibrate(agent, live, all_decks)
    with np.load(OUT / "vleaf.npz") as archive:
        leaf = {key: np.array(archive[key]) for key in archive.files}
    leaf["wizard_beta"] = np.float32(beta)
    np.savez(OUT / "vleaf.npz", **leaf)
    times = []
    scores = exam(agent, all_decks, times)
    device = "cuda"
    if choice["kind"] == "v9":
        v9_model = blueprint
    else:
        v9_model, _ = load_model(Path("/root/autodl-tmp/stage2_u250_gen2_20261103/u250_gen2_best.pt"), device)
        v9_model.eval()
    versus_v9 = h2h(agent, v9_model, all_decks, device, times)
    c200, _ = load_model(C200, device)
    c200.eval()
    versus_c200 = h2h(agent, c200, all_decks, device, times)
    equity = scores["equity"]["mbb_per_hand"]
    tag = scores["TAG_Pro"]["mbb_per_hand"]
    trap = scores["Trapper_CR"]["mbb_per_hand"]
    result = {
        "checkpoint": choice["pt"], "blueprint_npz": str(base), "kind": choice["kind"],
        "equity": round(equity, 1), "tag": round(tag, 1), "trapper": round(trap, 1),
        "style_min": round(min(tag, trap), 1),
        "h2h_v9": round(versus_v9["mbb_per_hand"], 1),
        "h2h_c200": round(versus_c200["mbb_per_hand"], 1),
        "mean_ms": round(percentile(times, 50), 2),
        "max_ms": round(max(times) * 1000.0, 2) if times else 0,
        "p99_ms": round(percentile(times, 99), 2),
        "decisions": len(times), "beta": beta,
        "scores": scores,
    }
    (OUT / "wizard_result.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    print("WIZARD_DONE", json.dumps({key: result[key] for key in (
        "equity", "tag", "trapper", "h2h_v9", "h2h_c200", "mean_ms", "max_ms", "p99_ms", "beta")}), flush=True)
    pack_wizard.pack(result)


if __name__ == "__main__":
    main()
