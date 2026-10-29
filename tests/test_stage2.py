"""Exact Stage 1 alignment, complete CPU matches, gradients and persistence."""
from dataclasses import replace
import json
import torch
import pytest
from alphanl.cfr.config import CFRConfig
from alphanl.cfr.distillation import PolicyNetwork, save_student, load_student, SEMANTIC_FIELDS
from alphanl.cfr.actions import CFRActionGuard
from alphanl.cfr.evaluation import RuleActor
from alphanl.cfr.traversal import HunlState
from alphanl.transformer.config import ModelConfig
from alphanl.transformer.hunl import HunlTable, HunlTokenizer
from alphanl.transformer.model import masked_log_probs
from alphanl.transformer.schema import ActionEvent
from alphanl.stage2 import MatchCtx, BotzoneMatchEnv, ResidualExploiterPolicy
from alphanl.stage2.pool import MatchOpponentPool, DEFAULT_WEIGHTS
from alphanl.stage2.policy import safe_all_in
from alphanl.stage2.train import collect_match, train_match, save_checkpoint, load_checkpoint

@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)
    torch.manual_seed(3)

@pytest.fixture
def config():
    return CFRConfig(equity_features=True, model=ModelConfig(d_model=32, nhead=2, num_layers=1,
                        dim_feedforward=64, max_seq_len=24, dropout=.1))

@pytest.mark.parametrize("equity", [False, True])
@pytest.mark.parametrize("seat", [0, 1])
def test_traversal_env_observation_bitwise_equal_all_streets(config, equity, seat):
    config = replace(config, equity_features=equity)
    env = BotzoneMatchEnv(config, RuleActor("call", seed=3))
    table = HunlTable(seed=17)
    table.reset()
    env.table = table
    env.ctx.begin_hand()
    streets = set()
    while not table.is_over():
        actor = table.current_player()
        env.hero = actor
        if actor == seat:
            expected = HunlState(table, HunlTokenizer(config.model.max_seq_len, **config.tokenizer_kwargs()),
                                 CFRActionGuard.from_config(config)).observation()
            actual = env.observe_poker()
            assert actual.keys() == expected.keys()
            for key in expected:
                assert torch.equal(actual[key], expected[key]), key
            assert actual["numeric"][0, 1, actor] == 1
            if equity:
                assert 0 <= actual["numeric"][0, 2, 0] <= 1
            streets.add(table.street)
        table.step(1)
    assert streets == {0, 1, 2, 3}


def event(street=0, me=False, kind="call", chips=50, owed=50, stack=19950):
    return ActionEvent(street, 0, "BB", me, None, kind, 150, chips, chips / 150, 1, owed, stack, (0,1,2,3,4))


def test_context_opportunities_priors_and_roundtrip():
    ctx = MatchCtx()
    assert ctx.as_dict()["three_bet"] == .1
    ctx.begin_hand()
    ctx.observe_action(event(me=True, kind="raise", chips=250))
    ctx.observe_action(event(kind="raise", chips=600, owed=200))
    ctx.observe_action(event(me=True, kind="all_in", chips=19000, owed=350))
    ctx.observe_action(event(kind="all_in", chips=19350, owed=19350, stack=19350))
    ctx.observe_action(event(street=1, kind="fold", chips=0, owed=100))
    assert ctx.opportunities["three_bet"] == 1
    assert ctx.successes["three_bet"] == 1
    assert ctx.successes["pfr"] == 0  # committed at hand end
    assert ctx.successes["flop_aggression"] == 0
    assert ctx.opportunities["flop_aggression"] == 1
    ctx.finish_hand(10, showdown=False)
    assert ctx.as_dict()["bankroll_diff"] == .1
    assert ctx.as_dict()["hand_progress"] == .02
    assert ctx.opportunities["showdown_win"] == 0
    assert ctx.as_dict()["vpip"] < 1
    assert MatchCtx.from_state_dict(json.loads(json.dumps(ctx.state_dict()))).as_dict() == ctx.as_dict()
    ctx.begin_hand()
    ctx.finish_hand(-20, showdown=True)
    assert ctx.successes["showdown_win"] == 1

@pytest.mark.parametrize("width", [128, 192])
def test_zero_residual_exact_reference_even_train_and_bound(config, width):
    config.model = ModelConfig(d_model=width, nhead=4 if width == 128 else 6,
                              num_layers=4, dim_feedforward=256 if width == 128 else 512)
    ref = PolicyNetwork(config).eval()
    model = ResidualExploiterPolicy(ref).train()
    env = BotzoneMatchEnv(config, RuleActor("call", seed=1))
    batch = env.reset()
    expected = ref(batch).softmax(-1)
    actual = model(batch)[0].softmax(-1)
    assert torch.equal(expected, actual)
    assert not model.reference.training
    assert all(not p.requires_grad for p in model.reference.parameters())
    with torch.no_grad():
        model.delta_head.bias.fill_(100)
    prepared = model.prepare(batch)
    raw = model(prepared, mask_illegal=False)[0]
    assert (raw - prepared["ref_logits"]).abs().max() <= model.settings["alpha"]


def test_two_full_matches_backprop_frozen_reference_checkpoint(config, tmp_path):
    model = ResidualExploiterPolicy(PolicyNetwork(config))
    frozen = {k:v.clone() for k,v in model.reference.state_dict().items()}
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.001)
    for seed, name in [(1, "call"), (2, "equity")]:
        env = BotzoneMatchEnv(config, RuleActor(name, seed=seed, equity_samples=4), seed=seed)
        rollout = collect_match(model, env)
        assert env.ctx.hands_played == 50
        assert len(env.hands) == 50
        assert [h["hero_seat"] for h in env.hands] == [1,0] * 25
        assert rollout.dones.sum() == 1 and rollout.dones[-1] == 1
        assert float(rollout.rewards.sum()) == pytest.approx(env.ctx.hero_net_bb / 200, abs=1e-5)
        metrics = train_match(model, optimizer, rollout, epochs=2, minibatch_size=64)
        assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
        assert all(p.grad is None for p in model.reference.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.delta_head.parameters())
        with pytest.raises(RuntimeError):
            env.step(1)
    assert all(torch.equal(v, frozen[k]) for k,v in model.reference.state_dict().items())
    assert model.delta_head.weight.abs().sum() > 0
    penalty, kl, beta = model.kl_penalty(rollout.batch)
    assert torch.isfinite(penalty) and penalty > 0
    assert kl.min() >= 0
    assert beta[0] >= beta[-1]
    path = tmp_path / "stage2.pt"
    save_checkpoint(path, model, optimizer=optimizer, update=2)
    restored, blob = load_checkpoint(path)
    assert blob["update"] == 2
    assert torch.equal(restored(rollout.batch)[0], model(rollout.batch)[0])


def test_pfsp_defaults_difficulty_and_state(config):
    pool = MatchOpponentPool("unused", seed=5)
    assert pool.probabilities() == pytest.approx(DEFAULT_WEIGHTS)
    for _ in range(20):
        pool.record("equity", -1)
        pool.record("call", 1)
    assert pool.probabilities()["equity"] > DEFAULT_WEIGHTS["equity"]
    assert all(p > 0 for p in pool.probabilities().values())
    restored = MatchOpponentPool("unused")
    restored.load_state_dict(pool.state_dict())
    assert restored.probabilities() == pool.probabilities()


def test_optional_safe_all_in():
    table = HunlTable(seed=1); table.reset()
    snap = table.snapshot(1)
    probs = torch.tensor([.1,.4,.3,.19,.01])
    assert torch.equal(safe_all_in(probs, snap), probs)
    safe = safe_all_in(probs, snap, enabled=True)
    assert safe[4] == 0 and safe[1] == pytest.approx(.41)
    assert safe.sum() == pytest.approx(1.)


@pytest.mark.parametrize("width", [32,128,192])
def test_export_numpy_observation_logits_protocol_and_context(config, tmp_path, width):
    import importlib.util
    import subprocess
    import sys
    import numpy as np
    from alphanl.stage2.export import export_agent
    if width != 32:
        config.model = ModelConfig(d_model=width, nhead=6 if width == 192 else 4, num_layers=4,
            dim_feedforward=512 if width == 192 else 256)
    model = ResidualExploiterPolicy(PolicyNetwork(config)).eval()
    with torch.no_grad():
        model.delta_head.weight.normal_(0, .1)
    path = tmp_path / "stage2.pt"
    save_checkpoint(path, model)
    exported = tmp_path / "agent.py"
    export_agent(path, exported)
    spec = importlib.util.spec_from_file_location("stage2_agent", exported)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    agent = module.Agent()
    ctx = MatchCtx(); ctx.begin_hand()
    table = HunlTable(seed=5); table.reset()
    streets = set()
    while not table.is_over():
        snapshot = table.snapshot(table.current_player())
        batch = HunlState(table, HunlTokenizer(config.model.max_seq_len, **config.tokenizer_kwargs()),
                          CFRActionGuard.from_config(config)).observation()
        batch["match_ctx"] = ctx.tensor()
        portable = agent.observation(snapshot, ctx)
        for key in batch:
            assert np.array_equal(portable[key], batch[key].numpy()), key
        expected = model(batch)[0].detach().numpy()
        np.testing.assert_allclose(agent.logits(portable), expected, atol=2e-5, rtol=2e-5)
        streets.add(table.street)
        table.step(1)
    assert streets == {0,1,2,3}
    request = {"my_id":1,"dealer_id":0,"my_chips":19950,"my_cards":[0,5],"public_cards":[],
               "history":[],"hand":0,"total_win_chips":[0,0]}
    payload = {"requests":[request]}
    response = module.respond(agent, payload)
    assert "data" in response
    payload["data"] = response["data"]
    assert module.respond(agent, payload)["data"]["context"] == response["data"]["context"]
    result = subprocess.run([sys.executable, str(exported)], input=json.dumps(payload), text=True,
                            capture_output=True, check=True)
    assert not result.stderr
    assert "data" in json.loads(result.stdout)
    # Pure comparator must agree with the fast evaluator on random 7-card comparisons.
    from alphanl.stage2.portable_evaluator import evaluate_cards
    from phevaluator import evaluate_cards as fast
    rng = np.random.default_rng(5)
    deck = [r+s for s in "cdhs" for r in "23456789TJQKA"]
    for _ in range(200):
        cards = rng.choice(deck, size=9, replace=False).tolist()
        a, b = cards[:7], cards[2:9]
        assert (evaluate_cards(*a) < evaluate_cards(*b)) == (fast(*a) < fast(*b))
        assert (evaluate_cards(*a) == evaluate_cards(*b)) == (fast(*a) == fast(*b))


def test_portable_two_actual_judge_matches_context_survives_100_hands(config, tmp_path):
    import importlib.util
    import random
    from pathlib import Path
    from alphanl.stage2.export import export_agent
    model = ResidualExploiterPolicy(PolicyNetwork(config)).eval()
    path = tmp_path / "stage2.pt"; save_checkpoint(path, model)
    exported = tmp_path / "agent.py"; export_agent(path, exported)
    spec = importlib.util.spec_from_file_location("stage2_agent", exported)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    spec = importlib.util.spec_from_file_location("judge_fixture", Path(__file__).parent / "fixtures/botzone_judge.py")
    fixture = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)
    random.seed(17)
    agent = module.Agent()
    for match in range(2):
        payload = {"requests":[], "responses":[]}
        total = [0,0]
        for hand in range(50):
            game = fixture.Holdem([20000,20000], hand % 2)
            deck = list(range(52)); random.shuffle(deck)
            game.set_deck_array(deck); game.deal_cards_and_blind()
            result = None
            steps = 0
            while not result:
                pid = game.round_idx
                request = {"my_id":pid,"dealer_id":game.dealer_idx,"my_chips":game.player_chips[pid],
                    "my_cards":[c.to_int() for c in game.player_cards[pid]],
                    "public_cards":[c.to_int() for c in game.public_cards],
                    "history": [dict(e) for e in game.history],"hand":hand,"total_win_chips":list(total)}
                if pid == 0:
                    payload["requests"].append(request)
                    response = module.respond(agent, payload)
                    payload["data"] = response["data"]
                    payload["responses"].append(response["response"])
                    assert response["data"]["context"]["hands_played"] == hand
                    assert response["data"]["context"]["hero_net_bb"] == total[0]/100
                    chosen = response["response"]
                else:
                    view = module._package.reconstruct.reconstruct(request)
                    chosen = 0 if module._package.reconstruct.judge_accepts(view,0) else -2
                result = game.player_action(chosen)
                steps += 1
                assert steps < 100
            final = game.get_player_final_chips(result)
            total = [total[i]+final[i]-20000 for i in range(2)]
        assert total[0] + total[1] == 0
        assert payload["data"]["context"]["hands_played"] == 49


def test_kl_direction_anchor_schedule_and_mask(config):
    model = ResidualExploiterPolicy(PolicyNetwork(config), alpha=.7, beta_kl=2., late_beta_kl=.2)
    with torch.no_grad():
        model.delta_head.bias.copy_(torch.tensor([.7,-.2,100.,.4,-100.]))
    mask = torch.tensor([[1.,1.,0.,1.,0.]]).repeat(2,1)
    batch = {"poker_features":torch.zeros(2,32), "ref_logits":torch.tensor([[.1,.3,.2,.8,-1.]]).repeat(2,1),
             "legal_mask":mask, "match_ctx":torch.zeros(2,12)}
    batch["match_ctx"][1,0] = 1.
    final, _ = model(batch)
    _, logs, probs, legal = masked_log_probs(final,mask)
    ref = masked_log_probs(batch["ref_logits"],mask)[2]
    expected = torch.distributions.kl_divergence(torch.distributions.Categorical(probs=probs[:,[0,1,3]]),
                    torch.distributions.Categorical(probs=ref[:,[0,1,3]]))
    penalty, kl, beta = model.kl_penalty(batch)
    torch.testing.assert_close(kl,expected)
    torch.testing.assert_close(beta,torch.tensor([2.,.2]))
    torch.testing.assert_close(penalty,(expected*beta).mean())
    penalty.backward()
    assert model.delta_head.bias.grad[[2,4]].abs().sum() == 0


def test_history_pool_pins_same_snapshot_for_whole_match(config, tmp_path):
    from alphanl.cfr.model import RegretNetwork
    from alphanl.transformer.model import PokerTransformer
    from alphanl.transformer.hunl import RULES_VERSION
    directory = tmp_path / "strategies"; directory.mkdir()
    network = RegretNetwork(PokerTransformer(**config.to_dict()["model"]), fallback=config.fallback)
    for t in (100,101):
        torch.save({"format":"sd_cfr_strategy_v2", "iteration":t, "weight":t,
                    "config":config.to_dict(), "rules_version":RULES_VERSION, "numeric_version":config.numeric_version,
                    "models":{"shared":network.state_dict()}},directory / f"iteration_{t:07d}.pt")
    (directory / "index.json").write_text(json.dumps({"format":"sd_cfr_archive_v1","iterations":[100,101]}))
    model = ResidualExploiterPolicy(PolicyNetwork(config))
    pool = MatchOpponentPool(directory, history_buckets=[[100,101]])
    pool.initialize(model)
    actors = [pool.sample() for _ in range(20)]
    actor = next(actor for name, actor in actors if name == "history")
    iteration, network = actor.snapshot_id, actor.model
    env = BotzoneMatchEnv(config, actor)
    collect_match(model, env)
    assert actor.snapshot_id == iteration and actor.model is network
    assert actor.ctx.hands_played == 50
    assert pool.disk_loads == 2
