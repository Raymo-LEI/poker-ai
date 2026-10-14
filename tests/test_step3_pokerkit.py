"""PokerKit heads-up backend: action order, min-raise, side pots, and the training switch."""

import importlib.util
import time

import pytest
import torch

from alphanl.transformer.tables import PokerkitTable, RlcardTable, aligned_decks, chips_for_depth

HAS_POKERKIT = importlib.util.find_spec("pokerkit") is not None
HAS_RLCARD = importlib.util.find_spec("rlcard") is not None

# Seat 0 is the big blind in both engines when rlcard's dealer_id is 0.
_BB_AA = ("Ah", "Ad")
_SB_KK = ("Kh", "Kd")
_BOARD = ("2c", "7d", "9h", "Js", "Qd")


pytestmark = pytest.mark.skipif(not HAS_POKERKIT, reason="pokerkit is not installed")


def _table(stack_bb=50, **kwargs) -> PokerkitTable:
    return PokerkitTable(seed=0, starting_stack=chips_for_depth(stack_bb), **kwargs)


def _play(table, actions):
    trace = []
    for action in actions:
        legal = table.legal_action_ids()
        if action not in legal:
            raise AssertionError(f"scripted action {action} not in {legal} on street {table.street}")
        trace.append(
            {
                "street": table.street,
                "seat": table.current_player(),
                "action": action,
                "legal": legal,
                "chips": table.action_chips(action) if hasattr(table, "action_chips") else None,
            }
        )
        table.step(action)
    return trace


def _pay(table):
    return tuple(int(value) for value in table.payoffs())


def test_action_order_button_sb_preflop_bb_postflop():
    rl_deck, pk_deck = aligned_decks(_BB_AA, _SB_KK, _BOARD)
    table = _table()
    table.reset(deck=pk_deck)
    assert table.sb_seat == 1
    assert table.hole_cards(0) == ("HA", "DA")
    assert table.hole_cards(1) == ("HK", "DK")
    trace = _play(table, [1, 1, 1, 1, 1, 1, 1, 1])
    assert [(row["street"], row["seat"]) for row in trace] == [
        (0, 1),
        (0, 0),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 1),
    ]
    assert table.board_cards() == ("C2", "D7", "H9", "SJ", "DQ")
    # A free check has no fold. The big blind opens the flop.
    assert 0 not in trace[2]["legal"]
    assert trace[2]["seat"] == 0
    assert _pay(table) == (2, -2)
    assert table.chips_in_play() == 200


def test_min_raise_masks_underraises_and_reopens_on_a_full_raise():
    table = _table()
    table.reset()
    # Blinds 1/2, pot 3. Minimum raise is to 4. Half-pot adds 1 and is not a raise.
    assert table.current_player() == 1
    assert table.min_raise_to() == 4
    assert table.can_raise_to(4)
    assert not table.can_raise_to(3)
    assert table.legal_action_ids() == (0, 1, 3, 4)
    assert table.action_chips(2) is None
    assert table.action_chips(3) == 3
    assert table.action_chips(4) == 99

    table.step(3)  # small blind pots to 4; pot is now 6
    assert table.min_raise_to() == 6
    assert table.action_chips(2) is None  # to 5, short of the minimum
    assert table.action_chips(3) == 6  # to 8
    table.step(3)
    # The pot raise was a full raise, so the small blind may raise again. Minimum is to 12.
    assert table.current_player() == 1
    assert table.min_raise_to() == 12
    assert not table.can_raise_to(11)
    assert table.can_raise_to(12)
    assert table.action_chips(2) is None
    assert 3 in table.legal_action_ids()
    assert 4 in table.legal_action_ids()


def test_short_all_in_does_not_reopen_and_side_pot_returns_uncalled_chips():
    short = PokerkitTable(seed=1, starting_stack=100, stacks=(5, 100))
    short.reset()
    assert short.action_chips(3) == 3  # small blind still pots to 4
    short.step(3)
    # Big blind has 5 total and can only shove to 5, one chip short of the minimum raise to 6.
    assert short.min_raise_to() == 5
    assert short.action_chips(3) is None
    assert short.action_chips(4) == 3
    assert short.can_raise_to(5)
    assert not short.can_raise_to(6)
    short.step(4)
    assert short.legal_action_ids() == (0, 1)
    assert short.min_raise_to() is None
    assert short.action_chips(1) == 1
    short.step(0)
    assert short.chips_in_play() == 105
    assert sum(_pay(short)) == 0

    rl_deck, pk_deck = aligned_decks(_BB_AA, _SB_KK, _BOARD)
    sided = PokerkitTable(seed=2, starting_stack=100, stacks=(30, 100))
    sided.reset(deck=pk_deck)
    # Deep small blind shoves 100. Short big blind calls 30. 70 comes back uncalled.
    _play(sided, [4, 1])
    assert sided.board_cards() == ("C2", "D7", "H9", "SJ", "DQ")
    assert _pay(sided) == (30, -30)
    assert tuple(int(stack) for stack in sided.state.stacks) == (60, 70)
    assert sided.chips_in_play() == 130

    matched = PokerkitTable(seed=3, starting_stack=100, stacks=(100, 30))
    matched.reset(deck=pk_deck)
    _play(matched, [4, 1])
    assert _pay(matched) == (30, -30)
    assert tuple(int(stack) for stack in matched.state.stacks) == (130, 0)
    assert matched.chips_in_play() == 130


def test_legal_mask_and_real_pot_fraction():
    from alphanl.transformer.tokenizer import BettingHistoryTokenizer

    rl_deck, pk_deck = aligned_decks(_BB_AA, _SB_KK, _BOARD)
    table = _table()
    table.reset(deck=pk_deck)
    _play(table, [1, 1])
    assert table.street == 1
    assert table.legal_action_ids()[0] != 0
    assert table.action_chips(2) == 2
    assert table.min_raise_to() == 2
    table.step(2)
    events = table.events_for(me_seat=0)
    bet = events[-1]
    assert bet.action_type == "bet"
    assert bet.position == "BB"
    assert bet.street == 1
    assert bet.pot_before == 4
    assert bet.chips_added == 2
    assert bet.pot_fraction == 0.5
    assert bet.size_bucket == 2
    assert bet.action_id == 2

    snap = table.snapshot(0)
    batch = BettingHistoryTokenizer(max_seq_len=32).encode(snap)
    mask = [int(value) for value in batch["legal_mask"][0].tolist()]
    assert mask == [1 if action in snap.legal_action_ids else 0 for action in range(5)]
    assert set(mask) <= {0, 1}
    assert sum(mask) == len(snap.legal_action_ids) >= 1

    illegal = 2 if 2 not in table.legal_action_ids() else 99
    with pytest.raises(ValueError):
        table.step(illegal)


@pytest.mark.parametrize("stack_bb", [50, 100, 200])
def test_random_play_conserves_chips(stack_bb):
    import numpy as np

    stack = chips_for_depth(stack_bb)
    table = PokerkitTable(seed=stack_bb, starting_stack=stack)
    rng = np.random.default_rng(stack_bb)
    saw_flop = False
    for _hand in range(40):
        table.reset()
        assert table.chips_in_play() == 2 * stack
        first_on_street = {}
        for _step in range(120):
            if table.is_over():
                break
            street = table.street
            first_on_street.setdefault(street, table.current_player())
            legal = table.legal_action_ids()
            assert legal and set(legal) <= {0, 1, 2, 3, 4}
            minimum = table.min_raise_to()
            bet = int(table.state.bets[table.current_player()])
            for action in legal:
                if action in (2, 3, 4):
                    assert minimum is not None
                    assert bet + table.action_chips(action) >= minimum
            table.step(int(rng.choice(legal)))
            assert table.chips_in_play() == 2 * stack
        else:
            raise AssertionError("hand did not finish")
        assert sum(_pay(table)) == 0
        stacks = tuple(int(value) for value in table.state.stacks)
        assert sum(stacks) == 2 * stack
        assert _pay(table) == (stacks[0] - stack, stacks[1] - stack)
        assert first_on_street[0] == 1
        if 1 in first_on_street:
            saw_flop = True
            assert first_on_street[1] == 0
            for street in (2, 3):
                if street in first_on_street:
                    assert first_on_street[street] == 0
    assert saw_flop


def test_random_play_throughput_on_cpu(capsys):
    import numpy as np

    table = _table()
    rng = np.random.default_rng(0)
    hands = 200
    started = time.perf_counter()
    for _hand in range(hands):
        table.reset()
        guard = 0
        while not table.is_over():
            table.step(int(rng.choice(table.legal_action_ids())))
            guard += 1
            if guard > 120:
                raise AssertionError("hand did not finish")
    elapsed = time.perf_counter() - started
    rate = hands / elapsed
    print(f"pokerkit random-play throughput: {rate:.1f} hands/sec over {hands} hands ({elapsed:.2f}s)")
    assert rate > 5.0


def test_env_masks_checkpoint_and_depths():
    import numpy as np

    from alphanl.transformer.env import SelfPlayEnv
    from alphanl.transformer.pool import OpponentPool
    from alphanl.transformer.tokenizer import BettingHistoryTokenizer

    class _Call:
        def act(self, snapshot, choice):
            legal = list(snapshot.legal_action_ids)
            return 1 if 1 in legal else int(legal[0])

        def prepare_pool(self, weights):
            return None

    def _make(seed, stack):
        tokenizer = BettingHistoryTokenizer(max_seq_len=32)
        pool = OpponentPool(max_size=1, selfplay_prob=0.0, random_prob=1.0, seed=seed)
        env = SelfPlayEnv(
            tokenizer,
            _Call(),
            pool,
            seed=seed,
            starting_stack=stack,
            backend="pokerkit",
            record_results=False,
        )
        env.force_opponent = {"kind": "random", "policy_id": None, "weights": None}
        return env

    for stack_bb in (50, 100, 200):
        env = _make(stack_bb, chips_for_depth(stack_bb))
        obs = env.reset()
        assert obs["legal_mask"].shape[-1] == 5
        assert int(obs["legal_mask"].sum()) >= 1
        assert set(int(v) for v in obs["legal_mask"][0].tolist()) <= {0, 1}
        done = False
        reward = 0.0
        for _ in range(40):
            legal = torch.where(obs["legal_mask"][0] > 0.5)[0]
            obs, reward, done, info = env.step(int(legal[0]))
            assert info["kind"] == "random"
            if done:
                break
        assert done
        assert abs(reward) <= stack_bb + 1e-6

    env = _make(7, 100)
    obs = env.reset()
    other = _make(8, 100)
    other.load_state_dict(env.state_dict())
    assert torch.equal(obs["token_ids"], other.observe()["token_ids"])
    assert torch.equal(obs["legal_mask"], other.observe()["legal_mask"])
    for _ in range(30):
        action = int(torch.where(env.observe()["legal_mask"][0] > 0.5)[0][0])
        _, reward_a, done_a, _ = env.step(action)
        _, reward_b, done_b, _ = other.step(action)
        assert reward_a == reward_b and done_a == done_b
        if done_a:
            break
        assert torch.equal(env.observe()["token_ids"], other.observe()["token_ids"])
    else:
        raise AssertionError("checkpointed hand did not finish")

    with pytest.raises(ValueError):
        PokerkitTable(seed=1, starting_stack=200).load_state_dict(env.table.state_dict())


def test_env_clamps_illegal_abstract_action():
    from alphanl.transformer.env import SelfPlayEnv
    from alphanl.transformer.pool import OpponentPool
    from alphanl.transformer.tokenizer import BettingHistoryTokenizer

    class _Call:
        def act(self, snapshot, choice):
            return int(snapshot.legal_action_ids[0])

        def prepare_pool(self, weights):
            return None

    tokenizer = BettingHistoryTokenizer(max_seq_len=32)
    pool = OpponentPool(max_size=1, selfplay_prob=0.0, random_prob=1.0, seed=0)
    env = SelfPlayEnv(
        tokenizer, _Call(), pool, seed=4, backend="pokerkit", record_results=False
    )
    env.force_opponent = {"kind": "random", "policy_id": None, "weights": None}
    env.reset()
    before = len(env.table.events_for(0))
    env.step(99)
    event = env.table.events_for(0)[before]
    assert event.action_id == event.legal_action_ids[0]


def test_one_ppo_update_pokerkit(tmp_path):
    from alphanl.transformer.config import ExperimentConfig
    from alphanl.transformer.train import run_training

    cfg = ExperimentConfig()
    cfg.model.d_model = 32
    cfg.model.nhead = 4
    cfg.model.num_layers = 1
    cfg.model.dim_feedforward = 64
    cfg.model.max_seq_len = 32
    cfg.model.dropout = 0.0
    cfg.device = "cpu"
    cfg.rollout_device = "cpu"
    cfg.seed = 0
    cfg.env.backend = "pokerkit"
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
    cfg.train.out_dir = str(tmp_path)
    cfg.train.checkpoint_dir = str(tmp_path / "ckpt")
    run_training(cfg)
    text = (tmp_path / "metrics.csv").read_text()
    assert "policy_loss" in text
    assert (tmp_path / "ckpt" / "update_0000000.pt").is_file()


@pytest.mark.skipif(not HAS_RLCARD, reason="rlcard is not installed")
def test_replay_matches_rlcard_except_postflop_order():
    rl_deck, pk_deck = aligned_decks(_BB_AA, _SB_KK, _BOARD)

    def _pair():
        rl = RlcardTable(seed=0, starting_stack=100, dealer_id=0)
        pk = PokerkitTable(seed=0, starting_stack=100)
        rl.reset(deck=rl_deck)
        pk.reset(deck=pk_deck)
        assert rl.sb_seat == pk.sb_seat == 1
        assert rl.hole_cards(0) == pk.hole_cards(0) == ("HA", "DA")
        assert rl.hole_cards(1) == pk.hole_cards(1) == ("HK", "DK")
        return rl, pk

    # Preflop lines do not reach the postflop order difference.
    for actions, expected in (
        ([0], (1, -1)),
        ([3, 0], (-2, 2)),
    ):
        rl, pk = _pair()
        rl_trace = _play(rl, actions)
        pk_trace = _play(pk, actions)
        assert [row["seat"] for row in rl_trace] == [row["seat"] for row in pk_trace]
        assert _pay(rl) == _pay(pk) == expected
        rl_chips = [event.chips_added for event in rl.events_for(0)]
        pk_chips = [event.chips_added for event in pk.events_for(0)]
        assert rl_chips == pk_chips

    # All-in before the flop. The board is dealt with no further decisions.
    rl, pk = _pair()
    _play(rl, [4, 1])
    _play(pk, [4, 1])
    assert rl.board_cards() == pk.board_cards() == ("C2", "D7", "H9", "SJ", "DQ")
    assert _pay(rl) == _pay(pk) == (100, -100)

    # Checkdown: the same checks, different postflop order, same showdown.
    rl, pk = _pair()
    rl_trace = _play(rl, [1, 1, 1, 1, 1, 1, 1, 1])
    pk_trace = _play(pk, [1, 1, 1, 1, 1, 1, 1, 1])
    assert [(row["street"], row["seat"]) for row in rl_trace[:2]] == [(0, 1), (0, 0)]
    assert [(row["street"], row["seat"]) for row in pk_trace[:2]] == [(0, 1), (0, 0)]
    assert rl_trace[2]["seat"] == 1  # rlcard: small blind still first on the flop
    assert pk_trace[2]["seat"] == 0  # pokerkit: big blind first
    assert rl.board_cards() == pk.board_cards()
    assert _pay(rl) == _pay(pk) == (2, -2)

    # Same abstract actions after the flop. rlcard's small blind bets; pokerkit's big blind bets.
    rl, pk = _pair()
    rl_trace = _play(rl, [1, 1, 3, 0])
    pk_trace = _play(pk, [1, 1, 3, 0])
    assert rl_trace[2]["seat"] == 1 and pk_trace[2]["seat"] == 0
    assert [row["chips"] for row in pk_trace] == [1, 0, 4, 0]
    assert _pay(pk) == (2, -2)
    assert _pay(rl) == (-2, 2)


def test_legacy_rlcard_checkpoint_still_loads():
    if not HAS_RLCARD:
        pytest.skip("rlcard is not installed")
    import pickle

    from alphanl.transformer.env import SelfPlayEnv
    from alphanl.transformer.pool import OpponentPool
    from alphanl.transformer.tokenizer import BettingHistoryTokenizer

    class _Call:
        def act(self, snapshot, choice):
            return int(snapshot.legal_action_ids[0])

        def prepare_pool(self, weights):
            return None

    tokenizer = BettingHistoryTokenizer(max_seq_len=32)
    pool = OpponentPool(max_size=1, selfplay_prob=0.0, random_prob=1.0, seed=0)
    env = SelfPlayEnv(tokenizer, _Call(), pool, seed=5, backend="rlcard", record_results=False)
    env.force_opponent = {"kind": "random", "policy_id": None, "weights": None}
    obs = env.reset()
    legacy = {
        "rng": env.rng.bit_generator.state,
        "rl": pickle.dumps(env.table.rl),
        "fields": pickle.dumps(
            {
                "history": env.table.history,
                "state": env.table.state,
                "player": env.table.player,
                "hero": env.hero,
                "sb_seat": env.sb_seat,
                "choice": env.choice,
                "passive_rewards_bb": list(env.passive_rewards_bb),
                "force_opponent": env.force_opponent,
            }
        ),
    }
    other = SelfPlayEnv(tokenizer, _Call(), pool, seed=6, backend="rlcard", record_results=False)
    other.load_state_dict(legacy)
    assert torch.equal(obs["token_ids"], other.observe()["token_ids"])
    pokerkit_env = SelfPlayEnv(
        tokenizer, _Call(), pool, seed=6, backend="pokerkit", record_results=False
    )
    with pytest.raises(ValueError):
        pokerkit_env.load_state_dict(legacy)
