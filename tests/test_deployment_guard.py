"""v10 tail-risk regressions: visible state, strict thresholds, every action path."""
from dataclasses import replace
import numpy as np
import pytest
from alphanl.stage2.deployment_guard import apply_guard, guard_config, strong_made_hand, V10_GUARD
from alphanl.transformer.schema import HandSnapshot, ActionEvent


def snapshot(**kw):
    values = dict(events=(), hole_cards=("DT", "H4"), board_cards=("DQ", "H3", "S2", "HT", "D6"),
                  me_seat=0, sb_seat=1, street=3, pot=1000, stacks=(19000, 19000), to_call=0,
                  legal_action_ids=(1, 2, 3, 4), big_blind=100, starting_stack=20000)
    values.update(kw)
    return HandSnapshot(**values)


def run(s, p, equity=.6, config=None, amounts=None):
    return apply_guard(s, p, guard_config(V10_GUARD if config is None else config),
                       lambda *a, **kw: equity, amounts)


def test_reported_ten_pair_never_shoves_small_pot():
    s = snapshot()
    assert not strong_made_hand(s)
    p, audit = run(s, [0, .10, .10, .10, .70])
    assert audit["active_cap"] and audit["remapped_to"] == 3
    np.testing.assert_allclose(p, [0, .1, .1, .8, 0])


def test_reported_pair_cannot_call_or_reraise_overbet_even_with_high_raw_probability():
    event = ActionEvent(3, 1, "SB", False, 4, "all_in", 1000, 19000, 19, 0, 0, 19000, ())
    s = snapshot(events=(event,), pot=20000, stacks=(19000, 0), to_call=19000, legal_action_ids=(0, 1))
    for equity in (.60, .99):  # a faulty optimistic equity alone cannot authorize one-pair calls
        p, audit = run(s, [.1, .9, 0, 0, 0], equity)
        assert p.tolist() == [1, 0, 0, 0, 0] and audit["call_fuse"] == "overbet_shove"
        assert audit["pot_before_bet"] == 1000 and audit["pot_before_source"] == "history"
    # Another large raise with chips behind must block all continuing slots.
    s = replace(s, legal_action_ids=(0, 1, 2, 3, 4), stacks=(19000, 9000), to_call=10000)
    assert run(s, [.1, .4, .1, .1, .3])[0].tolist() == [1, 0, 0, 0, 0]


def test_pruning_is_strict_and_confidence_uses_unpruned_call_probability():
    p, info = run(snapshot(), [0, .09, .10, .76, .05])
    assert info["pruned"] == [1, 4]
    np.testing.assert_allclose(p, [0, 0, .10/.86, .76/.86, 0])
    # .34 call would exceed .35 after pruning; original confidence must trigger fold.
    s = snapshot(pot=20000, to_call=4100, legal_action_ids=(0, 1, 2, 3, 4))
    assert run(s, [.04, .34, .04, .54, .04], equity=.9, config=dict(V10_GUARD, big_call_enabled=True))[1]["call_fuse"] == "big_call"


@pytest.mark.parametrize("pot,stack,blocked", [(1000, 3000, False), (2000,4000,False), (1999,4000,True)])
def test_active_cap_strict_boundaries(pot, stack, blocked):
    _, info = run(snapshot(pot=pot, stacks=(stack, stack)), [0, .2, .1, .1, .6])
    assert info["active_cap"] == blocked


def test_fuse_boundaries_and_real_strong_hand_exception():
    cfg = dict(V10_GUARD, big_call_enabled=False)
    for call, before, blocked in [(5000,1000,False),(6000,5000,False),(6001,5000,True)]:
        s = snapshot(to_call=call, pot=before+call, legal_action_ids=(0,1))
        assert bool(run(s, [.2,.8,0,0,0],config=cfg)[1]["call_fuse"]) == blocked
    s = snapshot(hole_cards=("S4", "D5"), to_call=19000, pot=20000, legal_action_ids=(0,1))
    assert strong_made_hand(s)
    assert run(s, [.2,.8,0,0,0],equity=.82,config=cfg)[1]["call_fuse"] is None
    assert run(s, [.2,.8,0,0,0],equity=.819,config=cfg)[1]["call_fuse"] == "overbet_shove"


def test_board_only_strength_and_kicker_not_monster():
    assert not strong_made_hand(snapshot(hole_cards=("SA","DK"), board_cards=("CQ","DQ","H3","S3","C2")))
    assert not strong_made_hand(snapshot(hole_cards=("SA","DK"), board_cards=("CQ","DQ","H3","S3"), street=2))
    assert not strong_made_hand(snapshot(hole_cards=("SA","DK"), board_cards=("CQ","DQ","HQ"), street=1))
    assert strong_made_hand(snapshot(hole_cards=("SQ","DK"), board_cards=("CQ","DQ","H3"), street=1))


def test_legal_min_raise_does_not_bypass_cap_and_disabled_preserves_policy():
    p, info = run(snapshot(), [0,.2,.1,.1,.6], amounts={2:10000,3:10000})
    assert info["remapped_to"] == 1 and p[4] == 0
    raw = [0,.04,.04,.04,.88]
    p, info = run(snapshot(), raw, config={"enabled":False})
    np.testing.assert_allclose(p, raw)
    assert not info["active_cap"]
    for config in ({"prune_below":float('nan')},{"enabled":"false"},{"typo":1},{"equity_samples":0}):
        with pytest.raises(ValueError):
            guard_config(config)
