import numpy as np
import pytest
from repair_runtime.search_game import PublicState, build_tree, utility_coefficients
from repair_runtime.search_math import DCFR, Node, Terminal

@pytest.mark.parametrize('pot,stacks,owed,sequence,equity,expected', [
    (1000,(9000,9000),0,[(1,0),(1,0)],1.,(10000,9000)),
    (1000,(9000,8600),400,[(1,400)],1.,(10000,8600)),
    (1000,(9000,8600),400,[(3,1000),(1,600)],1.,(10600,8000)),
    (1000,(9000,8600),400,[(0,0)],1.,(9000,9600)),
    (1000,(9000,4000),0,[(4,9000),(1,4000)],1.,(14000,0)),
    (1000,(300,0),700,[(1,300)],0.,(0,1300)),
])
def test_accounting(pot, stacks, owed, sequence, equity, expected):
    state = PublicState(pot, stacks, owed)
    root = state
    for action, amount in sequence:
        state = state.step(action, amount)
    assert np.allclose(state.settlement(equity), expected)
    a,b = utility_coefficients(root,state,100.)
    assert a*equity+b == pytest.approx((expected[0]-stacks[0]-pot/2)/100.)
    assert sum(state.settlement(equity)) == pot + sum(stacks)


def test_terminal_and_allin_domains():
    s = PublicState(1000,(500,0),500)
    assert set(s.actions()) == {0,1}
    leaf = s.step(1,500)
    assert leaf.actions() == {}
    with pytest.raises(ValueError):
        leaf.step(3,100)


def test_button_amounts_are_exact():
    s = PublicState(1000,(9000,9000))
    root = build_tree(s,100.,equity=np.array([[.75]]), root_amounts={1:0,2:500,3:1000,4:9000})
    assert set(root.children) == {1,2,3,4}
    assert root.children[2].key == ((2,500),)
    with pytest.raises(ValueError):
        build_tree(s,100.,equity=np.array([[.75]]), root_amounts={2:33})


def test_successors_and_root_delta_match_unmodified_referee():
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'fixtures/botzone_judge.py'
    spec=importlib.util.spec_from_file_location('math_oracle_judge',path)
    judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
    # Run all branches against the actual referee, including all-in and fold.
    for river_actions in ((0,0),(100,0),(100,300,0),(-2,-2),(100,-1)):
        game=judge.Holdem([20000,20000],dealer_idx=0)
        game.set_deck_array(list(range(52)));game.deal_cards_and_blind()
        for _ in range(6):
            game.player_action(0)
        assert game.round==3 and game.round_idx==1
        root=PublicState(game.pot,(game.player_chips[1],game.player_chips[0]),street=3)
        state=root
        winners=None
        for response in river_actions:
            owed=state.owed
            action=0 if response==-1 else 1 if response==0 or (response==-2 and state.view.chips[state.actor]<=owed) else 4 if response==-2 else 3
            x=0 if action==0 else min(owed,state.view.chips[state.actor]) if action==1 else state.view.chips[state.actor] if action==4 else response
            state=state.step(action,x)
            winners=game.player_action(response)
        assert winners
        final=game.get_player_final_chips(winners)
        eq=1. if winners==[1] else 0. if winners==[0] else .5
        assert np.allclose(state.settlement(eq),[final[1],final[0]])
        a,b=utility_coefficients(root,state,100.)
        assert a*eq+b==pytest.approx((final[1]-root.view.chips[0]-root.view.pot/2)/100.)


def test_limp_keeps_big_blind_option():
    s=PublicState(150,(19950,19900),50,street=0,street_put=(50,100),round_raise=100,pending=2)
    bb=s.step(1,50)
    assert bb.terminal is None and bb.actor==1
    assert set(bb.actions())=={1,2,3,4}
    assert bb.step(1,0).terminal=='round'
