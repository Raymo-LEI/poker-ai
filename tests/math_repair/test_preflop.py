from types import SimpleNamespace as S
import numpy as np
from repair_runtime.search_preflop import class_groups,history_key,solve_depth


def test_169_mass_is_combo_mass():
    groups=class_groups()
    assert len(groups)==169
    assert sum(map(len,groups.values()))==1326
    assert len(groups['AA'])==6 and len(groups['AKs'])==4 and len(groups['AKo'])==12


def test_history_state_machine_distinguishes_threebet_fivebet_allin_call():
    def event(seat,kind,x,owed):
        return S(street=0,actor_seat=seat,action_type=kind,chips_added=x,to_call_before=owed)
    three=[event(1,'raise',200,50),event(0,'raise',800,150)]
    five=three+[event(1,'raise',2200,650),event(0,'all_in',19100,1550)]
    assert history_key(S(events=three))==((1,'raise',200),(0,'raise',800))
    assert history_key(S(events=five))[-1]==(0,'raise',19100)
    call=five+[event(1,'all_in',17550,17550)]
    assert history_key(S(events=call))[-1]==(1,'call',17550)


def test_separate_depth_games_and_actual_call_ranges():
    hs=((('As','Ah'),('7s','2h')),(('Ks','Kh'),('Qs','Jh')))
    a=solve_depth(100,hs,([6,12],[6,12]),iterations=30,runouts=4)
    b=solve_depth(200,hs,([6,12],[6,12]),iterations=30,runouts=4)
    assert a['stack_bb']==100 and b['stack_bb']==200
    assert a['zero_sum_error']<1e-10 and b['zero_sum_error']<1e-10
    assert a['production_chart'] is False
    # Limp is a solved branch, not an assigned -0.5 or a class-index override.
    assert 1 in a['rows'][0]['actions']
    assert any(row['history']==[(1,50)] for row in a['rows'])
    assert a['rows']!=b['rows']
