from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from alphanl.cfr.config import CFRConfig
from alphanl.cfr.distillation import PolicyNetwork, SEMANTIC_FIELDS, save_student
from alphanl.cfr.evaluation import OPPONENTS, RuleActor
from alphanl.cfr.gauntlet import Candidate, ModelActor, match_factories, play_block, paired_candidate_gain, select_candidate, GAUNTLET_OPPONENTS
from alphanl.cfr.pro_bots import ProScriptActor, preflop_percentiles, texture
from alphanl.stage2.policy import ResidualExploiterPolicy
from alphanl.stage2.train import save_checkpoint
from alphanl.transformer.config import ModelConfig
from alphanl.transformer.hunl import HunlTable
from alphanl.transformer.schema import ActionEvent, HandSnapshot

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eval_pro_slumbot import evaluate_slumbot, public_event
from eval_cfr_slumbot import decision_table
from extract_stage2_eval_candidates import extract
from monitor_training import readiness_rows


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)
    torch.manual_seed(1)


def snapshot(hole=("SA", "HA"), board=("DA", "C7", "H2"), **kwargs):
    return HandSnapshot(events=(), hole_cards=hole, board_cards=board,
        me_seat=0, sb_seat=1, street=1, pot=600, stacks=(19700,19700), to_call=0,
        legal_action_ids=(1,2,3,4), big_blind=100, starting_stack=20000, **kwargs)


def event(seat=0, kind="check", chips=0, call=0):
    return ActionEvent(1, seat, "BB" if seat == 0 else "SB", seat == 0, None, kind,
                       600, chips, chips/600, 0, call, 19700, (0,1,2,3,4))


def tiny_model():
    return PolicyNetwork(CFRConfig(model=ModelConfig(d_model=16,nhead=2,num_layers=1,
        dim_feedforward=32,max_seq_len=24,dropout=0.))).eval()


def test_preflop_weighted_percentile_and_position():
    p = preflop_percentiles()
    assert p['AA'] > p['KK'] > p['72o']
    # 35% raise, 30% limp by combination-weighted random-equity rank.
    weight = lambda k: 6 if len(k)==2 else 4 if k.endswith('s') else 12
    assert sum(weight(k) for k,v in p.items() if v >= .65) / 1326 == pytest.approx(.35,abs=.01)
    assert sum(weight(k) for k,v in p.items() if v >= .35) / 1326 == pytest.approx(.65,abs=.01)
    table=HunlTable(seed=1); table.reset()
    s=table.snapshot(1)
    bot=ProScriptActor('TAG_Pro',seed=7)
    assert bot.act(replace(s,hole_cards=('SA','HA'))) in (2,3)
    assert bot.act(replace(s,hole_cards=('S7','H2'))) == 0
    # New evaluation names do not alter the original training/evaluation list.
    assert OPPONENTS == ('random','call','maniac','overbet','nit','equity')
    assert RuleActor('TAG_Pro',seed=3).act(s) in s.legal_action_ids


def test_monsters_draws_and_board_only_exclusion():
    assert texture(snapshot())['strong']
    assert texture(snapshot(hole=('SA','HK'),board=('DA','CK','H2')))['strong']
    assert texture(snapshot(hole=('S9','H8'),board=('D7','C6','H2')))['strong_draw']
    assert texture(snapshot(hole=('SA','SK'),board=('S2','S7','H3')))['strong_draw']
    assert not texture(snapshot(hole=('C2','H3'),board=('ST','SJ','SQ','SK','SA')))['strong']
    assert texture(snapshot(hole=('SK','HQ'),board=('DA','C7','H2')))['dry']


def test_trapper_checks_then_raises_and_resets():
    checks=0
    for seed in range(120):
        bot=ProScriptActor('Trapper_CR',seed=seed,equity_samples=4)
        s=snapshot()
        if bot.act(s) == 1:
            checks+=1
            response=replace(s,to_call=300,pot=900,legal_action_ids=(0,1,2,3,4),
                events=(event(),event(1,'bet',300)))
            assert bot.act(response) in (2,3)
            assert bot.counts['check_raises'] == 1
            bot.begin_hand(0)
            assert not bot.trap_streets
    assert 40 <= checks <= 80


def test_zero_residual_and_context_are_symmetric_and_complete():
    model=tiny_model()
    a=ModelActor(model,seed=7)
    b=ModelActor(ResidualExploiterPolicy(model).eval(),seed=7,residual=True)
    seeds=np.arange(50)
    left, first=play_block(a,RuleActor('call',seed=1),seeds,1)
    right, second=play_block(b,RuleActor('call',seed=1),seeds,1)
    assert left == right
    assert first == second
    assert a.ctx.hands_played == b.ctx.hands_played == 50
    torch.testing.assert_close(a.ctx.tensor(),b.ctx.tensor(),rtol=0,atol=0)
    c,d=ModelActor(model,seed=8),ModelActor(model,seed=9)
    play_block(c,d,seeds,0)
    assert c.ctx.hands_played == d.ctx.hands_played == 50
    assert c.ctx.hero_net_bb == -d.ctx.hero_net_bb
    assert sum(c.ctx.opportunities.values()) > 50


def test_clustered_common_deals_and_no_smoke_champion():
    records=[]
    model=tiny_model()
    schedule=np.arange(100).reshape(2,50)
    result=match_factories(lambda seed:ModelActor(model,seed=seed),
        lambda seed:RuleActor('call',seed=seed),schedule=schedule,seed=4,log=records.append)
    assert result['deck_pairs'] == 100 and result['hands'] == 200 and result['match_pairs'] == 2
    assert records[0]['deck_seeds'] == records[1]['deck_seeds']
    assert paired_candidate_gain(result,result)['ci95'] == [0.,0.]
    matrix={'B':{o:result for o in GAUNTLET_OPPONENTS}}
    selected=select_candidate(matrix,['B'],full_budget=False,all_required_present=True)
    assert selected['champion'] is None and selected['recommendation'] is None


def test_candidate_update_mismatch_and_bank_extraction(tmp_path):
    model=ResidualExploiterPolicy(tiny_model()).eval()
    path=tmp_path/'stage2.pt'
    save_checkpoint(path,model,update=300)
    with pytest.raises(ValueError,match='requested update'):
        Candidate(dict(name='C100',kind='stage2',path=str(path),update=100))
    blob=torch.load(path,weights_only=True)
    bank={100:{k:v.clone() for k,v in model.state_dict().items() if not k.startswith('reference.')}}
    blob['pool']={'self_weights':bank}
    torch.save(blob,path)
    extracted=extract(path,tmp_path/'out',[100])
    candidate=Candidate(dict(name='C100',kind='stage2',path=str(tmp_path/'out/update_0000100.pt'),update=100))
    assert extracted['artifacts']['100']['sha256'] == candidate.metadata['sha256']
    with pytest.raises(ValueError,match='exact complete'):
        extract(path,tmp_path/'bad',[200])


def test_teacher_window_is_exact_and_matches_complete_block(tmp_path):
    from alphanl.cfr.model import RegretNetwork
    from alphanl.transformer.model import PokerTransformer
    from alphanl.transformer.hunl import RULES_VERSION
    config=tiny_model().config
    directory=tmp_path/'strategies'
    directory.mkdir()
    for t in (1,2,3):
        model=RegretNetwork(PokerTransformer(**config.to_dict()['model']),fallback=config.fallback)
        torch.save(dict(format='sd_cfr_strategy_v2',iteration=t,weight=t,
            rules_version=RULES_VERSION,numeric_version=config.numeric_version,
            config=config.to_dict(),models={'shared':model.state_dict()}),directory/f'iteration_{t:07d}.pt')
    candidate=Candidate(dict(name='A',kind='teacher',path=str(directory),start=2,end=3))
    actor=candidate.actor(19)
    total,records=play_block(actor,RuleActor('call',seed=1),np.arange(50),1)
    assert {r['left_snapshot'] for r in records} <= {2,3}
    assert len(records)==50 and np.isfinite(total)
    assert len(candidate.metadata['snapshot_sha256'])==2
    with pytest.raises(ValueError,match='requested strategy snapshot'):
        Candidate(dict(name='missing',kind='teacher',path=str(directory),start=2,end=4))


def test_cli_preserves_required_missing_failure_and_html_escaping(tmp_path):
    from eval_pro_gauntlet import main, publish
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(candidates=[dict(name='A',kind='teacher',path='missing',start=150,end=350)])))
    with pytest.raises(FileNotFoundError,match='required candidate'):
        main(['--manifest',str(manifest),'--out-dir',str(tmp_path/'out'),'--pairs','100'])
    report=json.loads((tmp_path/'out/report.json').read_text())
    assert not report['complete'] and report['missing'][0]['name']=='A'
    report['matrix']={'<script>':{}}
    publish(tmp_path/'out',report)
    assert '<script>' not in (tmp_path/'out/matrix.html').read_text()
    assert '&lt;script&gt;' in (tmp_path/'out/matrix.html').read_text()


class FakeCandidate:
    name='fake'
    metadata={'name':'fake','kind':'student'}
    def actor(self,seed):
        return ModelActor(tiny_model(),seed=seed)


def test_slumbot_terminal_walk_context_and_secret_allowlist(tmp_path):
    class Response:
        def raise_for_status(self): pass
        def json(self):
            return dict(token='SECRET',client_pos=0,action='f',hole_cards=['As','Kh'],board=[],
                        winnings=50,bot_hole_cards=['2c','3d'])
    class Session:
        def post(self,*args,**kwargs):return Response()
    report=evaluate_slumbot(FakeCandidate(),tmp_path/'online',hands=100,seed=3,session=Session(),delay=0)
    assert report['performance']['mean'] == 500
    assert report['performance']['ci95'] == [500,500]
    assert report['completed_hands'] == 100
    assert report['audit']['counts']['hands_bb'] == 100
    assert 'SECRET' not in (tmp_path/'online/hands.jsonl').read_text()
    assert 'bot_hole_cards' not in (tmp_path/'online/hands.jsonl').read_text()
    assert readiness_rows(report)[0]['eval/winrate_vs_slumbot_mbb'] == 500


def test_slumbot_no_retry_or_secret_error(tmp_path):
    class Session:
        calls=0
        def post(self,*args,**kwargs):
            self.calls+=1
            raise TimeoutError('SECRET')
    session=Session()
    with pytest.raises(TimeoutError):
        evaluate_slumbot(FakeCandidate(),tmp_path/'failed',hands=1,seed=1,session=session,delay=0)
    assert session.calls == 1
    report=json.loads((tmp_path/'failed/report.json').read_text())
    assert not report['complete'] and report['completed_hands']==0
    assert 'SECRET' not in (tmp_path/'failed/report.json').read_text()


def test_slumbot_offtree_exact_chips_and_public_events():
    telemetry={}
    table,parsed=decision_table(dict(client_pos=0,action='b333',hole_cards=['As','Kh'],board=[]),
                                bet_telemetry=telemetry)
    assert telemetry[0] == dict(player=1,off_tree=True)
    e=public_event(parsed.events[0],0)
    assert e.chips_added == 283 and e.to_call_before == 50 and not e.actor_is_me
    assert table.snapshot(0).to_call == 233
