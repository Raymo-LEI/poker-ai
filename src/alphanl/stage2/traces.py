"""Bounded complete-hand JSONL for the Overwatch replay contract.

No inference, RNG draws, hidden opponent cards or counterfactual EV calculations.
"""
import json
from pathlib import Path


def category(events, hero):
    last = events[-1]
    if hero == 1 and events[0]['seat'] == hero and events[0]['action'] == 0:
        return 'sb_open_fold'
    if last['street'] == 0:
        return 'other_preflop_terminal'
    if last['action'] == 0:
        return 'hero_postflop_fold' if last['seat'] == hero else 'villain_postflop_fold'
    aggressive = [e for e in events if e['chips_added'] > e['to_call']]
    if aggressive and aggressive[-1]['seat'] == hero:
        return 'showdown_hero_last_aggressor'
    if any(e['seat'] == hero and e['street'] > 0 and e['action'] == 1
           and e['to_call'] > 0 and e['to_call'] >= .5 * e['pot'] for e in events):
        return 'showdown_villain_last_aggressor_large_hero_call'
    return 'other_showdown'


class HandTraceWriter:
    """New source per update; publish metadata only after closing the JSONL.

    The byte limit excludes report.json. Once full, discard subsequent complete
    hands and count them; never write partial JSONL rows. Sampling is positional,
    independent of result, so this file is not a representative league estimate.
    """
    def __init__(self, directory, *, update, scope, max_bytes=16 * 1024 * 1024):
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1024:
            raise ValueError('trace max_bytes must be an integer >= 1024')
        if scope not in ('training', 'evaluation'):
            raise ValueError('invalid trace scope')
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.stream = (self.directory / 'hands.jsonl').open('x', encoding='utf-8')
        self.metadata = dict(format='stage2_hand_trace_v1', update=update, scope=scope,
                             big_blind_chips=100, rules_version='standard_hunl_200bb_v1',
                             sampling='first configured complete matches / paired matches; independent of outcomes',
                             notes=['Recorded subset only; bucket means are descriptive, not counterfactual action EV.'])
        self.max_bytes, self.bytes, self.hands, self.dropped = max_bytes, 0, 0, 0

    def sink(self, **identity):
        def record(hand):
            row = {**hand, **identity, 'update': self.metadata['update'], 'scope': self.metadata['scope']}
            body = json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n'
            size = len(body.encode('utf-8'))
            if self.dropped or self.bytes + size > self.max_bytes:
                self.dropped += 1
                return
            self.stream.write(body)
            self.bytes += size
            self.hands += 1
        return record

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.stream.close()
        report = {**self.metadata, 'recorded_hands': self.hands, 'bytes': self.bytes,
                  'dropped_hands': self.dropped, 'truncated': bool(self.dropped),
                  'status': 'failed' if exc_type else 'complete'}
        temporary = self.directory / 'report.json.tmp'
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        temporary.replace(self.directory / 'report.json')


def trace_settings(settings):
    values = {'enabled': True, 'train_every_updates': 25, 'train_matches': 2,
              'eval_match_pairs_per_opponent': 1, 'max_bytes': 16 * 1024 * 1024,
              **settings.get('traces', {})}
    if not isinstance(values['enabled'], bool):
        raise ValueError('traces.enabled must be boolean')
    for key in ('train_every_updates', 'train_matches', 'eval_match_pairs_per_opponent', 'max_bytes'):
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < (1 if key == 'train_every_updates' else 0):
            raise ValueError(f'invalid traces.{key}')
    if values['max_bytes'] < 1024:
        raise ValueError('traces.max_bytes must be >= 1024')
    return values
