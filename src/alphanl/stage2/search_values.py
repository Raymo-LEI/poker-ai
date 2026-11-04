"""Versioned value inputs, exact small chance trees, and safe check-down leaves."""
import hashlib
import itertools
import json
import math
import numpy as np
from .search_math import (DECK, CARD_ID, card_text, hand_masks, rank,
                          DCFR, Terminal, Chance)
from .search_game import PublicState, betting_tree, utility_coefficients

SCHEMA = 'combo_value_v1'
VALUE_CONTRACT = dict(schema=SCHEMA, units='BB', baseline='root_centered',
                      players='zero_sum', scale=1., features='combo_board_state_ranges',
                      state_fields='actor,first,pending,street_put_bb,round_raise_bb,bb_chips',
                      buckets='none_combo_support')


def recognize_artifacts(weights, metadata=None):
    old = sorted(k for k in weights if k.startswith(('vleaf.', 'cfv_river.', 'cfv_turn.')) or k == 'preflop_freq')
    # This PR provides a generator, not a trained architecture/decoder. Even a
    # copied schema string cannot make arbitrary arrays into a trusted network.
    return dict(value='exact_or_checkdown', rejected_legacy=old,
                network='unavailable_no_trained_v1_decoder', schema=SCHEMA)


def inputs(state, board, hands, ranges, blind=100.):
    board = tuple(card_text(c) for c in board)
    expected = (0,3,4,5)[state.view.street]
    if len(board) != expected or len(set(board)) != len(board):
        raise ValueError('street/board schema mismatch')
    if blind <= 0:
        raise ValueError('invalid BB')
    result = dict(contract=dict(VALUE_CONTRACT), street=state.view.street,
                  board=[CARD_ID[c] for c in board], pot_bb=state.view.pot/blind,
                  stacks_bb=[v/blind for v in state.view.chips],
                  street_put_bb=[v/blind for v in state.view.street_put],
                  round_raise_bb=state.view.round_raise/blind,actor=state.actor,
                  first=state.first,pending=state.pending,bb_chips=blind,hands=[],ranges=[])
    for hs, ws in zip(hands,ranges):
        hand_masks(hs)
        if any(set(map(card_text,h)) & set(board) for h in hs):
            raise ValueError('board-blocked input hand')
        w = np.asarray(ws, dtype=float)
        if w.shape != (len(hs),) or (w < 0).any() or not np.isfinite(w).all() or w.sum() <= 0:
            raise ValueError('invalid value range')
        result['hands'].append([[CARD_ID[card_text(c)] for c in h] for h in hs])
        result['ranges'].append(w.tolist())
    if len(result['hands']) != 2:
        raise ValueError('two ranges required')
    return result


def flatten_input(row):
    """No hidden 128-column backbone placeholder. Variable support is explicit."""
    if row['contract'] != VALUE_CONTRACT:
        raise ValueError('incompatible value contract')
    # A small-model trainer must declare/freeze these support lists separately.
    return np.asarray([row['street'], row['pot_bb'],row['actor'],row['first'],row['pending'],
                       row['round_raise_bb'],row['bb_chips']] + row['stacks_bb'] + row['street_put_bb'] +
                      row['board'] + [c for side in row['hands'] for h in side for c in h] +
                      [w for side in row['ranges'] for w in side], dtype=np.float64)


def encode_outputs(bb_values):
    return np.asarray(bb_values, dtype=np.float64).copy()


def decode_outputs(encoded):
    return np.asarray(encoded, dtype=np.float64).copy()


def translate_leaf(value_bb, root, leaf, player, blind=100.):
    return np.asarray(value_bb) + (leaf.view.chips[player]-root.view.chips[player] +
                                   (leaf.view.pot-root.view.pot)/2.)/blind


def checkdown_equity(board, hands, samples=8, seed=17, exact=False):
    """Pair-conditioned legal runouts. Turn is always exhaustively enumerated.

    Flop/preflop use IID uniform runouts from each actual four-card exclusion
    set; no candidate hand is globally removed from another candidate's deck.
    """
    board = tuple(card_text(c) for c in board)
    masks = tuple(hand_masks(h) for h in hands)
    valid = (masks[0][:,None] & masks[1][None,:]) == 0
    eq = np.zeros(valid.shape)
    if samples < 1:
        raise ValueError("positive runout sample count required")
    missing = 5-len(board)
    for i,h in enumerate(hands[0]):
        for j,o in enumerate(hands[1]):
            if not valid[i,j]:
                continue
            pair=tuple(sorted((tuple(sorted(h)),tuple(sorted(o)))))
            digest=hashlib.sha256(repr((board,pair)).encode()).digest()
            rng=np.random.RandomState((int(seed)^int.from_bytes(digest[:4],'little')) & 0xffffffff)
            dead = set(board) | set(h) | set(o)
            deck = [c for c in DECK if c not in dead]
            if exact or missing <= 1:
                runouts = itertools.combinations(deck, missing)
            else:
                runouts = [tuple(rng.choice(deck, missing, replace=False)) for _ in range(samples)]
            total, n = 0., 0
            for runout in runouts:
                b = board+tuple(runout)
                left, right = rank(tuple(h)+b), rank(tuple(o)+b)
                total += (left > right) + .5*(left == right)
                n += 1
            eq[i,j] = total/n
    return eq, valid


def full_turn_tree(root, board, hands, blind=100., raise_cap=1):
    """Exact chance plus river betting on the supplied finite combo support.

    Public river cards are observed BEFORE river decisions. Each pair has 44
    legal rivers, so probability is valid_pair_and_card/44, not 1/48.
    All-in branches enumerate runouts but have no later decision nodes.
    """
    if root.view.street != 2 or len(board) != 4:
        raise ValueError('full turn tree requires a turn state')
    board = tuple(map(card_text,board))
    masks = tuple(hand_masks(h) for h in hands)
    valid = (masks[0][:,None] & masks[1][None,:]) == 0
    def finish(leaf, history):
        a,b = utility_coefficients(root,leaf,blind)
        if leaf.terminal == 'fold':
            return Terminal(np.full(valid.shape,b))
        branches = []
        for card in DECK:
            if card in board:
                continue
            legal = np.array([card not in h for h in hands[0]])[:,None] & np.array([card not in h for h in hands[1]])[None,:]
            prob = legal.astype(float)/44.
            rb = board+(card,)
            eq = np.zeros(valid.shape)
            # Blocked rows are zero-reach; do not evaluate duplicate cards.
            for i,h in enumerate(hands[0]):
                for j,o in enumerate(hands[1]):
                    if legal[i,j] and valid[i,j]:
                        x,y=rank(tuple(h)+rb),rank(tuple(o)+rb)
                        eq[i,j]=(x>y)+.5*(x==y)
            if leaf.terminal == 'showdown':
                child = Terminal(a*eq+b)
            else:
                river = PublicState(leaf.view.pot,leaf.view.chips,street=3,actor=leaf.first,first=leaf.first)
                def river_finish(end, rh, eq=eq):
                    aa,bb=utility_coefficients(root,end,blind)
                    return Terminal(aa*eq+bb)
                child = betting_tree(river, river_finish, raise_cap=raise_cap,
                                     history=history+(('river',card),))
            branches.append((prob,child))
        return Chance(branches)
    return betting_tree(root,finish,raise_cap=raise_cap)


def solve_label(state, board, hands, ranges, iterations=100, full_turn=True, blind=100.):
    row = inputs(state,board,hands,ranges,blind)
    eq,valid=checkdown_equity(board,hands)
    joint = np.asarray(ranges[0])[:,None]*np.asarray(ranges[1])[None,:]*valid
    if state.view.street == 2 and full_turn:
        tree = full_turn_tree(state,board,hands,blind)
        continuation='enumerated_river_betting'
    else:
        from .search_game import build_tree
        tree=build_tree(state,blind,equity=eq,raise_cap=2)
        continuation='river_terminal' if state.view.street == 3 else 'sampled_checkdown'
    result=DCFR(tree,joint).solve(iterations)
    row.update(targets=[[float(x) if np.isfinite(x) else None for x in encode_outputs(v)] for v in result['values']],
               reachable=[m.tolist() for m in result['reachable']],
               zero_sum_error=result['zero_sum_error'],iterations=iterations,
               continuation=continuation, production_training=False)
    return row, result
