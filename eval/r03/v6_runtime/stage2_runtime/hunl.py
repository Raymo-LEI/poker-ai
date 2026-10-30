from typing import Sequence, Tuple
import numpy as np
from . import arraytorch as torch
from .tokenizer import BettingHistoryTokenizer
from .schema import ActionEvent
NUMERIC_VERSION="hunl_exact_fraction_v1"
class HunlTokenizer(BettingHistoryTokenizer):
    """Same tokens and four channels, retaining ratios above 10x pot.

    Legacy PPO tokenizers still clip their first action channel to 1. This
    explicit CFR format divides the true fraction by 10 without clipping.
    Optional equity/odds/margin use the first hole-card token's numeric channels
    without changing tensor shapes. Checkpoints record numeric_version.
    """

    numeric_version = NUMERIC_VERSION

    def __init__(self, max_seq_len=64, *, position_conditioning=True,
                 equity_features=False, equity_samples=32, equity_seed=17):
        super().__init__(max_seq_len)
        self.position_conditioning = position_conditioning
        self.equity_features = equity_features
        self.equity_samples = equity_samples
        self.equity_seed = equity_seed
        if equity_features:
            from .equity import equity_numeric_version
            self.numeric_version = equity_numeric_version()

    def _tokens(self, snapshot):
        tokens = super()._tokens(snapshot)
        if self.position_conditioning:
            if snapshot.me_seat not in (0, 1) or snapshot.sb_seat != 1:
                raise ValueError("HUNL position encoding requires seat 0=BB, seat 1=SB/button")
            token, kind, _ = tokens[1]
            # HOLE is never truncated. One-hot position even at an empty
            # preflop history; no opponent private information is added.
            tokens[1] = (token, kind, (float(snapshot.me_seat == 0), float(snapshot.me_seat == 1), 0., 0.))
        if self.equity_features:
            from .equity import equity_features
            equity, odds, margin = equity_features(snapshot, samples=self.equity_samples, seed=self.equity_seed)
            # This existing hole-card token is retained even at max_seq_len=8.
            # Its unused numeric channels share the existing numeric projection;
            # CLS chip context, position, tensor shapes and parameter keys survive.
            token, kind, _ = tokens[2]
            tokens[2] = (token, kind, (equity, odds, margin, 0.))
        return tokens

    def _action_numeric(self, event: ActionEvent, blind: float) -> Tuple[float, float, float, float]:
        legacy = super()._action_numeric(event, blind)
        return (float(event.pot_fraction) / 10.0, *legacy[1:])

    def _pad(self, sequences):
        # A DFS frame retains its observation while visiting descendants, so
        # outputs must own their storage. Pack entire sequences in NumPy,
        # avoiding a small Tensor allocation and four indexed writes/token.
        shape = (len(sequences), self.max_seq_len)
        arrays = {"token_ids": np.zeros(shape, dtype=np.int64),
                  "type_ids": np.zeros(shape, dtype=np.int64),
                  "numeric": np.zeros((*shape, self.num_numeric), dtype=np.float32),
                  "pad_mask": np.ones(shape, dtype=bool),
                  "legal_mask": np.zeros((len(sequences), self.num_actions), dtype=np.float32)}
        for row, sequence in enumerate(sequences):
            if not sequence:
                continue
            ids, types, numeric = zip(*sequence)
            count = len(sequence)
            arrays["token_ids"][row, :count] = ids
            arrays["type_ids"][row, :count] = types
            arrays["numeric"][row, :count] = numeric
            arrays["pad_mask"][row, :count] = False
        return {key: torch.from_numpy(value) for key, value in arrays.items()}