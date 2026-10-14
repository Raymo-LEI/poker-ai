"""Torch batches of the AlphaNL conv observation. The numpy layout lives in ``nl_obs``."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from alphanl.nl_obs import encode_observation
from .tokenizer import TensorBatch


def arrays_to_batch(arrays: dict) -> TensorBatch:
    return {
        key: torch.from_numpy(np.ascontiguousarray(value)).unsqueeze(0)
        for key, value in arrays.items()
    }


def encode_table_conv(table: Any, me_seat: int) -> dict:
    """Numpy conv tensors for a decision at ``me_seat`` (must be the actor)."""

    seat = int(me_seat)
    if int(table.current_player()) != seat:
        raise RuntimeError(
            "conv encoding requires the acting seat, got %s vs %s" % (seat, table.current_player())
        )
    from .tables import PokerkitTable, RlcardTable

    if isinstance(table, RlcardTable):
        return encode_observation(table.state["raw_obs"], table.history, seat)
    if isinstance(table, PokerkitTable):
        return encode_observation(table.rlcard_raw_obs(seat), table.wrapper_history(), seat)
    raise TypeError("unsupported table type %r" % type(table))


class AlphaNLObserver:
    """Build the control-group batch from a self-play table or rlcard wrapper fields."""

    def observe_table(self, table: Any, player_id: int) -> TensorBatch:
        return arrays_to_batch(encode_table_conv(table, int(player_id)))

    def observe(self, raw_obs: dict, history: Sequence[Sequence[Any]], player_id: int) -> TensorBatch:
        return arrays_to_batch(encode_observation(raw_obs, history, player_id))

    def blank(self) -> TensorBatch:
        batch = {
            "card_info": torch.zeros(1, 4, 13, 6, dtype=torch.float32),
            "action_info": torch.zeros(1, 4, 5, 25, dtype=torch.float32),
            "extra_info": torch.zeros(1, 2, dtype=torch.float32),
            "legal_mask": torch.zeros(1, 5, dtype=torch.float32),
        }
        batch["legal_mask"][0, 1] = 1.0
        return batch
