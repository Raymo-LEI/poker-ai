"""Full retained reservoir staged for one player's training phase only."""

from __future__ import annotations

import numpy as np
import torch


class DeviceTrainingPool:
    """Read-only full-history copy, preserving all numerical features in FP32.

    uint16 token IDs use int32 on device because eager uint16 indexing has
    limited kernel support. IDs widen to long only for a drawn minibatch.
    This adds 128 bytes per length-64 sample (2M rows: 2.882 GB total).
    CPU is supported here for schema tests; the trainer opts in on CUDA.
    """

    def __init__(self, buffer, device, *, reserve_bytes=0, chunk_rows=16384):
        self.device = torch.device(device)
        self.size = len(buffer)
        if not self.size:
            raise ValueError("cannot stage an empty reservoir")
        dtypes = {key: torch.int32 if key == "token_ids" else torch.from_numpy(np.empty((), dtype=dtype)).dtype
                  for key, (dtype, _) in buffer._schema.items()}
        self.nbytes = sum(self.size * int(np.prod(tail)) * torch.empty((), dtype=dtypes[key]).element_size()
                          for key, (_, tail) in buffer._schema.items())
        if self.device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(self.device)
            if self.nbytes + reserve_bytes > free:
                raise RuntimeError(f"full training pool needs {self.nbytes} bytes plus {reserve_bytes} reserve; CUDA free={free}")
        self.arrays = {key: torch.empty((self.size,) + tail, dtype=dtypes[key], device=self.device)
                       for key, (_, tail) in buffer._schema.items()}
        # Read contiguous bounded prefixes, without duplicating a full 2M
        # RAM pool or performing a large random gather during transfer.
        if buffer.directory is not None:
            chunks = ((start, {key: value[start:min(start + chunk_rows, self.size)]
                               for key, value in buffer._maps.items()})
                      for start in range(0, self.size, chunk_rows))
        else:
            chunks = ((i * buffer.chunk_size, chunk) for i, chunk in enumerate(buffer._chunks))
        for start, chunk in chunks:
            count = min(len(chunk["iterations"]), self.size - start)
            for key, value in chunk.items():
                array = np.asarray(value[:count])
                if key == "token_ids":
                    array = array.astype(np.int32)
                self.arrays[key][start:start + count].copy_(torch.from_numpy(array))

    def sample(self, batch_size, generator):
        indexes = torch.randint(self.size, (batch_size,), device=self.device, generator=generator)
        selected = {key: value.index_select(0, indexes) for key, value in self.arrays.items()}
        batch = {key: selected[key] for key in
                 ("token_ids", "type_ids", "numeric", "pad_mask", "legal_mask")}
        batch["token_ids"] = batch["token_ids"].long()
        batch["type_ids"] = batch["type_ids"].long()
        batch["legal_mask"] = batch["legal_mask"].float()
        return batch, selected["regrets"], selected["iterations"]

    def close(self):
        self.arrays.clear()
