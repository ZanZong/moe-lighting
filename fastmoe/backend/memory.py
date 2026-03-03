"""Memory pool."""
import logging

import torch

logger = logging.getLogger(__name__)

class TokenToKVPool:
    def __init__(self, gpu_size, cpu_size, dtype, head_num, head_dim, layer_num):
        self.cur_start_loc = 0
        self.cur_cl = 0
        self.dtype = dtype
        self.cache_line = gpu_size // 2
        self.head_num = head_num
        self.head_dim = head_dim

        # round cpu_size to be a multiple of (gpu_size // 2)
        self.cpu_size = (cpu_size // self.cache_line) * self.cache_line

        # Per-session KV snapshot dict: {(session_id, layer): tensor}
        # Populated by save_session(), consumed by load_session().
        self._session_kv: dict = {}

        # [size, key/value, head_num, head_dim]
        # This serves as a direct mapped cache on gpu for prefill stage
        self.kv_data = torch.empty((2 * self.cache_line, 2, head_num, head_dim), dtype=dtype, device="cuda")
        

        self.kv_data_cpu = [
            torch.empty((self.cpu_size, 2, head_num, head_dim), dtype=dtype, device="cpu")
            for _ in range(layer_num)
        ]
    
    def store(self, dst_start, size, layer):
        src_start = dst_start % (2 * self.cache_line)
        dst_end = dst_start + size
        src_end = src_start + size
        self.kv_data_cpu[layer][dst_start:dst_end, :, :, :].copy_(self.kv_data[src_start:src_end, :, :, :], non_blocking=True)
    
    def load(self, src_start, size, layer):
        dst_start = src_start % (2 * self.cache_line)
        src_end = src_start + size
        dst_end = dst_start + size
        self.kv_data[dst_start:dst_end, :, :, :].copy_(self.kv_data_cpu[layer][src_start:src_end, :, :, :], non_blocking=True)

    def get_key_buffer(self):
        return self.kv_data[:, 0, :, :]

    def get_value_buffer(self):
        return self.kv_data[:, 1, :, :]
    
    def get_key_buffer_cpu(self, layer):
        return self.kv_data_cpu[layer][:, 0, :, :]
    
    def get_value_buffer_cpu(self, layer):
        return self.kv_data_cpu[layer][:, 1, :, :]
    
    def alloc_cpu(self):
        start_loc = self.cur_start_loc
        self.cur_start_loc = self.cur_start_loc + self.cache_line
        cl_idx = self.cur_cl
        self.cur_cl = (self.cur_cl + 1) % 2
        return start_loc, cl_idx * self.cache_line, cl_idx

    def save_session(self, session_id: str, start_loc: int, num_tokens: int, layer: int) -> None:
        """Persist the GPU KV cache for a session to a dedicated CPU buffer.

        Unlike the regular :meth:`store` path which writes into the shared
        rolling CPU pool, ``save_session`` copies KV data into a
        *session-dedicated* CPU tensor so that it survives across batch
        boundaries.  The caller is responsible for ensuring that the GPU
        buffer at ``start_loc`` contains up-to-date data for ``num_tokens``
        tokens (i.e., :meth:`store` must have been called first).

        Parameters
        ----------
        session_id: Unique identifier for the conversation session.
        start_loc:  Start token index in the CPU pool where KV data lives.
        num_tokens: Number of consecutive tokens to snapshot.
        layer:      Layer index.
        """
        key = (session_id, layer)
        # Snapshot: clone so later pool modifications don't corrupt the cache
        self._session_kv[key] = self.kv_data_cpu[layer][
            start_loc: start_loc + num_tokens, :, :, :
        ].clone()

    def load_session(self, session_id: str, dst_start: int, num_tokens: int, layer: int) -> bool:
        """Load a session's saved KV back into the shared CPU pool at *dst_start*.

        Returns True on a cache hit, False if the session has no saved KV for
        this layer (caller must then compute the KV from scratch).
        """
        key = (session_id, layer)
        if key not in self._session_kv:
            return False
        saved = self._session_kv[key]
        actual = min(num_tokens, saved.shape[0])
        self.kv_data_cpu[layer][dst_start: dst_start + actual].copy_(
            saved[:actual], non_blocking=True
        )
        return True

    def has_session(self, session_id: str, layer: int = 0) -> bool:
        """Return True if session KV data is stored for the given layer."""
        return (session_id, layer) in self._session_kv

    def evict_session(self, session_id: str) -> None:
        """Remove all saved KV tensors for *session_id* from CPU memory."""
        keys_to_remove = [k for k in self._session_kv if k[0] == session_id]
        for k in keys_to_remove:
            del self._session_kv[k]

    def delete_gpu_cache(self):
        self.kv_data = None

    def clear(self):
        self.cur_start_loc = 0
        self.cur_cl = 0
        self._session_kv.clear()
