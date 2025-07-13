import torch
import torch.nn.functional as F
from typing import Optional, List, override
from src.caches.base import BaseCache, register_class

@register_class("lru")
class LRUCache(BaseCache):
    """
    Least‑Recently‑Used cache with duplicate‑skip.

    • Keeps at most `capacity` vectors.
    • When the cache is full, evicts the row whose timestamp is smallest.
    • Skips inserting queries that are `is_near()` an existing entry
      (regardless of whether the cache is full).
    """

    def __init__(
        self,
        encodings: Optional[torch.Tensor] = None,
        labels: Optional[List[int]] = None,
        d_thresh: float = 0.5,
        k: int = 5,
        capacity: int = 100,
    ) -> None:
        if capacity <= 0:
            raise ValueError("Cache capacity must be a positive integer")
        super().__init__(encodings=encodings, labels=labels,
                         d_thresh=d_thresh, k=k)
        self.capacity   = capacity
        init_len        = 0 if encodings is None else encodings.size(0)
        self._last_used = torch.arange(init_len)          # timestamps
        self._counter   = init_len                        # monotonic clock

    def _topk_static(self, query: torch.Tensor):
        dist, idx = torch.topk(
            1 - F.cosine_similarity(query, self.database, dim=1),
            k=self.k,
            largest=False,
        )
        return dist, idx          # 1‑D idx

    @override
    def top_k(self, query: str | torch.Tensor):
        dist, idx = self._topk_static(query)
        self._counter += 1
        self._last_used[idx] = self._counter
        return dist.unsqueeze(0), idx.unsqueeze(0)

    def _topk_w_centroid(self, query: str | torch.Tensor):
        d, idx = self._topk_static(query)     # NO timestamp bump
        w = 1 / (d + 1e-6) ** 2
        w = w / torch.sum(w)
        return (w @ self.database[idx]).squeeze(0)

    @override
    def is_near(self, query: str | torch.Tensor) -> bool:
        c = self._topk_w_centroid(query)
        dist = 1 - F.cosine_similarity(query, c, dim=1)
        return torch.lt(dist, self.d_thresh)

    @override
    def add(self, query: torch.Tensor, label: int | torch.Tensor) -> None:
        # --- duplicate‑skip FIRST (even if not full) ---
        if self.database.numel() > 0 and self.is_near(query):
            return                              # skip near‑duplicate

        if isinstance(label, torch.Tensor):
            label = label.item()

        if self.database.numel() == 0:
            self.database   = query.clone().detach()
            self.labels     = [label]
            self._last_used = torch.tensor([self._counter])
            self._counter  += 1
            return

        if len(self) < self.capacity:
            self.database   = torch.cat((self.database, query))
            self.labels.append(label)
            self._last_used = torch.cat((self._last_used,
                                         torch.tensor([self._counter])))
            self._counter  += 1
            return

        evict_idx = torch.argmin(self._last_used).item()
        self.database[evict_idx]   = query
        self.labels[evict_idx]     = label
        self._last_used[evict_idx] = self._counter
        self._counter += 1

    @override
    def fit(self, vectors: torch.Tensor, labels: list[int]) -> None:
        if len(vectors) != len(labels):
            raise ValueError("vectors and labels must have equal length")
        self.database   = vectors.clone().detach()
        self.labels     = [int(l) for l in labels]
        self._last_used = torch.arange(len(labels))
        self._counter   = len(labels)
