import torch  # PyTorch deep learning library
from torch.utils.data import TensorDataset
import torch.nn.functional as F  # PyTorch functional interface

from src.caches.base import BaseCache, register_class
from typing import override, Optional, List


@register_class("lfu")
class LFUCache(BaseCache):
    """
    LFU cache with redundancy-aware eviction.
    If capacity is reached, evicts the least-frequently used vector; ties are
    broken by evicting the vector that is most similar to any other vector.

    Parameters
    ----------
    capacity
        The maximum size of the cache.
    encodings
        The encodings of the training data.
    labels
        The labels of the training data.
    d_thresh
        The distance threshold for the smart cache.
    k
        The number of nearest neighbors to consider for the smart cache.
    """
    def __init__(
        self,
        encodings: Optional[torch.Tensor] = None,
        labels: Optional[List[int]] = None,
        d_thresh: float = 0.5,
        k: int = 5,
        capacity: int = 100,
    ) -> None:
        super().__init__(encodings=None, labels=None, d_thresh=d_thresh, k=k)
        self._capacity = capacity
        self._freq = torch.ones(len(self.database), dtype=torch.long)
        if encodings is not None and labels is not None:
            self.fit(encodings, labels)


    @override
    def top_k(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Compute the Cosine distance between the query and the database vectors
        dist = 1 - F.cosine_similarity(query.to(self.database.device), self.database, dim=1)
        # Get the top k nearest neighbors
        d, top_k = torch.topk(dist, k=min(self.k, len(dist)), dim=0, largest=False)
        self._freq[top_k] += 1  # LFU update
        # Reshape the tensors to have the same shape
        top_k , d = top_k.unsqueeze(0), d.unsqueeze(0)
        return d, top_k
    

    def _topk_w_centroid(self, query: str | torch.Tensor) -> torch.Tensor:
        """
        Get the weighted centroid of the top k nearest neighbors of the query.

        Parameters
        ----------
        query
            The query vector or text.

        Returns
        -------
        torch.Tensor
            The weighted centroid of the top k nearest neighbors of the query.
        """
        # Get the top k nearest neighbors
        d, top_k = self.top_k(query)
        # Add weights to the top k nearest neighbors
        weights = 1 / (d + 1e-6) ** 2
        # normalize the weights
        weights = weights / torch.sum(weights)
        # Compute the weighted centroid by multiplying the weights with the top k nearest neighbors
        weighted_centroid = weights @ self.database[top_k]
        return weighted_centroid.squeeze(0)
    

    @override
    def is_near(self, query: torch.Tensor) -> bool:
        """
        Check if the query is near the weighted centroid of the top k nearest neighbors.
        
        Parameters
        ----------
        query
            The query vector or text.

        Returns
        -------
        bool
            True if the query is near the weighted centroid of the top k nearest neighbors based on the
            distance threshold, False otherwise.
        """
        if len(self) == 0:
            return False
        # Calculate the weighted centroid
        weighted_centroid = self._topk_w_centroid(query)
        # Calculate the distance between the query and the weighted centroid
        dist = 1 - F.cosine_similarity(query.to(device=weighted_centroid.device), weighted_centroid, dim=1)
        return torch.lt(dist, self.d_thresh)


    @override
    def add(self, query: torch.Tensor, label: int | torch.Tensor) -> None:
        if isinstance(label, torch.Tensor):
            label = label.item()
            assert isinstance(label, int)

        # If similar enough to existing vectors and we want deduplication,
        # simply skip insertion.
        if self.is_near(query):
            return

        if len(self) < self._capacity:
            # append
            self.database = torch.cat((self.database, query.unsqueeze(0).to(self.database.device)))
            self.labels.append(label)
            self._freq = torch.cat([self._freq, torch.ones(1, dtype=torch.long, device=self._freq.device)])
            return

        # --------  Eviction path  -------- #
        # 1. least frequency
        f_min = torch.min(self._freq)
        candidate_idx = torch.nonzero(self._freq == f_min, as_tuple=False).squeeze(1)

        # 2. redundancy score: similarity to *nearest* other vector
        #    (we use 1 - cosine similarity as distance)
        sim_to_rest = []
        for i in candidate_idx.tolist():
            # similarity to all others except itself
            candidate_v =  self.database[i]
            sim = F.cosine_similarity(candidate_v.unsqueeze(0), self.database, dim=1)
            sim[i] = -1.0  # ignore self-similarity
            sim_to_rest.append(torch.max(sim))
        sim_to_rest = torch.stack(sim_to_rest)
        # pick the most redundant (highest similarity)
        evict_pos = candidate_idx[torch.argmax(sim_to_rest)]

        # 3. in-place replacement keeps tensor contiguous
        self.database[evict_pos] = query
        self.labels[evict_pos] = label
        self._freq[evict_pos] = 1  # reset frequency

    
    @override
    def fit(self, vectors: torch.Tensor, labels: list[int]) -> None:
        for v, l in zip(vectors, labels):
            self.add(v, l)
