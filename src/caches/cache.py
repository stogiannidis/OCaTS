import torch  # PyTorch deep learning library
from torch.utils.data import TensorDataset
import torch.nn.functional as F  # PyTorch functional interface

from src.caches.base import BaseCache
from typing import override, Optional, List


class SimpleCache(BaseCache):
    """
    A simple cache that stores the vectors and labels of the training data

    Parameters
    ----------
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
        k: int = 5
    ) -> None:
        super().__init__(encodings=encodings, labels=labels, d_thresh=d_thresh, k=k)


    @override
    def top_k(self, query: str | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Compute the Cosine distance between the query and the database vectors
        dist = 1 - F.cosine_similarity(query, self.database, dim=1)
        # Get the top k nearest neighbors
        d, top_k = torch.topk(dist, k=self.k, dim=0, largest=False)
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
    def is_near(self, query: str | torch.Tensor) -> bool:
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
        # Calculate the weighted centroid
        weighted_centroid = self._topk_w_centroid(query)
        # Calculate the distance between the query and the weighted centroid
        dist = 1 - F.cosine_similarity(query, weighted_centroid, dim=1)
        return torch.lt(dist, self.d_thresh)


    @override
    def add(self, query: str, label: int | torch.Tensor) -> None:
        # Add the query to the database
        self.database = torch.cat((self.database, query))
        # Add the label to the labels list
        if isinstance(label, torch.Tensor):
            label = label.item()
            assert isinstance(label, int)
        self.labels.append(label)

    
    @override
    def fit(self, vectors: torch.Tensor, labels: list[int]) -> None:
        self.database = vectors
        self.labels = labels
