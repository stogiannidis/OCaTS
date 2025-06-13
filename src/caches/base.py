import torch
from torch.utils.data import TensorDataset

from typing import Optional, List, Type
from abc import ABC, abstractmethod


class BaseCache(torch.nn.Module, ABC):
    """
    A base class for a cache that stores the vectors and labels of the training data

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

    Methods
    -------
    to(device)
        Move the model and the database to the specified device.
    top_k(query)
        Get the top k nearest neighbors of the query.
    is_near(query)
        Check if the query is near the top k nearest neighbors.
    add(query, label)
        Add the query and its label to the database.
    get_last_p_added(p)
        Get the last p vectors and labels added to the database.
    set_threshold(d_thresh)
        Set the distance threshold.
    __len__() 
        Get the number of vectors in the database.
    fit(vectors, labels)
        Fit the model to the training data.
    """
    def __init__(
            self,
            encodings: Optional[torch.Tensor] = None,
            labels: Optional[List[int]] = None,
            d_thresh: float = 0.5,
            k: int = 5
        ) -> None:
        super(BaseCache, self).__init__()
        self.database = encodings if encodings is not None else torch.empty(0, 768)
        self.labels = labels.copy() if labels is not None else []
        self.d_thresh = d_thresh
        self.k = k


    def get_last_p_added(self, p: int = 100) -> TensorDataset:
        """
        Get the last p vectors and labels added to the database.

        Returns
        -------
        TensorDataset
            A TensorDataset containing the last p vectors and labels added to the database.
        """
        return TensorDataset(self.database[-p:], torch.tensor(self.labels[-p:]))
    

    def set_threshold(self, d_thresh: float) -> None:
        """
        Set the distance threshold.

        Parameters
        ----------
        d_thresh
            The distance threshold.
        """
        self.d_thresh = d_thresh


    def __len__(self) -> int:
        """
        Get the number of vectors in the database.

        Returns
        -------
        int
            The number of vectors in the database.
        """
        return self.database.shape[0]
    

    @abstractmethod
    def top_k(self, query: str | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the top k nearest neighbors of the query.

        Parameters
        ----------
        query
            The query vector or text.

        Returns
        -------
        tuple
            A tuple containing the top k nearest neighbors and their distances.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    

    @abstractmethod
    def is_near(self, query: str | torch.Tensor) -> bool:
        """
        Check if the query is near the top k nearest neighbors by a custom criteria.
        
        Parameters
        ----------
        query
            The query vector or text.

        Returns
        -------
        bool
            True if the query is near the top k nearest neighbors based on custom criteria, False otherwise.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")


    @abstractmethod
    def add(self, query: str, label: int | torch.Tensor)-> None:
        """
        Add the query and its label to the database.

        Parameters
        ----------
        query
            The query vector or text.
        label
            The label of the query.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")


    @abstractmethod
    def fit(self, vectors: torch.Tensor, labels: list[int]) -> None:
        """ 
        Fit the model to the training data.

        Parameters
        ----------
        corpus
            The training data vectors.
        labels
            The labels of the training data.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")


class CacheRegistry:
    def __init__(self):
        self._registry = {}

    def register(self, name: str, cls: Type[BaseCache]) -> None:
        self._registry[name] = cls

    def __getitem__(self, name: str) -> Type[BaseCache]:
        if name not in self._registry:
            raise KeyError(f"Cache class '{name}' is not registered.")
        return self._registry[name]

# Global registry
CACHE_REGISTRY = CacheRegistry()

# Decorator to register classes
def register_class(name):
    def decorator(cls):
        print(f"Registering cache class: {name}")
        CACHE_REGISTRY.register(name, cls)
        print(CACHE_REGISTRY._registry)
        return cls
    return decorator
