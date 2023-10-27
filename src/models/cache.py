import torch  # PyTorch deep learning library
from torch.utils.data import TensorDataset
import torch.nn.functional as F  # PyTorch functional interface

class Cache(torch.nn.Module):
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

    Methods
    -------
    to(device)
        Move the model and the database to the specified device.
    top_k(query)
        Get the top k nearest neighbors of the query.
    topk_w_centroid(query)
        Get the weighted centroid of the top k nearest neighbors of the query.
    is_near_wcentroid(query)
        Check if the query is near the weighted centroid of the top k nearest neighbors.
    add(query, label)
        Add the query and its label to the database.
    """
    def __init__(self, encodings:torch.Tensor = None, labels:list[int] = None, d_thresh: float = 0.5, k: int = 5) -> None:
        super(Cache, self).__init__()
        self.database = encodings if encodings is not None else torch.empty(0, 768)
        self.labels = labels.copy() if labels is not None else []
        self.d_thresh = d_thresh
        self.k = k
        return


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
        # Compute the Cosine distance between the query and the database vectors
        dist = 1 - F.cosine_similarity(query, self.database, dim=1)
        # Get the top k nearest neighbors
        d, top_k = torch.topk(dist, k=self.k, dim=0, largest=False)
        # Reshape the tensors to have the same shape
        top_k , d = top_k.unsqueeze(0), d.unsqueeze(0)
        return d, top_k
    

    def topk_w_centroid(self, query: str | torch.Tensor) -> torch.Tensor:
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


    def get_last_100_added(self) -> TensorDataset:
        """
        Get the last 100 vectors and labels added to the database.

        Returns
        -------
        TensorDataset
            A TensorDataset containing the last 100 vectors and labels added to the database.
        """
        return TensorDataset(self.database[-100:], torch.tensor(self.labels[-100:]))

    def is_near_wcentroid(self, query: str | torch.Tensor) -> bool:
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
        weighted_centroid = self.topk_w_centroid(query)
        # Calculate the distance between the query and the weighted centroid
        dist = 1 - F.cosine_similarity(query, weighted_centroid, dim=1)
        return torch.lt(dist, self.d_thresh)


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
        # Add the query to the database
        self.database = torch.cat((self.database, query))
        # Add the label to the labels list
        if isinstance(label, torch.Tensor):
            label = label.item()
            assert isinstance(label, int)
        self.labels.append(label)
        return

    
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

        # Add the training data to the database
        self.database = vectors
        self.labels = labels
        return

    def set_threshold(self, d_thresh: float) -> None:
        """
        Set the distance threshold.

        Parameters
        ----------
        d_thresh
            The distance threshold.
        """
        self.d_thresh = d_thresh
        return

    def __len__(self) -> int:
        """
        Get the number of vectors in the database.

        Returns
        -------
        int
            The number of vectors in the database.
        """
        return self.database.shape[0]
