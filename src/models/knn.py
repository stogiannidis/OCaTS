import torch  # PyTorch deep learning library
import torch.nn.functional as F  # PyTorch functional interface
from src.caches.cache import SimpleCache

class KNNClassifier():
    """
    A simple k-nearest neighbors classifier.

    Parameters
    ----------
    cache
        The cache to use for the k-nearest neighbors search.
    k
        The number of nearest neighbors to consider for the k-nearest neighbors search.

    Methods
    -------
    predict(query)
        Predict the label of the query.
    predict_proba(query)
        Predict the probability distribution over all the labels for the query.
    forward(query)
        Predict the probability distribution over all the labels for the query.
    """

    def __init__(self, cache: SimpleCache):
        super().__init__()
        self.cache = cache
        self.len_unique_labels = len(set(self.cache.labels))

    def predict_proba(self, query: str | torch.Tensor) -> tuple:
        """
        Predict the probability distribution over all the labels for the query.

        Parameters
        ----------
        query
            The query vector or text.

        Returns
        -------
        tuple
            A tuple containing the unique labels and their probabilities.
        """
        d , top_k = self.cache.top_k(query)
        # Find the top k labels
        labels = torch.tensor(self.cache.labels).to(d.device)[top_k[0]]
        # multiply the counts by the weights
        weights = 1 / (d+1e-10)**2
        # Calculate the unique labels and their counts
        sums = torch.bincount(labels, weights=weights[0], minlength=self.len_unique_labels)
        # Calculate the probabilities
        probabilities = sums / torch.sum(sums)
        return probabilities

    def predict(self, query: str | torch.Tensor) -> int:
        """
        Predict the label of the query.

        Parameters
        ----------
        query
            The query vector or text.

        Returns
        -------
        int
            The predicted label.
        """
        probabilities = self.predict_proba(query)
        return probabilities.argmax().item()
    def __call__(self, query: str | torch.Tensor) -> torch.Tensor:
        """
        Predict the probability distribution over all the labels for the query.

        Parameters
        ----------
        query
            The query vector or text.

        Returns
        -------
        torch.Tensor
            A tensor containing the probability distribution over all the labels.
        """
        distribution = self.predict_proba(query)
        # Normalize the distribution eliminating the 0s but keeping the shape
        distribution[distribution == 0] = 1e-10
        return distribution.view(1, -1)

    def __len__(self) -> int:
        """
        Get the number of vectors in the database.

        Returns
        -------
        int
            The number of vectors in the database.
        """
        return len(self.cache)