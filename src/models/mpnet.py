import torch
import torch.nn as nn
import torch.nn.functional as F

class MPNetClassifier(nn.Module):
    """
    A class representing the MPNet encoder model with a classification head.
    
    Attributes
    ----------
    model
        The SentenceTransformer model.
    hidden_size : int
        The size of the hidden layer.
    dropout : float
        The dropout probability.
    activation : str
        The activation function to use.
    Methods
    -------
    forward(x)
        Forward pass of the model.
    """

    def __init__(self, hidden_size=256, dropout=0.1, activation="relu", num_classes=77):
        """
        Parameters
        ----------
        model_name : str
            The name of the model to use.
        hidden_size : int
            The size of the hidden layer.
        dropout : float
            The dropout probability.
        activation : str
            The activation function to use.
        """

        super().__init__()

        self.hidden_size = hidden_size
        self.dropout = dropout
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "silu":
            self.activation = nn.SiLU()
        elif activation == "hardswish":
            self.activation = nn.Hardswish()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError("Invalid activation function, must be one of: relu, tanh, silu, hardswish, gelu")

        self.classifier = nn.Sequential(
            nn.Linear(768, self.hidden_size),
            self.activation,
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, num_classes)
        )

    def forward(self, x):
        """
        Forward pass of the model.
        
        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        
        Returns
        -------
        torch.Tensor
            The output tensor.
        """

        logits = self.classifier(x)

        return logits

        def __repr__(self):
            return f"MPNetClassifier(hidden_size={self.hidden_size}, dropout={self.dropout}, activation={self.activation.__class__.__name__})"
        
        def __str__(self):
            return f"MPNetClassifier(hidden_size={self.hidden_size}, dropout={self.dropout}, activation={self.activation.__class__.__name__})"