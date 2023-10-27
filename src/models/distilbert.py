import torch  # PyTorch deep learning library
import torch.nn.functional as F  # PyTorch functional interface
from sklearn.metrics import accuracy_score as acc  # For computing accuracy score
from tqdm.auto import tqdm  # For progress bar
from transformers import DistilBertModel
from transformers.modeling_outputs import TokenClassifierOutput

class DBertClassifier(torch.nn.Module):
    """
    A class to represent a DistilBERT classifier.
    
    Attributes
    ----------
    l1
        The DistilBERT model.
    pre_classifier
        The pre-classifier layer.
    dropout
        The dropout layer.
    classifier
        The classifier layer.

    Methods
    -------
    forward(input_ids, attention_mask)
        Forward pass.
    """

    def __init__(self, model_name='distilbert-base-uncased', dropout=0.3, hidden_size=768):
        """
        Parameters
        ----------
        model
            The DistilBERT model.
        optimizer
            The optimizer to use for training.
        scheduler
            The scheduler to use for training.
        """
        super(DBertClassifier, self).__init__()
        self.l1 = DistilBertModel.from_pretrained(model_name)
        self.pre_classifier = torch.nn.Linear(768, hidden_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(hidden_size, 77)
        self.softmax = torch.nn.Softmax(dim=1)
        self.silu = torch.nn.SiLU()

    def forward(self, input_ids, attention_mask):
        """
        Parameters
        ----------
        params
            The parameters in the following order:
            input_ids
                The input IDs.
            attention_mask
                The attention mask.

        Returns
        -------
        Tensor
            The output logits.
        """
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:,0,:].view(-1,768)
        pooler = self.pre_classifier(pooler)
        pooler = self.silu(pooler)
        pooler = self.dropout(pooler)
        logits = self.classifier(pooler)

        return logits

    def freeze_body(self, use_grad=False):
        # Set requires_grad=True only for the linear layers
        for param in self.l1.parameters():
            param.requires_grad = use_grad
        for param in self.pre_classifier.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True
        

