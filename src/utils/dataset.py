import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast
from sentence_transformers import SentenceTransformer

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')

class Banking77Dataset(Dataset):
    """
    A class to represent a Banking77 dataset.

    Attributes
    ----------
    tokenizer
        The tokenizer to use.
    max_len
        The maximum length of the input sequence.
    data
        The data.
    labels
        The labels.

    Methods
    -------
    __len__()
        Returns the length of the dataset.
    __getitem__(idx)
        Returns the item at the given index.
    """
    def __init__(self, data, labels,  max_len=128, tokenizer=tokenizer):
        """
        Parameters
        ----------
        tokenizer
            The tokenizer to use.
        max_len
            The maximum length of the input sequence.
        data
            The data.
        labels
            The labels.
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = data
        self.labels = labels

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns the item at the given index.

        Parameters
        ----------
        idx
            The index.

        Returns
        -------
        dict
            The item at the given index.
        """
        text = str(self.data[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label)
        }
