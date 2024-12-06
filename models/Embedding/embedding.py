import torch
from torch import nn

from models.Embedding.positional_encoding import PositionalEncoding


"""
TransformerEmbedding combines token embeddings with positional encodings
to produce input embeddings for a Transformer model. It also applies dropout
for regularization.

Args:
    vocab_size (int): Size of the vocabulary.
    d_model (int): Dimensionality of the embeddings.
    max_len (int): Maximum sequence length for positional encoding.
    p_dropout (float): Dropout probability.
"""
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, p_dropout):
        super().__init__()
        
        """
        Token embedding layer maps input token indices to dense vectors.
        Args:
            padding_idx: Index for padding token, ensuring its embedding remains zero.
        """
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=1)
        """
        Positional encoding layer adds positional information to token embeddings.
        """
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        """
        Dropout
        """
        self.emb_dropout = nn.Dropout(p=p_dropout)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of token indices with shape (batch_size, seq_len).
        Returns:
            Tensor: Output tensor of embeddings with shape (batch_size, seq_len, d_model).
        """
        token_embedded = self.token_embedding(x)
        positon_encoded = self.postional_encoding(token_embedded)
        
        return self.emb_dropout(position_encoded)