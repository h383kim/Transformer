import torch
from torch import nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        '''
        Args:
            :d_model (int): The embedding dimension.
            :max_len (int): The maximum length of the input sequences.

        Shape:
            - Input: (batch_size, seq_len, d_model)
            - Output: (batch_size, seq_len, d_model)
        '''
        super().__init__()
        # Create a matrix to hold the positional encodings
        pe = torch.zeros(max_len, d_model) # Shape: (max_len, d_model)
        # Position indices [0, 1, 2, ..., max_len-1]
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # Shape: (max_len, 1)
        # division term
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Even indices(2i) terms are Sine
        pe[:, 0::2] = torch.sin(pos * div_term)
        # Odd indices(2i+1) terms are Cosine 
        pe[:, 1::2] = torch.cos(pos * div_term)
        # Add a batch dimension at the first position
        pe = pe.unsqueeze(0) # Shape: (1, max_len, d_model)

        # Register pe as a buffer, not a parameter. It will not be updated during training.
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            :x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, d_model) with positional encodings added.
        """
        # Ensure the positional encoding is on the same device as the input
        pe = self.pe[:, x.size(1), :].to(x.device)
        # Add positional encoding to the input embeddings
        x = x + pe
        return x