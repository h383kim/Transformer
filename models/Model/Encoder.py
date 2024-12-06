import torch
from torch import nn

from models.Embedding.embedding import TransformerEmbedding
from models.Layers.feed_forward_network import FFN
from models.Layers.multi_head_attention import MultiHeadAttention


"""
Single block in the Transformer encoder.

Each block consists of:
    - A multi-head self-attention mechanism with residual connections and LayerNorm.
    - A feed-forward network (FFN) with residual connections and LayerNorm.
    
Args:
    d_model (int): Dimensionality of the input embeddings.
    d_ffn (int): Dimensionality of the hidden layer in the feed-forward network.
    num_heads (int): Number of attention heads in the multi-head self-attention mechanism.
    p_dropout (float): Dropout probability for regularization.
"""
class EncoderBlock(nn.Module):
    def __init__(self, d_model, d_ffn, num_heads, p_dropout):
        super().__init__()
        # Self-attention
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=p_dropout)
        # Feed-Forward Network
        self.ffn = FFN(d_model, d_ffn, p_dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=p_dropout)
        
    def forward(self, x, src_mask):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            src_mask (torch.Tensor): Mask tensor to prevent attention to specific positions.
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
            torch.Tensor: Attention scores from the self-attention mechanism.
        """
        # Self-attention and Skip_connect
        attn_context, attn_score = self.self_attention(x, x, x, mask=src_mask)
        attn_context = nn.dropout1(attn_context)
        x = self.norm1(x + attn_context)
        # Feed_forward and Skip-connect
        residual = self.ffn(x)
        residual = self.dropout2(residual)
        x = self.norm2(x + residual)

        return x, attn_score



"""
Represents the full Transformer encoder consisting of multiple encoder blocks.

The encoder applies:
    - Positional encoding to the input embeddings.
    - Multiple encoder blocks, each with self-attention and feed-forward layers.

Args:
    vocab_size (int): Size of the vocabulary.
    max_len (int): Maximum sequence length.
    num_blocks (int): Number of encoder blocks in the encoder.
"""
class Encoder(nn.Module):
    def __init__(self, vocab_size, max_len, num_blocks, d_model, d_ffn, num_heads, p_dropout):
        super().__init__()
        # Create position-encoded embedding
        self.input_emb = TransformerEmbedding(vocab_size, d_model, max_len, p_dropout)
        self.dropout = nn.Dropout(p=p_dropout)
        # Create encoder blocks
        self.enc_blocks = nn.ModuleList([EncoderBlock(d_moel, d_ffn, num_heads, p_dropout)
                                         for _ in range(num_blocks)])
        
    
    def forward(self, src, src_mask, save_attn_pattern=False):
        """
        Args:
            src (torch.Tensor): Input tensor of shape (batch_size, seq_len) with token indices.
            src_mask (torch.Tensor): Mask tensor to prevent attention to specific positions(e.g <PAD>).
            save_attn_pattern (bool): If True, saves attention patterns from each block for visualization.
        Returns:
            torch.Tensor: Final input tokens embedding tensor of shape (batch_size, seq_len, d_model).
            torch.Tensor: Attention patterns (if `save_attn_pattern` is True).
        """
        x = self.input_emb(src)
        x = self.dropout(x)

        attn_patterns = torch.tensor([]).to(DEVICE)
        for block in self.enc_blocks:
            x, attn_pattern = block(x, src_mask)
            # (Optional) if save_attn_pattern is True, save these and return for visualization/investigation
            if save_attn_pattern:
                attn_patterns = torch.cat([attn_patterns, attn_pattern[0].unsqueeze(0)], dim=0)

        return x, attn_patterns