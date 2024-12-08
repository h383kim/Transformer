import torch
from torch import nn

from models.Embedding.embedding import TransformerEmbedding
from models.Layers.multi_head_attention import MultiHeadAttention
from models.Layers.feed_forward_network import FFN


"""
Single block in the Transformer decoder.

Each block consists of:
    - A self-attention mechanism (decoder-only attention) with residual connections and LayerNorm.
    - A cross-attention mechanism (encoder-decoder attention) with residual connections and LayerNorm.
    - A feed-forward network (FFN) with residual connections and LayerNorm.

Args:
    d_model (int): Dimensionality of the input embeddings.
    d_ffn (int): Dimensionality of the hidden layer in the feed-forward network.
    num_heads (int): Number of attention heads in the multi-head attention mechanisms.
    p_dropout (float): Dropout probability for regularization.
"""
class DecoderBlock(nn.Module):
    def __init__(self, d_model, d_ffn, num_heads, p_dropout):
        super().__init__()
        
        # Decoder Self-Attention
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.dropout1 = nn.Dropout(p=p_dropout)
        self.norm1 = nn.LayerNorm(d_model)
        # Encoder-Decoder Attention (Cross-Attention)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.dropout2 = nn.Dropout(p=p_dropout)
        self.norm2 = nn.LayerNorm(d_model)
        # Feed-Forward Network
        self.ffn = FFN(d_model, d_ffn, p_dropout)
        self.dropout3 = nn.Dropout(p=p_dropout)
        self.norm3 = nn.LayerNorm(d_model)
        
    def forward(self, x, enc_out, self_attn_mask, cross_attn_mask):
        """
        Args:
            x (torch.Tensor): Target sequence tensor of shape (batch_size, tgt_seq_len, d_model).
            enc_out (torch.Tensor): Encoder output tensor of shape (batch_size, src_seq_len, d_model).
            self_attn_mask (torch.Tensor): Source mask of shape (batch_size, 1, src_seq_len, src_seq_len).
            cross_attn_mask (torch.Tensor): Target mask of shape (batch_size, 1, tgt_seq_len, tgt_seq_len).
        Returns:
            torch.Tensor: Transformed output tensor of shape (batch_size, tgt_seq_len, d_model).
            torch.Tensor: Self-attention scores for the target sequence.
            torch.Tensor: Cross-attention scores for the encoder-decoder interaction.
        """
        # Self-Attention with residual connection and normalization
        attn_context, attn_score = self.self_attention(x, x, x, mask=self_attn_mask)
        attn_context = self.dropout1(attn_context)
        x = self.norm1(x + attn_context)
        # Cross-Attention with residual connection and normalization
        cross_attn_context, cross_attn_score = self.cross_attention(x, enc_out, enc_out, mask=cross_attn_mask)
        cross_attn_context = self.dropout2(cross_attn_context)
        x = self.norm2(x + cross_attn_context)
        # Feed-forward network with residual connection and normalization
        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.norm3(x)

        return x, attn_score, cross_attn_score




"""
Represents the full Transformer decoder consisting of multiple decoder blocks.

The decoder applies:
    - Positional encoding to the target input embeddings.
    - Multiple decoder blocks, each with self-attention, cross-attention, and feed-forward layers.
    - A final linear layer to project the output to the vocabulary size for token predictions.

Args:
    dec_vocab_size (int): Size of the target vocabulary.
    max_len (int): Maximum sequence length for positional encoding.
    num_blocks (int): Number of decoder blocks in the decoder.
"""
class Decoder(nn.Module):
    def __init__(self, dec_vocab_size, max_len, num_blocks, d_model, d_ffn, num_heads, p_dropout):
        super().__init__()

        # Create position-encoded embedding
        self.input_emb = TransformerEmbedding(dec_vocab_size, d_model, max_len, p_dropout)
        self.dropout = nn.Dropout(p=p_dropout)
        # Create decoder blocks
        self.dec_blocks = nn.ModuleList([DecoderBlock(d_model, d_ffn, num_heads, p_dropout)
                                         for _ in range(num_blocks)])
        # Last FC Layer
        self.fc_out = nn.Linear(d_model, dec_vocab_size)
        
    def forward(self, trg, enc_out, self_attn_mask, cross_attn_mask, save_attn_pattern=False):
        """
        Args:
            trg (torch.Tensor): Target sequence tensor of shape (batch_size, tgt_seq_len).
            enc_out (torch.Tensor): Encoder output tensor of shape (batch_size, src_seq_len, d_model).
            self_attn_mask (torch.Tensor): Source mask to prevent attention to specific positions.
            cross_attn_mask (torch.Tensor): Target mask to enforce causal masking (no attention to future positions).
            save_attn_pattern (bool): If True, saves and returns attention patterns for visualization.
        Returns:
            torch.Tensor: Output logits of shape (batch_size, tgt_seq_len, dec_vocab_size).
            torch.Tensor: (Optional) Self-attention patterns from all decoder blocks.
            torch.Tensor: (Optional) Cross-attention patterns from all decoder blocks.
        """
        x = self.input_emb(trg)
        x = self.dropout(x)

        self_attn_patterns = torch.tensor([]).to(DEVICE)
        cross_attn_patterns = torch.tensor([]).to(DEVICE)
        for block in self.dec_blocks:
            x, self_attn_score, cross_attn_score = block(x, enc_out, self_attn_mask, cross_attn_mask)
            # (Optional) if save_attn_pattern is True, save these and return for visualization/investigation
            if save_attn_pattern:
                self_attn_patterns = torch.cat([self_attn_patterns, self_attn_score[0].unsqueeze(0)], dim=0)
                cross_attn_patterns = torch.cat([cross_attn_patterns, cross_attn_score[0].unsqueeze(0)], dim=0)

        x = self.fc_out(x)

        return x, self_attn_patterns, cross_attn_patterns