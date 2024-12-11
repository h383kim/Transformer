import torch
from torch import nn
from einops import rearrange

'''
Attention used in the paper
'''
class Scaled_DotProduct(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, scale=None, mask=None):
        '''
        Args:
            q, k, v: (batch_size, num_heads, seq_len, d_head)
        Returns:
            attention_context: (batch_size, num_heads, seq_len, d_head)
            attention_score: (batch_size, num_heads, seq_len, seq_len)
        '''
        # Step 1: apply dot product and scale down
        k_T = k.transpose(-2, -1)
        attention_score = (q @ k_T) / scale # (batch, n_heads, seq_len, seq_len)
        # Step 2: apply masking if applicable
        if mask is not None:
            attention_score[mask] = -1e10
        # Step 3: apply softmax to redistribute within (0, 1)
        attention_score = torch.softmax(attention_score, dim=-1)
        # Step 4: Weighted sum of values matrix
        attention_context = attention_score @ v

        return attention_context, attention_score

'''
Multi-head Attention
'''
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
        # FC Layers that create Queries, Keys, Values
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        # FC Layer after Multi-head attended results are concatenated
        self.W_out = nn.Linear(d_model, d_model)
        # sqrt(d_k) to reduce variance of dot product as embedding dim grows
        self.scale = torch.sqrt(torch.tensor(d_model / num_heads))
        # Attention Mechanism
        self.attention = Scaled_DotProduct()

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q, k, v (Tensor): Input tensor of token with shape (batch_size, seq_len, d_model).
        - Note that q, k, v as inputs are IDENTICAL input token embeddings in self-attention
        - But q is from decoder embedding while k and v are encoder output embedding in cross-attention
        Returns:
            attention_context (Tensor): Output tensor of attented vectors with shape (batch_size, seq_len, d_model).
            attention_score (Tensor): Attention_pattern for investigation use
        """

        # Step 1: Create Query, Key, Value vectors
        Q = self.W_Q(q)
        K = self.W_K(k)
        V = self.W_V(v)
        
        # Step 2: Split Query, Key, Value vectors into multiple heads using rearrange method
        '''
        (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, d_head)
        Namings:
            b = batch_size
            s = seq_len
            h = num_heads
            d = d_head
        '''
        Q, K, V = [
            rearrange(x, 'b s (h d) -> b h s d', h=self.num_heads)
            for x in (Q, K, V)
        ]
        
        # Step 3: Apply attention to create attention pattern
        attention_context, attention_score = self.attention(Q, K, V, self.scale, mask)

        # Step 4: Merge the heads back
        attention_context = rearrange(attention_context, 'b h s d -> b s (h d)')

        # Step 5: Apply the last FC layer after concatenation
        attention_context = self.W_out(attention_context)

        return attention_context, attention_score
