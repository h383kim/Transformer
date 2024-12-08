import torch
from torch import nn

from models.Model.Encoder import Encoder
from models.Model.Decoder import Decoder


"""
Transformer model for sequence-to-sequence tasks, consisting of an encoder and a decoder.

Args:
    src_pad_idx (int): Index of the <PAD> token in the source vocabulary.
    trg_pad_idx (int): Index of the <PAD> token in the target vocabulary.
    enc_vocab_size (int): Size of the source vocabulary.
    dec_vocab_size (int): Size of the target vocabulary.
    max_len (int): Maximum length of input and output sequences.
    num_blocks (int): Number of encoder and decoder blocks.
    d_model (int): Dimensionality of input embeddings and model representations.
    d_ffn (int): Dimensionality of the feed-forward network's hidden layer.
    num_heads (int): Number of attention heads in multi-head attention.
    p_dropout (float): Dropout probability for regularization.
       
Helper Methods:
    _make_enc_pad_mask(src): Generates a padding mask for the encoder's self-attention
    _make_dec_pad_mask(src, trg): Generates a padding mask for the decoder's cross-attention.
    _make_pad_future_mask(trg): Generates a combined padding and future mask for the decoder's self-attention.
"""
class Transformer(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, enc_vocab_size, dec_vocab_size,
                 max_len, num_blocks, d_model, d_ffn, num_heads, p_dropout):
        super().__init__()

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.num_heads = num_heads

        # Initialize encoder and decoder
        self.encoder = Encoder(enc_vocab_size, max_len, num_blocks, d_model, d_ffn, num_heads, p_dropout)
        self.decoder = Decoder(dec_vocab_size, max_len, num_blocks, d_model, d_ffn, num_heads, p_dropout)
    
    def forward(self, src, trg):
        """
        Args:
            src (torch.Tensor): Source input tensor of shape (batch_size, src_seq_len).
            trg (torch.Tensor): Target input tensor of shape (batch_size, trg_seq_len).
        Returns:
            tuple: A tuple containing:
                - decoder_out (torch.Tensor): Decoder output logits of shape (batch_size, trg_seq_len, dec_vocab_size).
                - dec_self_attn_pattern (torch.Tensor): Self-attention patterns from the decoder.
                - dec_cross_attn_pattern (torch.Tensor): Cross-attention patterns from the decoder.
        """
        # Create masking
        enc_pad_mask = self._make_enc_pad_mask(src) # Used in "self-attention of the encoder"
        dec_pad_mask = self._make_dec_pad_mask(src, trg) # Used in "cross-attention of the decoder"
        pad_future_mask = self._make_pad_future_mask(trg) # Used in "self-attention of the decoder"
        
        # Encoder Pass
        encoder_out, enc_attn_pattern = self.encoder(src, enc_pad_mask, True)
        
        # Decoder Pass
        decoder_out, dec_self_attn_pattern, dec_cross_attn_pattern = self.decoder(trg, 
                                                                                  encoder_out, 
                                                                                  self_attn_mask=pad_future_mask,
                                                                                  cross_attn_mask=dec_pad_mask, save_attn_pattern=True)

        return decoder_out, dec_self_attn_pattern, dec_cross_attn_pattern
        

    def _make_enc_pad_mask(self, src):
        """
        Creates a padding mask for encoder's self-attention.
        Args:
            src (torch.Tensor): Source tensor of shape (batch_size, src_seq_len).
        Returns:
            torch.Tensor: Padding mask of shape (batch_size, num_heads, src_seq_len, src_seq_len), 
                          where True indicates <PAD> tokens.
        Example:
            Initial pad_mask for a single sequence (sentence): 
            [F F F T T] (F = False, T = True for <PAD>)
            Expanded across heads and queries(columns):
            [F F F T T]
            [F F F T T]  x num_heads
            [F F F T T]
        """
        pad_mask = (src == self.src_pad_idx)          # (batch_size, seq_len)
        pad_mask = pad_mask.unsqueeze(1).unsqueeze(2) # (batch_size, 1, 1, seq_len)
        pad_mask = pad_mask.expand(src.shape[0], self.num_heads, src.shape[1], src.shape[1]) # (batch_size, num_heads, seq_len, seq_len)
        
        return pad_mask

    def _make_dec_pad_mask(self, src, trg):
        """
        Creates a padding mask for decoder's cross-attention.
        Args:
            src (torch.Tensor): Source tensor of shape (batch_size, src_seq_len).
            trg (torch.Tensor): Target input tensor of shape (batch_size, trg_seq_len).
        Returns:
            torch.Tensor: Padding mask of shape (batch_size, num_heads, trg_seq_len, src_seq_len), 
                          where True indicates <PAD> tokens.
        Note:
            Because the key vectors in cross-attention are retrieved from the encoder's output,
            the pad tokens of encoder's input (i.e. src) will be masked
        """
        pad_mask = (src == self.src_pad_idx)          # (batch_size, seq_len)
        pad_mask = pad_mask.unsqueeze(1).unsqueeze(2) # (batch_size, 1, 1, seq_len)
        pad_mask = pad_mask.expand(src.shape[0], self.num_heads, trg.shape[1], src.shape[1]) # (batch_size, num_heads, seq_len, seq_len)
        
        return pad_mask
    
    def _make_pad_future_mask(self, trg):
        """
        Creates a combined padding and future mask for the decoder.
        Args:
            trg (torch.Tensor): Target tensor of shape (batch_size, seq_len).
        Returns:
            torch.Tensor: Combined mask of shape (batch_size, num_heads, seq_len, seq_len), 
                          where True indicates positions to mask (padding or future tokens).
        Steps:
            1. **Padding Masking**: Identify positions corresponding to <PAD> tokens.
            2. **Future Masking**: Mask future tokens using a lower triangular matrix.
            3. **Combine Masks**: Apply logical OR to combine both masks.
        """
        # Step 1: Pad Masking
        pad_future_mask = (trg == self.trg_pad_idx)                 # (batch_size, seq_len)
        pad_future_mask = pad_future_mask.unsqueeze(1).unsqueeze(2) # (batch_size, 1, 1, seq_len)
        pad_future_mask = pad_future_mask.expand(
            trg.shape[0], self.num_heads, trg.shape[1], trg.shape[1]
        ) # Expanded to (batch_size, num_heads, seq_len, seq_len)
        
        # Step 2: Future Masking
        upper_trig_mask = torch.tril(torch.ones(trg.shape[0], self.num_heads, trg.shape[1], trg.shape[1]))
        upper_trig_mask = (upper_trig_mask == 0).to(DEVICE)

        # Step 3: Combine Masks
        pad_future_mask = pad_future_mask | upper_trig_mask # Mask <PAD> and future tokens

        return pad_future_mask

        