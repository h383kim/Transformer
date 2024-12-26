import torch
from torch import nn

def translate(model, text, tokenizer, SOS_token='</s>', EOS_token='</s>', save_attn_pattern=False):
    """
    Translates a given input text using a sequence-to-sequence model with an encoder-decoder architecture.

    Parameters:
        model: nn.Module
            The sequence-to-sequence model for translation. It should include an encoder and decoder.
        text: str or list of str
            The input text(s) to be translated.
        tokenizer: Tokenizer
            Tokenizer object to encode the input text and decode the output tokens.
        SOS_token: str, optional (default='</s>')
            The start-of-sequence token used in the target language.
        EOS_token: str, optional (default='</s>')
            The end-of-sequence token used in the target language.
        save_attn_pattern: bool, optional (default=False)
            If True, saves the attention patterns during translation for analysis.

    Returns:
        str:
            The translated text.

    Notes:
        This function assumes that the `model` has methods `_make_enc_pad_mask`, `_make_dec_pad_mask`, and
        `_make_pad_future_mask` for generating padding and future masks. These masks are used to control the attention
        mechanism in the transformer during encoding and decoding.
    """
    # Tokenize the input text(s) and convert to tensors
    input_texts = tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors='pt',
        max_length=100,  # Limit the maximum length of input sequences
        add_special_tokens=False  # Do not add special tokens like [CLS], [SEP], etc.
    ).input_ids.to(DEVICE)  # Move tensors to the target device (e.g., GPU or CPU)

    # Encode the Start-of-Sequence (SOS) token for the target language
    SOS = tokenizer.encode(SOS_token)[0]  # Get the integer ID for SOS_token
    output_texts = torch.tensor([[SOS]]).to(DEVICE)  # Initialize the output sequence with SOS token

    # Set the model to evaluation mode to disable dropout and other training-specific behaviors
    model.eval()

    # Disable gradient calculations to save memory and computation
    with torch.no_grad():
        # Encoder Pass: Process the input text(s) through the encoder once
        enc_pad_mask = model._make_enc_pad_mask(input_texts)  # Generate padding mask for the encoder
        enc_out, _ = model.encoder(input_texts, enc_pad_mask, save_attn_pattern)  # Get encoded representations

        # Decoder Pass: Iteratively generate tokens until the end-of-sequence (EOS) token is produced
        for _ in range(MAX_LEN):  # Limit the maximum length of the output sequence
            # Generate masks for the decoder
            dec_pad_mask = model._make_dec_pad_mask(input_texts, output_texts)  # Mask for padding in decoder inputs
            pad_future_mask = model._make_pad_future_mask(output_texts)  # Mask to prevent attending to future tokens

            # Pass the current output sequence and encoded representations through the decoder
            dec_out, _, _ = model.decoder(
                output_texts,  # Current decoder input sequence
                enc_out,       # Encoder outputs
                pad_future_mask,  # Prevent attending to future tokens
                dec_pad_mask,  # Padding mask
                save_attn_pattern  # Save attention patterns if needed
            )

            # Select the most likely next token from the decoder output
            # `dec_out[:, -1, :]` represents the logits for the last token in the sequence
            next_token = dec_out[:, -1, :].argmax(dim=-1)  # Shape: (1,)

            # Append the selected token to the output sequence
            output_texts = torch.cat([output_texts, next_token.unsqueeze(0)], dim=-1)

            # Break the loop if the end-of-sequence (EOS) token is produced
            if next_token == tokenizer.encode(EOS_token)[0]:
                break

    # Decode the generated output tokens (excluding the initial SOS token) into a string
    final_translation = tokenizer.decode(output_texts[0, 1:])

    return final_translation
