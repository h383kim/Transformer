import torch
from torch import nn
import copy
from time import time
from tqdm import tqdm

"""
One epoch of training over the provided training dataloader.
Args:
    model (nn.Module): The PyTorch model to be trained.
    train_dataloader (DataLoader): Dataloader providing the training data.
                                   Each batch is expected to be a list of strings.
    loss_fn (nn.Module): The loss function used (e.g., nn.CrossEntropyLoss()).
    optimizer (torch.optim.Optimizer): The optimizer used for parameter updates.
    tokenizer: Tokenizer to preprocess the input and target sequences.
    scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler. Default is None.
Returns:
    float: The average training loss over the entire epoch.
"""
def epoch_train(model, train_dataloader, loss_fn, optimizer, tokenizer, scheduler=None):
    # Set model to train mode
    model.train()
    train_loss = 0.0
    # Loop through batches in the training dataloader
    for src, trg in tqdm(train_dataloader, desc='Train'):
        # Tokenizing src texts (texts to be translated)
        src_seq = tokenizer(
            src, 
            padding=True, 
            truncation=True, 
            max_length=100, 
            return_tensors='pt',
            add_special_tokens=False,
        ).input_ids.to(DEVICE) # (batch_size, src_seq_len)

        # Tokenizing trg texts (texts to be the result of translation)
        # add <SOS> Token to each sequence
        ''' Note: Replace '</s>' with SOS token of your tokenizer '''
        trg = ['</s> ' + trg_sent for trg_sent in trg]
        trg_seq = tokenizer(
            trg, 
            padding=True, 
            truncation=True, 
            max_length=100, 
            return_tensors='pt',
            add_special_tokens=True, # Add <EOS> token in the end
        ).input_ids.to(DEVICE) # (batch_size, trg_seq_len)
        
        # Forward pass
        # src_seq shape: (batch_size, src_seq_len)
        # We predict for every token except the last one, hence inputs[:, :-1]
        # y_logits shape: (batch_size, trg_seq_len, trg_vocab_size)
        y_logits, _, _ = model(src_seq, trg_seq[:, :-1])
        
        # Compute the loss:
        #   - permute(0,2,1) transforms y_logits from (B, S, V) to (B, V, S)
        #     to match nn.CrossEntropyLoss input spec: (N, C, d1, d2, ...)
        #   - Compare with the shifted target inputs[:, 1:]
        #     since the "true" next token is at position i+1
        loss = loss_fn(y_logits.permute(0, 2, 1), trg_seq[:, 1:])
        train_loss += loss.item() * src_seq.shape[0]

        # Optimizer zero_grad
        optimizer.zero_grad()

         # Backpropagate to compute gradients
        loss.backward()

        # Update the model parameters
        optimizer.step()

        # Scheduler update
        if scheduler is not None:
            scheduler.step()

     # Compute the average loss across all batches
    train_loss /= len(train_dataloader)
    return train_loss


"""
Evaluate the model on a validation dataset (without parameter updates).
Args:
    val_dataloader (DataLoader): Dataloader providing the validation data.
                                 Each batch is expected to be a list of strings.
Returns:
    float: The average validation loss over the entire dataset.
"""
def evaluate(model, val_dataloader, loss_fn, tokenizer):
    # Set model to evaluation mode
    model.eval()
    val_loss = 0.0

    # Evaluation does not require gradient tracking
    with torch.no_grad():
        for src, trg in tqdm(val_dataloader, desc='Validation'):
            # Tokenizing src texts (texts to be translated)
            src_seq = tokenizer(
                src, 
                padding=True, 
                truncation=True, 
                max_length=100, 
                return_tensors='pt',
                add_special_tokens=False,
            ).input_ids.to(DEVICE) # (batch_size, src_seq_len)

            # Tokenizing trg texts (texts to be the result of translation)
            # add <SOS> Token to each sequence
            ''' Note: Replace '</s>' with SOS token of your tokenizer '''
            trg = ['</s> ' + trg_sent for trg_sent in trg]
            trg_seq = tokenizer(
                trg, 
                padding=True, 
                truncation=True, 
                max_length=100, 
                return_tensors='pt',
                add_special_tokens=True, # Add <EOS> token in the end
            ).input_ids.to(DEVICE) # (batch_size, trg_seq_len)
            
            # Forward Pass
            y_logits, _, _ = model(src_seq, trg_seq[:, :-1]) # (batch_size, trg_seq_len, trg_vocab_size)
            # Calculate Loss
            loss = loss_fn(y_logits.permute(0, 2, 1), trg_seq[:, 1:])
            val_loss += loss.item() * src_seq.shape[0]

    val_loss /= len(val_dataloader)
    return val_loss



"""
Main training loop that orchestrates the training and evaluation phases,
and tracks the best model weights based on validation loss.
Args:
    model (nn.Module): The PyTorch model to be trained.
    train_dataloader (DataLoader): Dataloader providing the training data.
    val_dataloader (DataLoader): Dataloader providing the validation data.
    loss_fn (nn.Module): Loss function to be used during training/evaluation.
    optimizer (torch.optim.Optimizer): Optimizer used for model updates.
    num_epochs (int, optional): Number of epochs to train. Defaults to 1.
Returns:
    nn.Module: The trained model with the best weights (lowest validation loss).
"""
def train(model, train_dataloader, val_dataloader, loss_fn, optimizer, tokenizer, num_epochs=1, scheduler=None):
    # Track the best validation loss and corresponding model weights
    best_loss = float('inf')
    best_wts = copy
    # For logging losses across epochs
    loss_dict = {
        'train_loss': [],
        'val_loss': []
    }

    # Loop over the desired number of epochs
    for epoch in range(1, num_epochs+1):
        start = time()
        # Feed forward / backprop on train_dataloader
        train_loss = epoch_train(model, train_dataloader, loss_fn, optimizer, tokenizer, scheduler)
        # Feed forward on val_dataloader
        val_loss = evaluate(model, val_dataloader, loss_fn, tokenizer)

        # Storing epoch histories
        loss_dict['train_loss'].append(train_loss)
        loss_dict['val_loss'].append(val_loss)

        # Update model depending on its peformance on validation data
        if val_loss < best_loss:
            best_loss = val_loss
            best_wts = copy.deepcopy(model.state_dict())

        # Print epoch summary
        end = time()
        time_elapsed = end - start
        print(f"------------ epoch {epoch} ------------")
        print(f"Train loss: {train_loss:.4f}")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Time taken: {time_elapsed / 60:.0f}min {time_elapsed % 60:.0f}s")

    # Load the best weights (lowest validation loss) into the model
    model.load_state_dict(best_wts)
    return model