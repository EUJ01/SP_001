from time import time
import pandas as pd
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from tqdm import trange, tqdm
import wandb
import json
import os

# MOD
def train_user_model(model, train_dataloader, val_dataloader, device, num_epoch, lr, step_size, gamma, data_summary):
    """Train the model given the training dataloader and validate after each epoch.
    
    Args:
        model (nn.Module): The model to train.
        train_dataloader (DataLoader): Batch iterator containing the training data.
        val_dataloader (DataLoader): Batch iterator containing the validation data.
        num_epoch (int): Number of epochs to train.
        lr (float): Learning rate for the optimizer.
    
    Returns:
        log (DataFrame): Log containing epoch-wise training losses and validation scores.
        saved_model_state_dict: State dictionary of the best saved model.
    """
    with open(os.path.join('settings', f'local_test.json'), 'r') as fp:
        config = json.load(fp)
        config = config[0]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)  

    bar_desc = 'Training, avg loss: %.3f'
    log = []
    saved_model_state_dict = None
    best_loss = 1e9
    
    # wandb
    run = wandb.init(
        entity = "SP_001",
        project = "TrajFM TUL",
        config={
            **data_summary,
            "epoch" : int(config["finetune"]["config"]["num_epoch"]),
            "batch_size" : int(config["finetune"]["dataloader"]["batch_size"]),
            "learning_rate" : float(config["finetune"]["config"]["lr"]),
            "step_size" : step_size,
            "gamma" : gamma,
            "embed_size": config["trajfm"]["embed_size"],
            "d_model": config["trajfm"]["d_model"],
            "rope_layer": config["trajfm"]["rope_layer"],
        },
        id = config["save_name"],
        resume="allow" if id else None,  
    )
    print(f"The run id is {run.id}")


    with trange(num_epoch, desc=bar_desc % 0.0) as bar:
        for epoch_i in bar:
            loss_values = []
            epoch_time = 0
            
            # Training loop
            model.train()
            for batch in tqdm(train_dataloader, desc='-->Traversing', leave=False):
                (input_tensor, output_tensor, pos_tensor) = batch
                
                input_tensor, output_tensor, pos_tensor = input_tensor.to(device), output_tensor.to(device), pos_tensor.to(device)
                optimizer.zero_grad()
                s_time = time()
                loss = model.user_loss(input_tensor, output_tensor, pos_tensor)
                loss.backward()
                optimizer.step()
                e_time = time()
                loss_values.append(loss.item())
                epoch_time += e_time - s_time
            
            # Compute average training loss
            loss_epoch = np.mean(loss_values)
            bar.set_description(bar_desc % loss_epoch)
            log.append({'epoch': epoch_i, 'time': epoch_time, 'loss': loss_epoch})
            scheduler.step()
            
            # Validation
            val_metrics, val_loss = test_user_model(model, device, val_dataloader)

            for key, value in val_metrics.items():
                print(f"{key}: {round(value * 100, 2)}%,")
            print("val_loss", round(val_loss, 3))
            # Log validation metrics
            log[-1].update(val_metrics)
            log_data = {
                "epoch": epoch_i + 1,
                "Train_Loss": loss,
                "Val_loss": val_loss,
            }
            for key, value in val_metrics.items():
                log_data[f"{key}"] = round(value * 100, 2)
            run.log(log_data)

            # Save best model
            if loss_epoch < best_loss:
                best_loss = loss_epoch
                saved_model_state_dict = model.state_dict()

    # Convert log to DataFrame
    log_df = pd.DataFrame(log)
    log_df = log_df.set_index('epoch')
    
    return log_df, saved_model_state_dict


@torch.no_grad()
def test_user_model(model, device, dataloader):
    """Test the model given the testing dataloader.
    
    Args:
        model (nn.Module): The model to test.
        device (torch.device): The device (CPU or GPU) to run the model on.
        dataloader (dataloader): Batch iterator containing the testing data.
    
    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    model.eval()
    total_metrics = {
        'ACC@1': 0.0,
        'ACC@5': 0.0,
        'Macro-R': 0.0,
        'Macro-P': 0.0,
        'Macro-F1': 0.0,
    }
    
    num_batches = len(dataloader)
    loss_values = []
    
    for batch in tqdm(dataloader, desc='Testing/Validating'):
        (input_tensor, output_tensor, pos_tensor) = batch
        input_tensor, output_tensor, pos_tensor = input_tensor.to(device), output_tensor.to(device), pos_tensor.to(device)

        # Call the test_user function to obtain metrics for each batch
        metrics = model.test_user(input_tensor, output_tensor, pos_tensor)
        
        loss = model.user_loss(input_tensor, output_tensor, pos_tensor)
        loss_values.append(loss.item())
        
        # Aggregate scores
        total_metrics['ACC@1'] += metrics['ACC@1']
        total_metrics['ACC@5'] += metrics['ACC@5']
        total_metrics['Macro-R'] += metrics['Macro-R']
        total_metrics['Macro-P'] += metrics['Macro-P']
        total_metrics['Macro-F1'] += metrics['Macro-F1']
        
    loss_epoch = np.mean(loss_values)
    
    # Average metrics over the number of batches
    for key in total_metrics:
        total_metrics[key] /= num_batches

    return total_metrics, loss_epoch
# MOD

def pad_batch_arrays(arrs):
    """Pad a batch of arrays with representing feature sequences of different lengths.

    Args:
        arrs (list): each item is an array with shape (B, L, ...). The length L is different for different arrays.

    Returns:
        np.array: padded arrays with shape (B_agg, L_max, ...) that are concatenated along the batch axis.
    """
    max_len = max(a.shape[1] for a in arrs)
    arrs = [
        np.concatenate([a, np.repeat(a[:, -1:], repeats=max_len-a.shape[1], axis=1)], axis=1)
        for a in arrs
    ]
    return np.concatenate(arrs, 0)
