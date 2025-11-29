import torch
import numpy as np
import os
from tqdm import tqdm
import datetime

class EarlyStopping:
    """
       Early Stopping
    """
    def __init__(self, patience=10, verbose=False, delta=0, save_dir='./checkpoints'):
        """
        Args:
            patience (int): Number of epochs to wait after last improvement.
            verbose (bool): Whether to print messages.
            delta (float): Minimum change in the monitored metric to qualify as an improvement.
            save_dir (str): Directory to save the model.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.save_dir = save_dir
        self.best_model_path = ""

    def __call__(self, val_metric, model, optimizer, epoch, config, args):
        score = -val_metric 

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model, optimizer, epoch, config, args)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_metric, model, optimizer, epoch, config, args)
            self.counter = 0

    def save_checkpoint(self, val_metric, model, optimizer, epoch, config, args):
        if self.verbose:
            print(f'Validation metric improved. Saving model...')
        
        current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f'best_model_{current_time}_ep{epoch}_mae{val_metric:.4f}.pth'
        save_path = os.path.join(self.save_dir, model_name)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'args': args, 
            'best_mae': val_metric
        }, save_path)
        
        self.best_model_path = save_path
        print(f"  [Saved Best Model] -> {save_path} (MAE: {val_metric:.2f})")