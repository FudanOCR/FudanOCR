import os
import numpy as np
import torch
from tqdm import tqdm

import config as cfg


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.

    Modified from https://github.com/Bjarten/early-stopping-pytorch
    """
    def __init__(self, patience, val_loss_min):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.

            val_loss_min: Load minimal validation loss when training resumed.
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = val_loss_min

    def __call__(self, val_loss):

        score, save = -val_loss, False

        if self.best_score is None:
            save = True
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            tqdm.write(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            save = True
            self.best_score = score
            self.counter = 0

        return self.early_stop, save

    def save_checkpoint(self, state, val_loss):
        '''Saves checkpoint when validation loss decrease.'''
        if not os.path.exists(cfg.result_dir):
            os.mkdir(cfg.result_dir)
        epoch = state['epoch']
        filename = cfg.task_id + '_best.pth.tar'
        file_path = os.path.join(cfg.result_dir, filename)
        torch.save(state, file_path)
        if self.val_loss_min is not None:
            tqdm.write(f'Validate loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f}) Saving checkpoint - Epoch: [{epoch}]')
        self.val_loss_min = val_loss
