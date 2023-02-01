import numpy as np

class EarlyStopper:
    def __init__(self, patience=1):
        self.patience = patience
        self.counter = 0
        self.min_loss = np.inf
        self.best_model_weights = None
        self.best_epoch = None

    def early_stop(self, loss, model_weights, epoch):
        if loss > self.min_loss:
            self.counter += 1
            
        elif loss < self.min_loss:
            self.min_loss = loss
            self.best_model_weights = model_weights
            self.best_epoch = epoch
            self.counter = 0
            
        return self.counter >= self.patience