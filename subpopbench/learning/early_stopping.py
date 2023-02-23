import torch

lower_is_better = {
    'AUROC': False,
    'AUPRC': False,
    'BCE': True,
    'ECE': True,
    'accuracy': False,
    'balanced_accuracy': False,
    'precision': False,
    'TPR': False,
    'TNR': False,
    'FPR': True,
    'FNR': True
}


class EarlyStopping:

    def __init__(self, patience=5, lower_is_better=True):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.step = 0
        self.lower_is_better = lower_is_better

    def __call__(self, metric, step, state_dict, path):
        if self.lower_is_better:
            score = -metric
        else:
            score = metric

        if self.best_score is None:
            self.best_score = score
            self.step = step
            save_model(state_dict, path)
        elif score <= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            save_model(state_dict, path)
            self.best_score = score
            self.step = step
            self.counter = 0


def save_model(state_dict, path):
    torch.save(state_dict, path)
