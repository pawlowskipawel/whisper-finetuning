# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/metrics.ipynb.

# %% auto 0
__all__ = ['WER']

# %% ../nbs/metrics.ipynb 1
import evaluate
import torch

# %% ../nbs/metrics.ipynb 2
class WER:
    def __init__(self):
        super().__init__()

        self.targets = []
        self.predictions = []

        self.current_metric = 0
        self.metric = evaluate.load("wer")
        
    def reset(self):
        self.targets = []
        self.predictions = []

        self.current_metric = 0

    def update(self, predictions=None, labels=None):
        
        self.predictions.extend(predictions)
        self.targets.extend(labels)

    def compute(self):

        self.current_metric = self.metric.compute(references=self.targets, \
            predictions=self.predictions)

        return self.current_metric
    
    def get_metric(self):
        if torch.is_tensor(self.current_metric):
            return self.current_metric.item()

        return self.current_metric
