import torch
from torchmetrics import Metric

class MaxError(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("max_error", default=torch.tensor(0.0), dist_reduce_fx="max")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.max_error = torch.max(self.max_error, torch.max(torch.abs(preds - target)))

    def compute(self):
        return self.max_error

class MedianAbsoluteError(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("errors", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.errors.append(torch.abs(preds - target))

    def compute(self):
        errors = torch.cat(self.errors)
        return torch.median(errors)
