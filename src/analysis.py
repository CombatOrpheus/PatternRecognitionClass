from pathlib import Path

import numpy as np
import torch
from torch.nn import Module
from torch.nn import functional as F
from torch_geometric.loader import DataLoader

from .data_parser import get_average_tokens


def variance(sample) -> torch.tensor:
    return torch.mean((sample - torch.mean(sample)) ** 2)


def std_deviation(sample) -> torch.tensor:
    return variance(sample) ** 2


def statistical_values(sample) -> torch.tensor:
    return torch.mean(sample), std_deviation(sample), variance(sample)


def evaluate_model(model: Module, dataset: DataLoader) -> np.array:
    actual = torch.tensor([batch.y for batch in dataset])
    pred = list(map(model, dataset))

    pred = torch.tensor(pred)
    pred = torch.flatten(pred)

    mre = torch.mean(torch.abs(pred - actual) / actual) * 100
    return F.l1_loss(pred, actual), torch.nn.MSELoss()(pred, actual), mre


def evaluate_metrics(model: Module, path: Path) -> np.array:
    files = map(get_average_tokens, path.glob("*all*"))
    for dataset in files:
        pass
