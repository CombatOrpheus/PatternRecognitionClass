from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.nn import Module

from .data_parser import get_average_tokens


def variance(sample) -> torch.tensor:
    return torch.mean((sample - torch.mean(sample))**2)


def std_deviation(sample) -> torch.tensor:
    return variance(sample)**2


def statistical_values(sample) -> torch.tensor:
    return torch.mean(sample), std_deviation(sample), variance(sample)


def evaluate_metrics(model: Module, path: Path) -> np.array:
    files = map(get_average_tokens, path.glob('*all*'))
    for dataset in files:
        pass
