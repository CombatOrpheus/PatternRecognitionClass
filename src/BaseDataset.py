import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pytoolconfig import field
from torch import Tensor
from torch_geometric.data import Data

from src.petri_nets import SPNData


@dataclass
class BaseDataset(ABC):
    source_path: Path
    batch_size: int
    size: int = field(init=False, default=None)
    features: int = field(init=False, default=None)
    data: list[Data] = field(init=False, default=None)

    def get_num_features(self) -> int:
        assert self.features is not None, "Please, create the dataset by calling `create_dataloader`"
        return self.features

    def get_dataset_size(self) -> int:
        assert self.size is not None, "Please, create the first by calling `create_dataloader`"
        return self.size

    def get_actual_as_tensor(self) -> Tensor:
        assert len(self.data) > 0, "Please, create the first by calling `create_dataloader`"
        return Tensor([data.y for data in self.data])

    @abstractmethod
    def create_dataloader(self):
        """Create a DataLoader from the processed data."""
        raise NotImplementedError

    def _get_data(self) -> Iterable[SPNData]:
        with open(self.source_path) as f:
            for line in f:
                yield SPNData(json.loads(line))
