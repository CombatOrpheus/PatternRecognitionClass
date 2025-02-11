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
    size: int = field(init=False)
    features: int = field(init=False)
    data: list[Data] = field(init=False)

    def get_num_features(self) -> int:
        return self.features

    def get_dataset_size(self) -> int:
        return self.size

    def get_data_as_tensor(self) -> Tensor:
        pass

    @abstractmethod
    def create_dataloader(self, data):
        """Create a DataLoader from the processed data."""
        raise NotImplementedError


    def _get_data(self) -> Iterable[SPNData]:
        with open(self.source_path) as f:
            for line in f:
                yield SPNData(json.loads(line))
