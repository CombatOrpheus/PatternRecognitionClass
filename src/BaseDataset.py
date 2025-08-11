import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.PetriNets import SPNData, SPNAnalysisResultLabel


@dataclass
class BaseDataset(ABC):
    source_path: Path
    batch_size: int
    size: int = field(init=False, default=None)
    features: int = field(init=False, default=None)
    data: list[Data] = field(init=False, default=None)
    label: SPNAnalysisResultLabel

    def __post_init__(self):
        print(f"Loading dataset from: {self.source_path}")
        self._create_dataloader()
        if not hasattr(self, "loader") or not hasattr(self, "size") or not hasattr(self, "features"):
            raise NotImplementedError(
                "Subclass's _create_dataloader must set 'loader', 'size', and 'features' attributes."
            )
        print("Dataset loaded successfully.")

    def get_num_features(self) -> int:
        assert self.features is not None, "Please, create the dataset by calling `create_dataloader`"
        return self.features

    def get_dataset_size(self) -> int:
        assert self.size is not None, "Please, create the first by calling `create_dataloader`"
        return self.size

    def get_actual_as_tensor(self) -> Tensor:
        # TODO: Review the utility of this function and whether it's doing what it's supposed to.
        return Tensor([data.y for data in self._get_data()])

    def get_dataloader(self, batch=None) -> DataLoader:
        batch_size = batch if batch is not None else self.batch_size
        return DataLoader(self.data, batch_size, shuffle=True, drop_last=True)

    @abstractmethod
    def _create_dataloader(self) -> None:
        """Create a DataLoader from the processed data."""
        raise NotImplementedError

    def _get_data(self) -> Iterable[SPNData]:
        with open(self.source_path) as f:
            for line in f:
                yield SPNData(json.loads(line))
