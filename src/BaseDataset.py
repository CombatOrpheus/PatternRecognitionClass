import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Literal

from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from SPNData import SPNData


@dataclass
class BaseDataset(ABC):
    """
    Abstract base class for datasets of Stochastic Petri Nets (SPNs).

    This class provides a common interface for loading and processing SPN data
    from JSON-L files, converting it to PyTorch Geometric `Data` objects,
    and creating a `DataLoader` for use in GNN models.  Concrete subclasses
    must implement the `_create_dataloader` method to handle the specifics of
    data transformation and `DataLoader` creation.

    Attributes:
        source_path (Path): Path to the JSON-L file containing the SPN data.
        batch_size (int): The batch size to use for the `DataLoader`.
        size (int): The total number of SPN instances in the dataset.  This
            is automatically determined by counting the lines in the source file.
        features (int): The number of features per node in the graph
            representation (determined by subclasses).
        data (list[Data]): A list of PyTorch Geometric `Data` objects, one for
            each SPN instance.  Populated by `_create_dataloader`.
        loader (DataLoader): A PyTorch Geometric `DataLoader` for iterating
            over the dataset in batches. Populated by `_create_dataloader`.

    Raises:
        FileNotFoundError: If the `source_path` does not exist or is not a file.
        NotImplementedError:  If `_create_dataloader` is not implemented by a subclass.
        AssertionError: If the dataloader is accessed before `create_dataloader` is executed.
    """
    LABELS = Literal['network_average', 'place_average', 'steady_state', 'mark_density']
    source_path: Path
    batch_size: int
    size: int = field(init=False, default=None)
    features: int = field(init=False, default=None)
    data: List[Data] = field(init=False, default=None)
    loader: DataLoader = field(init=False, default=None)

    def __post_init__(self):
        """
        Initializes the dataset size by counting lines in the source file.

        Raises:
            FileNotFoundError: If the `source_path` does not exist or is not a file.
        """
        if not self.source_path.exists() or not self.source_path.is_file():
            raise FileNotFoundError(f"The specified file '{self.source_path}' does not exist or is not a file.")

        with open(self.source_path) as f:
            self.size = sum(1 for _ in f)  # Efficiently count lines

    def get_actual_as_tensor(self) -> Tensor:
        """
        Returns the ground truth labels (y values) for all data instances as a single PyTorch tensor.

        Returns:
            Tensor: A tensor containing the 'y' attribute from each `Data` object in `self.data`.

        Raises:
            AssertionError: If no data has been loaded using `_create_dataloader`.
        """
        assert self.data, "Please, create the first by calling `create_dataloader`"
        return Tensor([data.y for data in self.data])

    def get_dataloader(self) -> DataLoader:
        """
        Returns the PyTorch Geometric DataLoader.

        Returns:
            DataLoader: The DataLoader instance associated with this dataset.
        """
        return DataLoader(dataset=self.data, batch_size=self.batch_size, shuffle=True, drop_last=True)

    @abstractmethod
    def _create_dataloader(self) -> None:
        """
        Abstract method to create the PyTorch Geometric DataLoader.

        Subclasses must implement this method to:
            1.  Load and process the SPN data from `self.source_path`.
            2.  Convert each SPN instance into a PyTorch Geometric `Data` object.
            3.  Store the `Data` objects in `self.data`.
            4.  Create a `DataLoader` from `self.data` and store it in `self.loader`.
            5. Set up the number of features.

        Raises:
            NotImplementedError:  This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def _get_data(self) -> Iterable[SPNData]:
        """
        Yields SPNData objects parsed from the source JSON-L file.

        This is a helper method for reading the raw data. It's used by `_create_dataloader`.

        Yields:
            SPNData:  An SPNData object for each line in the source file.

        Raises:
            Exception: Catches and prints JSON parsing exceptions.
        """
        with open(self.source_path) as f:
            for line in f:
                try:
                    yield SPNData(json.loads(line))
                except Exception as e:
                    print(f"Error processing the input file: {e}")