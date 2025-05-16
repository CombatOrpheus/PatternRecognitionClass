"""
This module defines the SPNDataset class, a subclass of BaseDataset, for handling SPN (Sum-Product Network) data.
It includes methods for creating data loaders and padding features.
"""
from torch import from_numpy
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.BaseDataset import BaseDataset


class SPNDataset(BaseDataset):
    pad: bool = True
    def _create_dataloader(self) -> None:
        nets = ((net.to_information(), net.get_analysis_result(self.label)) for net in self._get_data())
        data = [
            Data(
                x=from_numpy(net[0]).float(),
                edge_attr=from_numpy(net[1]).float(),
                edge_index=from_numpy(net[2]).long(),
                y=label
            )
            for net, label in nets]

        self.data = data

        self.size = len(data)
        self.features = data[0].x.shape[1]
        self.loader = DataLoader(data, self.batch_size, shuffle=True, drop_last=True)


def _pad(data: list[Data]) -> list[Data]:
    """
    Pads the edge features of the graphs in the dataset to have the same size.
    This is necessary for batching the graphs together.
    """
    from torch import cat, zeros

    max_features = max(d.edge_attr.shape[0] for d in data)
    for d in data:
        features = d.edge_attr
        padding = max_features - features.shape[0]
        if padding > 0:
            d.edge_attr = cat((features, zeros(padding, d.edge_attr.shape[1])), dim=0)

    assert (len(set(d.edge_attr.shape for d in data)) == 1)
    return data
