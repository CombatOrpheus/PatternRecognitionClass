from torch import from_numpy
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.BaseDataset import BaseDataset


class SPNDataset(BaseDataset):
    def _create_dataloader(self) -> None:
        nets = (net.to_information() for net in self._get_data())
        data = [
            Data(
                x=from_numpy(net[2]),
                edge_attr=from_numpy(net[1]),
                edge_index=from_numpy(net[0])
            )
            for net in nets]
        self.data = data
        self.size = len(data)
        self.features = 1
        self.loader = DataLoader(data, self.batch_size, shuffle=True, drop_last=True)
