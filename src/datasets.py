from itertools import starmap
from pathlib import Path
from typing import List

import dgl
import numpy as np
import torch
from dgl.data import DGLDataset

from src.data_parser import get_reachability_graphs


class SPNDataset(DGLDataset):
    def __init__(self):
        super(SPNDataset, self).__init__(
            name="Stochastic Petri Nets"
        )

    def process(self):
        file = Path('Data/GridData_DS3.processed')
        graphs = get_reachability_graphs(file)
        self._labels = []
        self._graphs = []

        labels = []
        for graph, label in graphs:
            g = dgl.graph((graph[1][:, 0], graph[1][:, 1]), num_nodes=len(graph[0]))
            g.ndata['feat'] = torch.tensor(graph[0]).float()
            g.edata['feat'] = torch.tensor(graph[3][graph[2]]).view(-1, 1).float()

            self._graphs.append(g)
            labels.append(label)
        self._labels = torch.tensor(labels).float()

    def __getitem__(self, idx):
        """ Get graph and label by index

        Parameters
        ----------
        idx : int
            Item index

        Returns
        -------
        (dgl.DGLGraph, Tensor)
        """
        return self._graphs[idx], self._labels[idx]

    def __len__(self):
        """Number of graphs in the dataset"""
        return len(self._graphs)
