import pytest
import torch
from torch_geometric.data import Data, Batch

from src.HomogeneousModels import (
    GraphGNN_SPN_Model,
    NodeGNN_SPN_Model,
    MixedGNN_SPN_Model,
)


@pytest.fixture
def graph_level_batch():
    """Creates a batch of homogeneous graph data for graph-level prediction."""
    data_list = [
        Data(
            x=torch.randn(10, 4),
            edge_index=torch.randint(0, 10, (2, 20)),
            y=torch.randn(1),
        )
        for _ in range(4)
    ]
    return Batch.from_data_list(data_list)


@pytest.fixture
def node_level_batch():
    """Creates a batch of homogeneous graph data for node-level prediction."""
    data_list = []
    for _ in range(4):
        x = torch.randn(10, 4)
        # Create a mask for places (first column of x is 1 for places)
        place_mask = torch.zeros(10, 1)
        place_mask[::2] = 1
        x[:, 0:1] = place_mask
        data = Data(
            x=x,
            edge_index=torch.randint(0, 10, (2, 20)),
            y=torch.randn(5),  # 5 places per graph
        )
        data_list.append(data)
    return Batch.from_data_list(data_list)


@pytest.mark.parametrize("gnn_operator", ["gcn", "tag", "sgc", "ssg"])
def test_graph_gnn_model_forward(gnn_operator, graph_level_batch):
    """Tests the forward pass of the GraphGNN_SPN_Model."""
    model = GraphGNN_SPN_Model(
        node_features_dim=4,
        hidden_dim=16,
        out_channels=1,
        num_layers=2,
        gnn_operator_name=gnn_operator,
    )
    output = model(graph_level_batch)
    assert output.shape == (graph_level_batch.num_graphs,)


def test_graph_gnn_model_step(graph_level_batch):
    """Tests the training step of the GraphGNN_SPN_Model."""
    model = GraphGNN_SPN_Model(
        node_features_dim=4, hidden_dim=16, out_channels=1, num_layers=2
    )
    loss = model.training_step(graph_level_batch, 0)
    assert torch.is_tensor(loss)
    assert loss > 0


def test_model_compilation():
    """Tests that the model is compiled when to_be_compiled=True."""
    model = GraphGNN_SPN_Model(
        node_features_dim=4,
        hidden_dim=16,
        out_channels=1,
        num_layers=2,
        to_be_compiled=True,
    )
    model.setup(stage="fit")
    assert model.forward.__code__.co_name == "compile_wrapper"


@pytest.mark.parametrize("gnn_operator", ["gcn", "tag", "sgc", "ssg"])
def test_node_gnn_model_forward(gnn_operator, node_level_batch):
    """Tests the forward pass of the NodeGNN_SPN_Model."""
    model = NodeGNN_SPN_Model(
        node_features_dim=4,
        hidden_dim=16,
        out_channels=1,
        num_layers=2,
        gnn_operator_name=gnn_operator,
    )
    output = model(node_level_batch)
    assert output.shape == (node_level_batch.num_nodes,)


def test_node_gnn_model_step(node_level_batch):
    """Tests the training step of the NodeGNN_SPN_Model."""
    model = NodeGNN_SPN_Model(
        node_features_dim=4, hidden_dim=16, out_channels=1, num_layers=2
    )
    loss = model.training_step(node_level_batch, 0)
    assert torch.is_tensor(loss)
    assert loss > 0


def test_mixed_gnn_model_forward(graph_level_batch):
    """Tests the forward pass of the MixedGNN_SPN_Model."""
    model = MixedGNN_SPN_Model(
        node_features_dim=4, hidden_dim=16, out_channels=1
    )
    output = model(graph_level_batch)
    assert output.shape == (graph_level_batch.num_graphs,)


def test_mixed_gnn_model_step(graph_level_batch):
    """Tests the training step of the MixedGNN_SPN_Model."""
    model = MixedGNN_SPN_Model(
        node_features_dim=4, hidden_dim=16, out_channels=1
    )
    loss = model.training_step(graph_level_batch, 0)
    assert torch.is_tensor(loss)
    assert loss > 0
