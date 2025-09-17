import pytest
import torch
from torch_geometric.data import HeteroData, Batch

from src.HeterogeneousModels import RGAT_SPN_Model, HEAT_SPN_Model


@pytest.fixture
def hetero_batch():
    """Creates a batch of heterogeneous graph data for testing."""
    data_list = []
    for _ in range(4):
        data = HeteroData()
        data["place"].x = torch.randn(5, 2)
        data["place"].y = torch.randn(5)
        data["transition"].x = torch.randn(3, 2)
        data["place", "to", "transition"].edge_index = torch.randint(0, 5, (2, 10))
        data["transition", "to", "place"].edge_index = torch.randint(0, 3, (2, 8))
        data_list.append(data)
    return Batch.from_data_list(data_list)


@pytest.fixture
def heat_batch():
    """Creates a batch of heterogeneous graph data specifically for the HEAT model."""
    data_list = []
    for _ in range(4):
        data = HeteroData()
        data["place"].x = torch.randn(5, 2)
        data["place"].y = torch.randn(5)
        data["place"].node_type = torch.zeros(5, dtype=torch.long)
        data["transition"].x = torch.randn(3, 2)
        data["transition"].node_type = torch.ones(3, dtype=torch.long)
        data["place", "to", "transition"].edge_index = torch.randint(0, 5, (2, 10))
        data["place", "to", "transition"].edge_type = torch.zeros(10, dtype=torch.long)
        data["place", "to", "transition"].edge_attr = torch.randn(10, 1)
        data["transition", "to", "place"].edge_index = torch.randint(0, 3, (2, 8))
        data["transition", "to", "place"].edge_type = torch.ones(8, dtype=torch.long)
        data["transition", "to", "place"].edge_attr = torch.randn(8, 1)
        data_list.append(data)
    return Batch.from_data_list(data_list)


def test_rgat_model_forward(hetero_batch):
    """Tests the forward pass of the RGAT_SPN_Model."""
    model = RGAT_SPN_Model(
        in_channels_dict={"place": 2, "transition": 2},
        hidden_channels=16,
        out_channels=1,
        num_heads=2,
        num_layers=2,
        edge_dim=0,
        num_relations=2,
    )
    output = model(hetero_batch)
    assert isinstance(output, dict)
    assert "place" in output
    assert "transition" in output
    assert output["place"].shape == (20, 1)
    assert output["transition"].shape == (12, 1)


def test_rgat_model_step(hetero_batch):
    """Tests the training step of the RGAT_SPN_Model."""
    model = RGAT_SPN_Model(
        in_channels_dict={"place": 2, "transition": 2},
        hidden_channels=16,
        out_channels=1,
        num_heads=2,
        num_layers=2,
        edge_dim=0,
        num_relations=2,
    )
    loss = model.training_step(hetero_batch, 0)
    assert torch.is_tensor(loss)
    assert loss > 0


def test_heat_model_forward(heat_batch):
    """Tests the forward pass of the HEAT_SPN_Model."""
    model = HEAT_SPN_Model(
        in_channels_dict={"place": 2, "transition": 2},
        hidden_channels=16,
        out_channels=1,
        num_heads=2,
        num_layers=2,
        edge_dim=1,
        num_node_types=2,
        num_edge_types=2,
        node_type_emb_dim=4,
        edge_type_emb_dim=4,
        edge_attr_emb_dim=4,
    )
    output = model(heat_batch)
    assert isinstance(output, dict)
    assert "place" in output
    assert "transition" in output
    assert output["place"].shape == (20, 1)
    assert output["transition"].shape == (12, 1)


def test_heat_model_step(heat_batch):
    """Tests the training step of the HEAT_SPN_Model."""
    model = HEAT_SPN_Model(
        in_channels_dict={"place": 2, "transition": 2},
        hidden_channels=16,
        out_channels=1,
        num_heads=2,
        num_layers=2,
        edge_dim=1,
        num_node_types=2,
        num_edge_types=2,
        node_type_emb_dim=4,
        edge_type_emb_dim=4,
        edge_attr_emb_dim=4,
    )
    loss = model.training_step(heat_batch, 0)
    assert torch.is_tensor(loss)
    assert loss > 0
