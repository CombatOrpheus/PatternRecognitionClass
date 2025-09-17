import pytest
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data, HeteroData

from src.SPNDataModule import SPNDataModule


@pytest.fixture
def homogeneous_data_list():
    """Creates a list of homogeneous Data objects for testing."""
    return [Data(x=torch.randn(10, 4), edge_index=torch.randint(0, 10, (2, 20)), y=torch.randn(1)) for _ in range(100)]


@pytest.fixture
def heterogeneous_data_list():
    """Creates a list of heterogeneous HeteroData objects for testing."""
    data_list = []
    for _ in range(100):
        data = HeteroData()
        data["place"].x = torch.randn(5, 1)
        data["place"].y = torch.randn(5)
        data["transition"].x = torch.randn(3, 1)
        data["place", "to", "transition"].edge_index = torch.randint(0, 5, (2, 10))
        data_list.append(data)
    return data_list


def test_datamodule_init(homogeneous_data_list):
    """Tests the initialization of the SPNDataModule."""
    dm = SPNDataModule(
        train_data_list=homogeneous_data_list,
        label_to_predict="average_tokens_network",
        batch_size=16,
    )
    assert dm.hparams.batch_size == 16
    assert dm.hparams.label_to_predict == "average_tokens_network"


def test_datamodule_setup_val_split(homogeneous_data_list):
    """Tests the setup method with a validation split."""
    dm = SPNDataModule(
        train_data_list=homogeneous_data_list,
        label_to_predict="average_tokens_network",
        val_split=0.2,
    )
    dm.setup(stage="fit")
    assert len(dm.train_dataset) == 80
    assert len(dm.val_dataset) == 20
    assert dm.label_scaler is not None


def test_datamodule_setup_predefined_val(homogeneous_data_list):
    """Tests the setup method with a predefined validation set."""
    train_list = homogeneous_data_list[:80]
    val_list = homogeneous_data_list[80:]
    dm = SPNDataModule(
        train_data_list=train_list,
        val_data_list=val_list,
        label_to_predict="average_tokens_network",
    )
    dm.setup(stage="fit")
    assert len(dm.train_dataset) == 80
    assert len(dm.val_dataset) == 20


def test_datamodule_label_scaling(homogeneous_data_list):
    """Tests that the label scaler is correctly applied."""
    dm = SPNDataModule(train_data_list=homogeneous_data_list, label_to_predict="average_tokens_network")
    dm.setup(stage="fit")
    assert dm.label_scaler.mean_ is not None
    assert dm.label_scaler.scale_ is not None


def test_datamodule_prefitted_scaler(homogeneous_data_list):
    """Tests using a pre-fitted label scaler."""
    labels = torch.cat([d.y for d in homogeneous_data_list]).numpy().reshape(-1, 1)
    scaler = StandardScaler().fit(labels)
    dm = SPNDataModule(
        train_data_list=homogeneous_data_list,
        label_to_predict="average_tokens_network",
        label_scaler=scaler,
    )
    dm.setup(stage="fit")
    assert dm.label_scaler is scaler


def test_datamodule_heterogeneous(heterogeneous_data_list):
    """Tests the datamodule with heterogeneous data."""
    dm = SPNDataModule(
        train_data_list=heterogeneous_data_list,
        label_to_predict="average_tokens_per_place",
        heterogeneous=True,
    )
    dm.setup(stage="fit")
    assert dm.num_node_features == {"place": 1, "transition": 1}
    assert dm.num_edge_features == {("place", "to", "transition"): 0}
    assert dm.train_dataloader() is not None
    assert dm.val_dataloader() is not None


def test_dataloaders(homogeneous_data_list):
    """Tests that dataloaders are created correctly."""
    dm = SPNDataModule(
        train_data_list=homogeneous_data_list,
        test_data_list=homogeneous_data_list,
        label_to_predict="average_tokens_network",
        batch_size=10,
    )
    dm.setup()
    assert dm.train_dataloader().batch_size == 10
    assert dm.val_dataloader().batch_size == 10
    assert dm.test_dataloader().batch_size == 10
