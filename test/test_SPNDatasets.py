from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data, HeteroData

from src.SPNDatasets import (
    HomogeneousSPNDataset,
    HeterogeneousSPNDataset,
    ReachabilityGraphInMemoryDataset,
)


@pytest.fixture
def data_dir(tmp_path):
    """Create a temporary directory for raw and processed data."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    processed_dir.mkdir()
    # Copy sample data to the temp raw directory
    sample_data = Path("test/sample.processed")
    (raw_dir / sample_data.name).write_text(sample_data.read_text())
    return {"root": tmp_path, "raw_dir": raw_dir, "raw_file_name": sample_data.name}


def test_homogeneous_spn_dataset(data_dir):
    """Tests the creation and processing of a HomogeneousSPNDataset."""
    dataset = HomogeneousSPNDataset(
        root=data_dir["root"],
        raw_data_dir=data_dir["raw_dir"],
        raw_file_name=data_dir["raw_file_name"],
        label_to_predict="average_tokens_network",
    )
    assert len(dataset) > 0
    assert isinstance(dataset[0], Data)
    assert "x" in dataset[0]
    assert "edge_index" in dataset[0]
    assert "y" in dataset[0]
    assert dataset[0].num_node_features == 4


def test_heterogeneous_spn_dataset_graph_label(data_dir):
    """Tests the creation of a HeterogeneousSPNDataset with graph-level labels."""
    dataset = HeterogeneousSPNDataset(
        root=data_dir["root"],
        raw_data_dir=data_dir["raw_dir"],
        raw_file_name=data_dir["raw_file_name"],
        label_to_predict="average_tokens_network",
    )
    assert len(dataset) > 0
    assert isinstance(dataset[0], HeteroData)
    assert "place" in dataset[0].node_types
    assert "transition" in dataset[0].node_types
    assert "y" in dataset[0]
    assert "y" not in dataset[0]["place"]


def test_heterogeneous_spn_dataset_node_label(data_dir):
    """Tests the creation of a HeterogeneousSPNDataset with node-level labels."""
    dataset = HeterogeneousSPNDataset(
        root=data_dir["root"],
        raw_data_dir=data_dir["raw_dir"],
        raw_file_name=data_dir["raw_file_name"],
        label_to_predict="average_tokens_per_place",
    )
    assert len(dataset) > 0
    assert isinstance(dataset[0], HeteroData)
    assert "place" in dataset[0].node_types
    assert "transition" in dataset[0].node_types
    assert not hasattr(dataset[0], "y")
    assert "y" in dataset[0]["place"]
    assert dataset[0]["place"].y.shape[0] == dataset[0]["place"].x.shape[0]


def test_reachability_graph_dataset(data_dir):
    """Tests the creation of a ReachabilityGraphInMemoryDataset."""
    dataset = ReachabilityGraphInMemoryDataset(
        root=data_dir["root"],
        raw_data_dir=data_dir["raw_dir"],
        raw_file_name=data_dir["raw_file_name"],
        label_to_predict="average_tokens_network",
    )
    assert len(dataset) > 0
    assert isinstance(dataset[0], Data)
    # Test padding
    max_nodes = max(d.x.shape[1] for d in dataset)
    for d in dataset:
        assert d.x.shape[1] == max_nodes


def test_processed_filename_uniqueness(data_dir):
    """Tests that processed filenames are unique based on label."""
    dataset1 = HomogeneousSPNDataset(
        root=data_dir["root"],
        raw_data_dir=data_dir["raw_dir"],
        raw_file_name=data_dir["raw_file_name"],
        label_to_predict="average_tokens_network",
    )
    dataset2 = HomogeneousSPNDataset(
        root=data_dir["root"],
        raw_data_dir=data_dir["raw_dir"],
        raw_file_name=data_dir["raw_file_name"],
        label_to_predict="average_firing_rates",
    )
    assert dataset1.processed_file_names != dataset2.processed_file_names
