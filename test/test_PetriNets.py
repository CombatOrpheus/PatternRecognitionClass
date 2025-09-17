import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData

from src.PetriNets import (
    SPNData,
    load_spn_data_from_files,
    load_spn_data_lazily,
    to_incidence_matrix,
)


@pytest.fixture
def sample_spn_data_dict():
    """Provides a sample SPN data dictionary from the first line of the
    sample.processed file."""
    with open("test/sample.processed", "r") as f:
        return json.loads(f.readline())


@pytest.fixture
def sample_spn_data(sample_spn_data_dict):
    """Provides an instance of SPNData for testing."""
    return SPNData(sample_spn_data_dict)


def test_spndata_init(sample_spn_data, sample_spn_data_dict):
    """Tests the initialization and validation of the SPNData class."""
    assert sample_spn_data.spn is not None
    assert sample_spn_data.incidence_matrix is not None
    assert sample_spn_data.average_firing_rates is not None
    assert sample_spn_data.average_tokens_network == sample_spn_data_dict["spn_mu"]


def test_spndata_init_missing_keys():
    """Tests that SPNData raises ValueError for missing keys."""
    with pytest.raises(ValueError):
        SPNData({})


def test_to_incidence_matrix(sample_spn_data):
    """Tests the conversion to an incidence matrix."""
    pn = sample_spn_data.spn
    num_transitions = pn.shape[1] // 2
    pre = pn[:, :num_transitions]
    post = pn[:, num_transitions : 2 * num_transitions]
    expected_incidence = post - pre
    np.testing.assert_array_equal(sample_spn_data.incidence_matrix, expected_incidence)


def test_to_incidence_matrix_invalid_input():
    """Tests that to_incidence_matrix raises ValueError for invalid input."""
    with pytest.raises(ValueError):
        to_incidence_matrix(np.array([1, 2, 3]))  # Not 2D
    with pytest.raises(ValueError):
        to_incidence_matrix(np.array([[1, 2, 3, 4]]))  # Even columns


def test_to_information(sample_spn_data):
    """Tests the conversion to a homogeneous graph representation."""
    node_features, edge_features, edge_pairs = sample_spn_data.to_information()
    num_places = sample_spn_data.spn.shape[0]
    num_transitions = sample_spn_data.spn.shape[1] // 2
    num_nodes = num_places + num_transitions

    assert node_features.shape == (num_nodes, 4)
    assert edge_pairs.shape[0] == 2
    assert edge_features.shape[0] == edge_pairs.shape[1]
    assert np.all(node_features[:num_places, 0] == 1)  # Is_Place
    assert np.all(node_features[num_places:, 1] == 1)  # Is_Transition


def test_to_hetero_information(sample_spn_data):
    """Tests the conversion to a PyG HeteroData object."""
    hetero_data = sample_spn_data.to_hetero_information()
    assert isinstance(hetero_data, HeteroData)
    assert "place" in hetero_data.node_types
    assert "transition" in hetero_data.node_types
    assert ("place", "to", "transition") in hetero_data.edge_types
    assert ("transition", "to", "place") in hetero_data.edge_types
    assert torch.all(hetero_data["place"].node_type == 0)
    assert torch.all(hetero_data["transition"].node_type == 1)


def test_get_analysis_result(sample_spn_data):
    """Tests fetching analysis results by label."""
    avg_firing_rates = sample_spn_data.get_analysis_result("average_firing_rates")
    np.testing.assert_array_equal(avg_firing_rates, sample_spn_data.average_firing_rates)

    with pytest.raises(AttributeError):
        sample_spn_data.get_analysis_result("non_existent_label")


def test_load_spn_data_from_files():
    """Tests eager loading of SPN data from a file."""
    file_path = Path("test/sample.processed")
    spn_data_list = load_spn_data_from_files(file_path)
    with open(file_path, "r") as f:
        num_lines = len(f.readlines())
    assert len(spn_data_list) == num_lines
    assert all(isinstance(spn, SPNData) for spn in spn_data_list)


def test_load_spn_data_lazily():
    """Tests lazy loading of SPN data from a file."""
    file_path = Path("test/sample.processed")
    spn_data_iterator = load_spn_data_lazily(file_path)
    with open(file_path, "r") as f:
        num_lines = len(f.readlines())
    assert sum(1 for _ in spn_data_iterator) == num_lines


def test_load_spn_data_file_not_found():
    """Tests that loading functions raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_spn_data_from_files(Path("non_existent_file.processed"))
    with pytest.raises(FileNotFoundError):
        # We need to consume the generator to trigger the error
        list(load_spn_data_lazily(Path("non_existent_file.processed")))
