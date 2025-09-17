from pathlib import Path

import pytest

from src.name_utils import (
    generate_experiment_name,
    generate_plot_name,
    get_dataset_base_name,
)


@pytest.mark.parametrize(
    "file_path, expected_base_name",
    [
        ("Data/GridData_DS1_train_data.processed", "GridData_DS1"),
        ("Data/RandData_DS2_test_data.processed", "RandData_DS2"),
        ("Another/Path/SomeData_Set3_all.processed", "SomeData_Set3"),
    ],
)
def test_get_dataset_base_name(file_path, expected_base_name):
    """Tests that the dataset base name is extracted correctly."""
    assert get_dataset_base_name(Path(file_path)) == expected_base_name


def test_generate_experiment_name():
    """Tests the generation of a unique experiment name."""
    train_file = Path("Data/GridData_DS1_train_data.processed")
    test_file = Path("Data/GridData_DS2_test_data.processed")
    label = "some_property"
    expected_name = "GridData_DS1-GridData_DS2-some_property"
    assert generate_experiment_name(train_file, test_file, label) == expected_name


@pytest.mark.parametrize(
    "plot_type, metric, file_format, expected_name",
    [
        ("critical_difference", "accuracy", "svg", "critical_difference_accuracy.svg"),
        ("performance_heatmap", "roc/auc", "png", "performance_heatmap_roc_auc.png"),
        ("complexity_plot", "mse", "svg", "complexity_plot_mse.svg"),
    ],
)
def test_generate_plot_name(plot_type, metric, file_format, expected_name):
    """Tests the generation of a descriptive plot name."""
    assert generate_plot_name(plot_type, metric, file_format) == expected_name
