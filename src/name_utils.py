"""
Utilities for consistently naming experiments, studies, and artifacts.
"""

from pathlib import Path


def get_dataset_base_name(file_path: Path) -> str:
    """
    Extracts the base name from a dataset file name.

    Example:
        >>> get_dataset_base_name(Path("Data/GridData_DS1_train_data.processed"))
        'GridData_DS1'
    """
    return "_".join(file_path.stem.split("_")[:2])


def generate_experiment_name(train_file: Path, test_file: Path, label: str) -> str:
    """
    Generates a unique and descriptive name for an experiment run.

    Args:
        train_file: Path to the training data file.
        test_file: Path to the testing data file.
        label: The prediction target (label).

    Returns:
        A string representing the experiment name.
    """
    train_base = get_dataset_base_name(train_file)
    test_base = get_dataset_base_name(test_file)
    return f"{train_base}-{test_base}-{label}"
