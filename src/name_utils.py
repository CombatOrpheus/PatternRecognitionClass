"""This module provides utility functions for consistently naming experiments,
studies, and other artifacts within the project.

Using these functions ensures that all generated files and directories follow a
predictable and parseable naming scheme, which is crucial for organization and
reproducibility.
"""

from pathlib import Path


def get_dataset_base_name(file_path: Path) -> str:
    """Extracts the base name from a dataset file path.

    This function assumes the file name is structured like
    `[BaseName]_[...].processed` and extracts the `[BaseName]` part.

    Args:
        file_path: The path to the dataset file.

    Returns:
        The extracted base name of the dataset.

    Example:
        >>> get_dataset_base_name(Path("Data/GridData_DS1_train_data.processed"))
        'GridData_DS1'
    """
    return "_".join(file_path.stem.split("_")[:2])


def generate_experiment_name(train_file: Path, test_file: Path, label: str) -> str:
    """Generates a unique and descriptive name for an experiment run.

    The name is composed of the base names of the training and testing
    datasets, along with the prediction target label.

    Args:
        train_file: The path to the training data file.
        test_file: The path to the testing data file.
        label: The prediction target (label).

    Returns:
        A string representing the unique experiment name.
    """
    train_base = get_dataset_base_name(train_file)
    test_base = get_dataset_base_name(test_file)
    return f"{train_base}-{test_base}-{label}"


def generate_plot_name(plot_type: str, metric: str, file_format: str = "svg") -> str:
    """Generates a descriptive and unique filename for a plot.

    The name includes the type of plot and the metric being visualized,
    ensuring clarity and avoiding naming conflicts.

    Args:
        plot_type: The type of plot (e.g., 'critical_difference', 'performance_vs_complexity').
        metric: The metric being plotted (e.g., 'accuracy', 'roc_auc').
        file_format: The file format for the plot (e.g., 'svg', 'png').

    Returns:
        A string to be used as the plot's filename.
    """
    metric_suffix = metric.replace('/', '_')
    return f"{plot_type}_{metric_suffix}.{file_format}"
