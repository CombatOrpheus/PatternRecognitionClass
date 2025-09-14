import re
from pathlib import Path
import argparse
from typing import TYPE_CHECKING


class PathHandler:
    """
    A centralized handler for constructing and managing file and directory paths for the project.
    """

    def __init__(self, io_config: argparse.Namespace):
        """
        Initializes the PathHandler with the I/O configuration.
        """
        self.io_config = io_config

    @staticmethod
    def get_dataset_base_name(file_path: Path | str) -> str:
        """Extracts the base name from a dataset file name (e.g., 'GridData_DS1')."""
        return "_".join(Path(file_path).stem.split("_")[:2])

    def get_experiment_name(self, train_file: Path | str, test_file: Path | str, label: str) -> str:
        """Constructs the experiment name from dataset names and the label."""
        train_base = self.get_dataset_base_name(train_file)
        test_base = self.get_dataset_base_name(test_file)
        return f"{train_base}-{test_base}-{label}"

    def get_study_db_path(self, experiment_name: str, gnn_operator: str) -> Path:
        """Returns the path to the Optuna study database file."""
        return self.io_config.studies_dir / f"{experiment_name}-{gnn_operator}.db"

    def get_study_config_path(self, study_name: str) -> Path:
        """Returns the path to the saved configuration file for a study."""
        return self.io_config.studies_dir / f"{study_name}_config.yaml"

    def get_study_storage_url(self, study_db_path: Path) -> str:
        """Returns the storage URL for an Optuna study."""
        return f"sqlite:///{study_db_path}"

    def get_artifact_dir(self, experiment_name: str, gnn_operator_name: str, run_id: int, seed: int) -> Path:
        """Constructs the directory path for storing a single run's artifacts."""
        run_dir_name = f"{gnn_operator_name}_run_{run_id}_seed_{seed}"
        return self.io_config.state_dict_dir / experiment_name / run_dir_name

    def get_checkpoint_path(self, artifact_dir: Path) -> Path:
        """Returns the path to the final checkpoint file within an artifact directory."""
        return artifact_dir / "best_model.ckpt"

    def get_tensorboard_logger_dir(self) -> str:
        """Returns the base directory for TensorBoard logs."""
        return str(self.io_config.log_dir)

    def get_stats_results_path(self) -> Path:
        """Returns the path to the statistical results parquet file."""
        return self.io_config.stats_results_file

    def get_cross_eval_results_path(self) -> Path:
        """Returns the path to the cross-evaluation results parquet file."""
        return self.io_config.cross_eval_results_file

    def get_analysis_output_dir(self) -> Path:
        """Returns the directory for saving analysis outputs (plots, summaries)."""
        return self.io_config.output_dir
