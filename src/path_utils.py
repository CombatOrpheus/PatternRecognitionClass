import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config_utils import IOConfig


class PathHandler:
    """
    A centralized handler for constructing and managing file and directory paths for the project.

    This class is designed to be initialized with the I/O configuration object and provides
    a consistent interface for generating paths for datasets, model artifacts, logs, and results.
    """

    def __init__(self, io_config: "IOConfig"):
        """
        Initializes the PathHandler with the I/O configuration.

        Args:
            io_config: The I/O configuration object, typically `config.io`.
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

    def get_study_storage_url(self, study_db_path: Path) -> str:
        """Returns the storage URL for an Optuna study."""
        return f"sqlite:///{study_db_path}"

    def get_study_config_path(self, study_name: str) -> Path:
        """Returns the path to the saved configuration file for a study."""
        return self.io_config.studies_dir / f"{study_name}_config.yaml"

    def find_matching_studies(self, experiment_name: str, gnn_operators: list[str]) -> list[Path]:
        """Finds all Optuna study files matching the experiment name and operators."""
        search_pattern = f"{experiment_name}-*.db"
        all_matching_studies = sorted(list(self.io_config.studies_dir.glob(search_pattern)))
        return [
            study_path
            for study_path in all_matching_studies
            if study_path.stem.split("-")[-1] in gnn_operators
        ]

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

    def find_model_checkpoints(self, experiment_dir: Path | None = None) -> list[Path]:
        """
        Finds all 'best.ckpt' files within an experiment directory.
        If no directory is provided, it uses the configured experiment_dir.
        """
        target_dir = experiment_dir or self.io_config.experiment_dir
        all_ckpts = sorted(list(target_dir.glob("**/checkpoints/best.ckpt")))
        return all_ckpts

    def find_processed_data_files(self, data_dir: Path | None = None) -> list[Path]:
        """Finds all '.processed' data files in a directory."""
        target_dir = data_dir or self.io_config.data_dir
        return sorted(list(target_dir.glob("*.processed")))

    @staticmethod
    def parse_metadata_from_path(ckpt_path: Path) -> dict[str, str | int]:
        """
        Infers metadata such as the experiment name, operator, and run ID from the path structure.
        """
        metadata = {}
        try:
            # e.g., .../state_dicts/GridData_DS1-GridData_DS1-MAE/gcn_run_0_seed_42/best_model.ckpt
            run_dir = ckpt_path.parent
            exp_dir = run_dir.parent

            metadata["experiment_name"] = exp_dir.name

            # e.g., gcn_run_0_seed_42
            run_dir_name = run_dir.name
            match = re.match(r"(.+)_run_(\d+)_seed_(\d+)", run_dir_name)
            if match:
                metadata["gnn_operator"] = match.group(1)
                metadata["run_id"] = int(match.group(2))
                metadata["seed"] = int(match.group(3))
        except (IndexError, AttributeError, ValueError) as e:
            print(f"Warning: Could not parse metadata from path {ckpt_path}. Reason: {e}")
        return metadata
