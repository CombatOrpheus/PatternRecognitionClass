import argparse
import json
from pathlib import Path
from typing import List

import lightning.pytorch as pl
import optuna
import pandas as pd
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from tqdm import tqdm

from src.HomogeneousModels import BaseGNN_SPN_Model, GraphGNN_SPN_Model, MixedGNN_SPN_Model, NodeGNN_SPN_Model
from src.SPNDataModule import SPNDataModule
from src.SPNDatasets import HomogeneousSPNDataset
from src.config_utils import load_config

# Base seed for ensuring reproducibility of statistical runs
BASE_SEED = 42
pl.seed_everything(BASE_SEED, workers=True)


def _convert_namespace_to_dict(data):
    """
    Recursively converts objects to make them serializable for parquet.
    - Converts Path objects to strings.
    - Converts namespace-like objects (e.g., argparse.Namespace) to dictionaries.
    - Extracts single-item tensor values.
    """
    if isinstance(data, Path):
        return str(data)
    if isinstance(data, torch.Tensor) and data.numel() == 1:
        return data.item()

    if isinstance(data, argparse.Namespace):
        return {k: _convert_namespace_to_dict(v) for k, v in vars(data).items()}

    if hasattr(data, "__dict__") and not isinstance(data, dict):
        return {k: _convert_namespace_to_dict(v) for k, v in vars(data).items()}

    if isinstance(data, dict):
        return {key: _convert_namespace_to_dict(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [_convert_namespace_to_dict(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(_convert_namespace_to_dict(item) for item in data)

    return data


def get_dataset_base_name(file_name: str) -> str:
    """Extracts the base name from a dataset file name."""
    return "_".join(Path(file_name).stem.split("_")[:2])


def load_params_from_study(study_db_path: Path) -> dict:
    """Loads the best hyperparameters from an Optuna study."""
    storage_url = f"sqlite:///{study_db_path}"
    study = optuna.load_study(study_name=None, storage=storage_url)
    return {**study.best_trial.params, **study.user_attrs}


def setup_model(run_config: argparse.Namespace, node_features_dim: int) -> BaseGNN_SPN_Model:
    """Instantiates the model based on the provided arguments."""
    model_classes = {"node": NodeGNN_SPN_Model, "graph": GraphGNN_SPN_Model, "mixed": MixedGNN_SPN_Model}
    model_class = model_classes.get(
        run_config.gnn_operator_name if run_config.gnn_operator_name == "mixed" else run_config.prediction_level
    )

    model_kwargs = vars(run_config).copy()

    if "num_layers_gnn" in model_kwargs:
        model_kwargs["num_layers"] = model_kwargs.pop("num_layers_gnn")

    accepted_args = model_class.__init__.__code__.co_varnames
    filtered_kwargs = {k: v for k, v in model_kwargs.items() if k in accepted_args}

    return model_class(node_features_dim=node_features_dim, out_channels=1, **filtered_kwargs)


def run_single_training_run(
    run_config: argparse.Namespace,
    run_id: int,
    data_module: SPNDataModule,
    exp_name: str,
    config: argparse.Namespace,
) -> tuple[str, dict, List[dict]]:
    """Trains one model instance, runs cross-val, and returns artifact path and results."""
    seed = BASE_SEED + run_id
    pl.seed_everything(seed, workers=True)

    model = setup_model(run_config, data_module.num_node_features)

    logger = pl.loggers.TensorBoardLogger(
        save_dir=str(run_config.log_dir),
        name=exp_name,
        version=f"{run_config.gnn_operator_name}_run_{run_id}",
    )
    checkpoint_callback = ModelCheckpoint(monitor="val/loss", mode="min", filename="best")

    trainer = pl.Trainer(
        max_epochs=run_config.max_epochs,
        accelerator="auto",
        logger=logger,
        callbacks=[checkpoint_callback, EarlyStopping(monitor="val/loss", patience=run_config.patience)],
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=1,
    )
    trainer.fit(model, datamodule=data_module)

    best_ckpt_path = trainer.checkpoint_callback.best_model_path
    if not best_ckpt_path or not Path(best_ckpt_path).exists():
        raise FileNotFoundError("Best model checkpoint not found.")

    # --- Save model artifacts (checkpoint and hparams) ---
    artifact_dir = run_config.state_dict_dir / exp_name / f"{run_config.gnn_operator_name}_run_{run_id}_seed_{seed}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Instead of extracting the state_dict, we copy the whole checkpoint file for robust loading
    final_ckpt_path = artifact_dir / "best_model.ckpt"
    Path(best_ckpt_path).rename(final_ckpt_path)

    # Test on the primary test set
    results = trainer.test(ckpt_path=str(final_ckpt_path), datamodule=data_module, verbose=False)[0]
    results.update(
        {
            "run_id": run_id,
            "seed": seed,
            "final_train_loss": trainer.callback_metrics.get("train/loss_epoch", torch.tensor(-1.0)).item(),
            "final_val_loss": trainer.callback_metrics.get("val/loss", torch.tensor(-1.0)).item(),
        }
    )

    # Perform cross-validation
    cross_val_results = perform_cross_validation(
        trainer=trainer, data_module=data_module, config=config, run_config=run_config, run_id=run_id
    )

    return str(artifact_dir), results, cross_val_results


def perform_cross_validation(
    trainer: pl.Trainer,
    data_module: SPNDataModule,
    config: argparse.Namespace,
    run_config: argparse.Namespace,
    run_id: int,
) -> List[dict]:
    """
    Performs cross-validation on a trained model against multiple datasets.
    """
    if not hasattr(config.training, "cross_validation") or not config.training.cross_validation.enable:
        return []

    tqdm.write("\n--- Starting Cross-Validation Phase ---")
    cross_val_results = []

    # Determine which datasets to use for cross-validation
    cv_datasets_config = config.training.cross_validation.datasets
    if not cv_datasets_config:  # If empty, use all .processed files in the data directory
        all_files = list(config.io.raw_data_dir.glob("*.processed"))
        # Exclude the main training and test files from the cross-validation set
        exclude_files = {config.io.train_file.name, config.io.test_file.name}
        cv_datasets = [f for f in all_files if f.name not in exclude_files]
    else:
        cv_datasets = [config.io.raw_data_dir / f for f in cv_datasets_config]

    if not cv_datasets:
        tqdm.write("No datasets found for cross-validation.")
        return []

    tqdm.write(f"Found {len(cv_datasets)} datasets for cross-validation.")

    # Use the scaler fitted on the original training data
    original_scaler = data_module.label_scaler
    if not original_scaler:
        tqdm.write("Warning: Label scaler not found in the original data module. Skipping cross-validation.")
        return []

    for data_file in tqdm(cv_datasets, desc="Cross-validating", leave=False):
        try:
            # Create a new dataset and datamodule for the CV dataset
            cv_dataset = HomogeneousSPNDataset(
                root=str(config.io.root),
                raw_data_dir=str(config.io.raw_data_dir),
                raw_file_name=data_file.name,
                label_to_predict=config.model.label,
            )

            if not cv_dataset or len(cv_dataset) == 0:
                tqdm.write(f"  - Skipping empty or invalid dataset: {data_file.name}")
                continue

            cv_data_module = SPNDataModule(
                test_data_list=list(cv_dataset),
                label_to_predict=config.model.label,
                batch_size=run_config.batch_size,
                num_workers=run_config.num_workers,
                label_scaler=original_scaler,  # Pass the original scaler
            )
            cv_data_module.setup("test")

            # Run the test using the model from the trainer
            test_metrics = trainer.test(model=trainer.model, datamodule=cv_data_module, verbose=False)[0]

            # Append results with metadata
            result_dict = test_metrics.copy()
            result_dict["cross_eval_dataset"] = data_file.name
            result_dict["run_id"] = run_id
            result_dict["seed"] = BASE_SEED + run_id
            result_dict["gnn_operator"] = run_config.gnn_operator_name
            cross_val_results.append(result_dict)

        except Exception as e:
            tqdm.write(f"  - Failed to cross-validate on {data_file.name}: {e}")
            continue

    tqdm.write("--- Cross-Validation Phase Complete ---")
    return cross_val_results


def main():
    """Main function to orchestrate the entire experiment workflow."""
    config, _ = load_config()

    studies_dir = config.io.studies_dir

    # Construct the search pattern based on the current configuration
    train_base = get_dataset_base_name(str(config.io.train_file))
    test_base = get_dataset_base_name(str(config.io.test_file))
    label = config.model.label
    exp_name = f"{train_base}-{test_base}-{label}"

    # Define experiment-specific output paths
    stats_file = Path(config.io.stats_results_file)
    cross_eval_file = Path(config.io.cross_eval_results_file)

    search_pattern = f"{exp_name}-*.db"

    # Find all studies matching the pattern
    all_matching_studies = sorted(list(studies_dir.glob(search_pattern)))

    # Filter studies to only include those with operators specified in the config
    selected_studies = [
        study_path for study_path in all_matching_studies if study_path.stem.split("-")[-1] in config.model.gnn_operator
    ]

    if not selected_studies:
        print(f"No Optuna studies found matching the current configuration in '{studies_dir}'.")
        print(f"  (Searched for pattern: '{search_pattern}' with operators: {config.model.gnn_operator})")
        return

    print("\nFound and training the following studies:")
    for study_path in selected_studies:
        print(f"  - {study_path.name}")

    all_stats_results = []
    all_cross_val_results = []

    for study_path in selected_studies:
        study_params = load_params_from_study(study_path)

        # Create a copy of the config to avoid modification across loops
        run_config = argparse.Namespace(
            **vars(config.io), **vars(config.model), **vars(config.training), **vars(config.optimization)
        )
        run_config.__dict__.update(study_params)

        # Rename for consistency with model's __init__
        run_config.gnn_operator_name = run_config.gnn_operator

        print(f"\n--- Training Phase for operator: {run_config.gnn_operator_name} ---")

        train_dataset = HomogeneousSPNDataset(
            root=str(run_config.root),
            raw_data_dir=str(run_config.raw_data_dir),
            raw_file_name=str(run_config.train_file),
            label_to_predict=run_config.label,
        )
        test_dataset = HomogeneousSPNDataset(
            root=str(run_config.root),
            raw_data_dir=str(run_config.raw_data_dir),
            raw_file_name=str(run_config.test_file),
            label_to_predict=run_config.label,
        )

        data_module = SPNDataModule(
            train_data_list=list(train_dataset),
            test_data_list=list(test_dataset),
            label_to_predict=run_config.label,
            batch_size=run_config.batch_size,
            num_workers=run_config.num_workers,
        )
        data_module.setup()  # Setup both fit and test stages

        model_for_summary = setup_model(run_config, data_module.num_node_features)
        total_params = sum(p.numel() for p in model_for_summary.parameters())
        trainable_params = sum(p.numel() for p in model_for_summary.parameters() if p.requires_grad)

        for i in tqdm(range(run_config.num_runs), desc=f"Training {run_config.gnn_operator_name}"):
            seed = BASE_SEED + i
            artifact_dir_path = (
                run_config.state_dict_dir / exp_name / f"{run_config.gnn_operator_name}_run_{i}_seed_{seed}"
            )

            # Check if this run has already been completed
            if (artifact_dir_path / "best_model.ckpt").exists():
                tqdm.write(f"Skipping run {i} for {run_config.gnn_operator_name}: completed run found.")
                continue

            artifact_dir, stats_result, cross_val_run_results = run_single_training_run(
                run_config, i, data_module, exp_name, config
            )
            all_cross_val_results.extend(cross_val_run_results)

            hparams_to_save = vars(run_config).copy()
            hparams_to_save.update({"total_parameters": total_params, "trainable_parameters": trainable_params})

            stats_result.update(hparams_to_save)
            all_stats_results.append(stats_result)

    # --- Final saving of statistical and cross-validation results ---
    if all_stats_results:
        _save_results_to_parquet(all_stats_results, stats_file, "\nStatistical results saved to {}")
    if all_cross_val_results:
        _save_results_to_parquet(all_cross_val_results, cross_eval_file, "Cross-validation results saved to {}")


def _save_results_to_parquet(results: List[dict], file_path: Path, message: str):
    """
    Helper function to save a list of dictionaries to a parquet file.
    It handles the conversion of non-serializable types (e.g., Path, Namespace) to
    a format compatible with pyarrow and creates parent directories if they don't exist.
    """
    # Convert non-serializable objects and create DataFrame
    file_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(_convert_namespace_to_dict(results)).to_parquet(file_path, index=False)
    print(message.format(file_path))


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
