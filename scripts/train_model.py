"""This script orchestrates the model training and evaluation pipeline.

It identifies completed Optuna hyperparameter optimization studies, loads the
best parameters for each, and then runs a specified number of training runs
for each model configuration to ensure statistical significance.

For each run, it:
1.  Seeds everything for reproducibility.
2.  Initializes the model with the best hyperparameters.
3.  Trains the model with early stopping.
4.  Saves the best model checkpoint.
5.  Evaluates the model on the test set.
6.  Performs cross-validation against a broader set of datasets.
7.  Aggregates and saves all statistical and cross-validation results.
"""
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import lightning.pytorch as pl
import optuna
import pandas as pd
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from tqdm import tqdm

from src.CrossValidation import CrossValidator
from src.HomogeneousModels import BaseGNN_SPN_Model, GraphGNN_SPN_Model, MixedGNN_SPN_Model, NodeGNN_SPN_Model
from src.SPNDataModule import SPNDataModule
from src.SPNDatasets import HomogeneousSPNDataset
from src.config_utils import load_config
from src.name_utils import generate_experiment_name

# Base seed for ensuring reproducibility of statistical runs
BASE_SEED = 42
pl.seed_everything(BASE_SEED, workers=True)


def load_params_from_study(study_db_path: Path) -> dict:
    """Loads the best hyperparameters and user attributes from an Optuna study.

    Args:
        study_db_path: The path to the Optuna study's SQLite database file.

    Returns:
        A dictionary containing the best trial's parameters and study-level
        user attributes.
    """
    storage_url = f"sqlite:///{study_db_path}"
    study = optuna.load_study(study_name=None, storage=storage_url)
    return {**study.best_trial.params, **study.user_attrs}


def setup_model(run_config: argparse.Namespace, node_features_dim: int) -> BaseGNN_SPN_Model:
    """Instantiates a GNN model based on the provided configuration.

    Args:
        run_config: A namespace containing the model's configuration and
            hyperparameters.
        node_features_dim: The dimensionality of the input node features.

    Returns:
        An instantiated PyTorch Lightning GNN model.
    """
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
    run_config: argparse.Namespace, run_id: int, data_module: SPNDataModule, exp_name: str
) -> Tuple[pl.LightningModule, Dict]:
    """Trains and evaluates a single model instance.

    This function handles the complete lifecycle for one training run: setting
    the seed, initializing the model and trainer, training with early stopping,
    saving the best checkpoint, and running the final test evaluation.

    Args:
        run_config: The configuration for this specific run.
        run_id: The identifier for this run (used for seeding).
        data_module: The configured SPNDataModule.
        exp_name: The name of the experiment for logging.

    Returns:
        A tuple containing:
        - The trained model with the best weights loaded.
        - A dictionary of test results and metadata.

    Raises:
        FileNotFoundError: If the best model checkpoint cannot be found after training.
    """
    seed = BASE_SEED + run_id
    pl.seed_everything(seed, workers=True)

    model = setup_model(run_config, data_module.num_node_features)

    logger = pl.loggers.TensorBoardLogger(
        save_dir=str(run_config.log_dir),
        name=exp_name,
        version=f"{run_config.gnn_operator_name}_run_{run_id}",
    )

    # Early stopping will restore the best model weights at the end of training
    early_stopping_callback = EarlyStopping(monitor="val/loss", patience=run_config.patience, mode="min")
    checkpoint_callback = ModelCheckpoint(monitor="val/loss", mode="min", filename="best")

    trainer = pl.Trainer(
        max_epochs=run_config.max_epochs,
        accelerator="auto",
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=1,
    )
    trainer.fit(model, datamodule=data_module)

    best_ckpt_path = trainer.checkpoint_callback.best_model_path
    if not best_ckpt_path or not Path(best_ckpt_path).exists():
        raise FileNotFoundError("Best model checkpoint not found.")

    # --- Save model checkpoint for persistence ---
    artifact_dir = (
        run_config.state_dict_dir / exp_name / f"{run_config.gnn_operator_name}_run_{run_id}_seed_{seed}"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    final_ckpt_path = artifact_dir / "best_model.ckpt"
    Path(best_ckpt_path).rename(final_ckpt_path)

    # The `model` object has been updated in-place by the trainer and has the best weights.
    # Capture validation metrics right after fitting
    val_metrics = {
        "final_val_loss": trainer.callback_metrics["val/loss"].item(),
        "val/rmse": trainer.callback_metrics["val/rmse"].item(),
        "val/mae": trainer.callback_metrics["val/mae"].item(),
    }

    results = trainer.test(model=model, datamodule=data_module, verbose=False)[0]
    results.update(
        {
            "run_id": run_id,
            "seed": seed,
            "final_train_loss": trainer.callback_metrics.get("train/loss_epoch"),
        }
    )
    results.update(val_metrics)
    return model, results


def main(config: argparse.Namespace):
    """Main function to orchestrate the training and evaluation workflow.

    It finds completed Optuna studies, then for each, it runs multiple
    training instances to gather statistical results and performs cross-validation.

    Args:
        config: The main configuration namespace.
    """
    cross_validator = CrossValidator(config)

    studies_dir = config.io.studies_dir

    # Construct the search pattern based on the current configuration
    exp_name = generate_experiment_name(config.io.train_file, config.io.test_file, config.model.label)
    search_pattern = f"{exp_name}-*.db"

    # Find all studies matching the pattern
    all_matching_studies = sorted(list(studies_dir.glob(search_pattern)))

    # Filter studies to only include those with operators specified in the config
    selected_studies = [
        study_path
        for study_path in all_matching_studies
        if study_path.stem.split("-")[-1] in config.model.gnn_operator
    ]

    if not selected_studies:
        print(f"No Optuna studies found matching the current configuration in '{studies_dir}'.")
        print(f"  (Searched for pattern: '{search_pattern}' with operators: {config.model.gnn_operator})")
        return

    print("\nFound and training the following studies:")
    for study_path in selected_studies:
        print(f"  - {study_path.name}")

    all_stats_results = []
    all_cv_results = []

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
            str(run_config.root), str(run_config.raw_data_dir), str(run_config.train_file), run_config.label
        )
        test_dataset = HomogeneousSPNDataset(
            str(run_config.root), str(run_config.raw_data_dir), str(run_config.test_file), run_config.label
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

            trained_model, stats_result = run_single_training_run(
                run_config, i, data_module, exp_name
            )

            # --- Cross-validation ---
            cv_results_for_model = cross_validator.cross_validate_single_model(
                trained_model, run_config, i, seed
            )
            all_cv_results.extend(cv_results_for_model)

            hparams_to_save = vars(run_config).copy()
            hparams_to_save.update({"total_parameters": total_params, "trainable_parameters": trainable_params})

            stats_result.update(hparams_to_save)
            all_stats_results.append(stats_result)

    # --- Final saving of statistical results ---
    if all_stats_results:
        for r in all_stats_results:
            for k, v in r.items():
                if isinstance(v, Path):
                    r[k] = str(v)

        config.io.stats_results_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_stats_results).to_parquet(config.io.stats_results_file, index=False)
        print(f"\nStatistical results saved to {config.io.stats_results_file}")

    # --- Final saving of cross-validation results ---
    if all_cv_results:
        for r in all_cv_results:
            for k, v in r.items():
                if isinstance(v, Path):
                    r[k] = str(v)

        cv_results_file = config.io.cross_eval_results_file
        cv_results_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_cv_results).to_parquet(cv_results_file, index=False)
        print(f"\nCross-validation results saved to {cv_results_file}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    config, _ = load_config()
    main(config)
