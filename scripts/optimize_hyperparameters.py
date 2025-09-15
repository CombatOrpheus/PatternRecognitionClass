"""This script runs a hyperparameter optimization study using Optuna.

It iterates through a list of GNN operators specified in the configuration,
creating and running a separate Optuna study for each. The script handles
data loading, preprocessing, and splitting once before all studies to improve
efficiency. Each study's results are saved to a unique SQLite database.
"""
import argparse
import logging
import shutil
from pathlib import Path
from typing import List

import lightning.pytorch as pl
import optuna
import torch
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from optuna.integration import PyTorchLightningPruningCallback
from sklearn.preprocessing import StandardScaler
from torch.utils.data import random_split, Subset
from tqdm import tqdm

from src.HomogeneousModels import GraphGNN_SPN_Model, NodeGNN_SPN_Model, MixedGNN_SPN_Model
from src.SPNDataModule import SPNDataModule
from src.SPNDatasets import HomogeneousSPNDataset
from src.config_utils import load_config
from src.name_utils import generate_experiment_name

pl.seed_everything(42, workers=True)
# Suppress verbose hardware information from PyTorch Lightning
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.ERROR)


def objective(
    trial: optuna.Trial,
    config: argparse.Namespace,
    train_dataset: Subset,
    val_dataset: Subset,
    label_scaler: StandardScaler,
    study_name: str,
) -> float:
    """The Optuna objective function to be minimized.

    This function defines the hyperparameter search space, creates a model and
    a data module, and trains the model using PyTorch Lightning. It returns the
    validation loss, which Optuna uses to guide the search.

    Args:
        trial: An Optuna Trial object.
        config: The configuration namespace.
        train_dataset: The training dataset subset.
        val_dataset: The validation dataset subset.
        label_scaler: The pre-fitted scaler for labels.
        study_name: The name of the Optuna study for logging purposes.

    Returns:
        The validation loss of the trained model.

    Raises:
        optuna.exceptions.TrialPruned: If the trial is pruned by the callback
            or if an error occurs during training.
    """
    gnn_operator = config.gnn_operator
    hyperparams = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [8, 16, 32, 64, 128, 256]),
        "num_layers_mlp": trial.suggest_int("num_layers_mlp", 1, 5),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512, 1024]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
    }

    if gnn_operator == "mixed":
        hyperparams["heads"] = trial.suggest_int("heads", 2, 8)
    else:
        hyperparams["num_layers_gnn"] = trial.suggest_int("num_layers_gnn", 2, 10)
        hyperparams["gnn_k_hops"] = (
            trial.suggest_int("gnn_k_hops", 2, 8) if gnn_operator in ["tag", "cheb", "sgc", "ssg"] else 3
        )
        hyperparams["gnn_alpha"] = trial.suggest_float("gnn_alpha", 0.05, 0.5) if gnn_operator == "ssg" else 0.1

    data_module = SPNDataModule(
        train_data_list=train_dataset,
        val_data_list=val_dataset,
        label_to_predict=config.label,
        batch_size=hyperparams["batch_size"],
        num_workers=config.num_workers,
        label_scaler=label_scaler,
    )
    data_module.setup("fit")  # This will now use the pre-fitted scaler

    model_params = hyperparams.copy()
    model_params.pop("batch_size")

    if gnn_operator == "mixed":
        if config.prediction_level != "graph":
            print("Warning: 'mixed' operator only supports 'graph' prediction level. Pruning trial.")
            raise optuna.exceptions.TrialPruned()

        # Parameters for MixedGNN_SPN_Model
        model_params.pop("num_layers_gnn", None)
        model_params.pop("gnn_k_hops", None)
        model_params.pop("gnn_alpha", None)
        model = MixedGNN_SPN_Model(node_features_dim=data_module.num_node_features, out_channels=1, **model_params)
    else:
        # Parameters for standard GNN models (GraphGNN or NodeGNN)
        model_params.pop("heads", None)
        model_params["num_layers"] = model_params.pop("num_layers_gnn")
        model_params["gnn_operator_name"] = gnn_operator

        model_class = {"graph": GraphGNN_SPN_Model, "node": NodeGNN_SPN_Model}[config.prediction_level]
        model = model_class(
            node_features_dim=data_module.num_node_features,
            out_channels=1,
            **model_params,
        )

    logger = TensorBoardLogger(save_dir="optuna_logs", name=study_name, version=f"trial_{trial.number}")
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=[
            EarlyStopping(monitor="val/loss", patience=config.patience, mode="min", verbose=False),
            PyTorchLightningPruningCallback(trial, monitor="val/loss"),
        ],
        enable_model_summary=False,
        enable_progress_bar=True,
        log_every_n_steps=1,
    )

    try:
        trainer.fit(model, datamodule=data_module)
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        raise optuna.exceptions.TrialPruned()

    return trainer.callback_metrics["val/loss"].item()


def main(config: argparse.Namespace, config_path: Path):
    """Runs the main hyperparameter optimization loop.

    This function pre-loads and processes the dataset, then iterates through the
    specified GNN operators, running an Optuna study for each one.

    Args:
        config: The main configuration namespace.
        config_path: The path to the configuration file for reproducibility.
    """
    config.io.studies_dir.mkdir(parents=True, exist_ok=True)
    operators_to_run = config.model.gnn_operator

    print("--- Pre-loading and processing data ---")
    train_dataset = HomogeneousSPNDataset(
        root=str(config.io.root),
        raw_data_dir=str(config.io.raw_data_dir),
        raw_file_name=str(config.io.train_file),
        label_to_predict=config.model.label,
    )

    # Perform train/val split and fit scaler ONCE before the loop
    train_size = int(len(train_dataset) * (1 - config.training.val_split))
    val_size = len(train_dataset) - train_size
    train_split, val_split = random_split(list(train_dataset), [train_size, val_size])

    labels = torch.cat([data.y for data in train_split]).numpy().reshape(-1, 1)
    label_scaler = StandardScaler().fit(labels)

    print("Data loaded successfully.")

    base_exp_name = generate_experiment_name(config.io.train_file, config.io.test_file, config.model.label)

    for gnn_operator in tqdm(operators_to_run, desc="Total Optimization Progress"):
        # Create a copy of the config to avoid modification across loops
        run_config = argparse.Namespace(
            **vars(config.io), **vars(config.model), **vars(config.training), **vars(config.optimization)
        )
        run_config.gnn_operator = gnn_operator
        if gnn_operator == "mixed":
            run_config.prediction_level = "graph"
        study_name = f"{base_exp_name}-{gnn_operator}"
        storage_name = f"sqlite:///{run_config.studies_dir / study_name}.db"

        # --- Save config for reproducibility ---
        config_save_path = run_config.studies_dir / f"{study_name}_config.yaml"
        if not config_save_path.exists():
            # Assuming load_config() returns an object with a 'config_path' attribute
            shutil.copy(config_path, config_save_path)
            print(f"Saved configuration for '{study_name}' to {config_save_path}")

        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(),
            load_if_exists=True,
        )

        # Create a dictionary from the run_config namespace for user attributes
        user_attrs = vars(run_config)
        for key, value in user_attrs.items():
            if not isinstance(value, Path):
                study.set_user_attr(key, value)

        study.optimize(
            lambda trial: objective(trial, run_config, train_split, val_split, label_scaler, study_name),
            n_trials=run_config.n_trials,
            timeout=run_config.timeout,
            show_progress_bar=False,
        )

        print(f"\n--- Optimization Finished for operator '{run_config.gnn_operator}' ---")
        print("Best trial:")
        best_trial = study.best_trial
        print(f"  Value (val/loss): {best_trial.value:.4f}")
        print("  Params: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
        print("-" * 80)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    config, config_path = load_config()
    main(config, config_path)
