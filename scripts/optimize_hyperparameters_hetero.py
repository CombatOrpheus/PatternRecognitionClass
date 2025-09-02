import argparse
import shutil
import tempfile
import typing
from pathlib import Path
from typing import List, Tuple

import lightning.pytorch as pl
import optuna
import torch
from lightning.pytorch.callbacks import EarlyStopping, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from optuna.integration import PyTorchLightningPruningCallback
from torch_geometric.data import HeteroData
from tqdm import tqdm

from src.HeterogeneousModels import RGAT_SPN_Model, HEAT_SPN_Model
from src.PetriNets import load_spn_data_from_files, SPNAnalysisResultLabel
from src.SPNDataModule import SPNDataModule

# Set a seed for reproducibility of data splits
pl.seed_everything(42, workers=True)


def get_args():
    """Parses command-line arguments for the optimization."""
    parser = argparse.ArgumentParser(description="Optimize Heterogeneous GNN model hyperparameters using Optuna.")

    data_group = parser.add_argument_group("Paths and Data")
    data_group.add_argument("--train_file", type=Path, default=Path("../Data/GridData_DS1_train_data.processed"))
    data_group.add_argument("--val_file", type=Path, default=Path("../Data/GridData_DS1_test_data.processed"))
    data_group.add_argument(
        "--label", type=str, default="average_tokens_per_place", choices=typing.get_args(SPNAnalysisResultLabel)
    )

    model_group = parser.add_argument_group("Model Settings")
    model_group.add_argument(
        "--gnn_operator",
        type=str,
        default="rgat",
        choices=["rgat", "heat"],
        help="The Heterogeneous GNN operator to optimize.",
    )
    model_group.add_argument(
        "--all", action="store_true", help="If specified, runs optimization for all available operators."
    )

    opt_group = parser.add_argument_group("Optimization Settings")
    opt_group.add_argument("--n_trials", type=int, default=100, help="Number of optimization trials.")
    opt_group.add_argument("--timeout", type=int, default=3600 * 2, help="Timeout for the study in seconds.")
    opt_group.add_argument(
        "--study_name", type=str, default="hetero_gnn_spn_optimization", help="Base name for the Optuna study."
    )
    opt_group.add_argument(
        "--storage_dir", type=Path, default=Path("../optuna_studies"), help="Directory to save Optuna study databases."
    )

    training_group = parser.add_argument_group("Fixed Training Hyperparameters")
    training_group.add_argument("--max_epochs", type=int, default=100)
    training_group.add_argument("--patience", type=int, default=10, help="Patience for early stopping.")
    training_group.add_argument("--num_workers", type=int, default=3)

    return parser.parse_args()


def prepare_and_cache_data(args: argparse.Namespace) -> Tuple[List[HeteroData], List[HeteroData], str]:
    """
    Loads, processes, and caches the training and validation data in heterogeneous format.
    """
    print("--- Preparing and Caching Heterogeneous Data ---")
    cache_dir = tempfile.mkdtemp()
    print(f"Cache directory created at: {cache_dir}")

    raw_train_data = load_spn_data_from_files(args.train_file)
    raw_val_data = load_spn_data_from_files(args.val_file)

    # Use SPNDataModule to process the data once in heterogeneous mode
    temp_dm = SPNDataModule(
        label_to_predict=args.label,
        train_data_list=raw_train_data,
        val_data_list=raw_val_data,
        batch_size=128,  # Placeholder, not used for processing
        heterogeneous=True,
    )
    temp_dm.setup("fit")

    train_cache_path = Path(cache_dir) / "train_data.pt"
    val_cache_path = Path(cache_dir) / "val_data.pt"
    torch.save(temp_dm.train_data, train_cache_path)
    torch.save(temp_dm.val_data, val_cache_path)
    print("Processed heterogeneous data has been cached.")

    return temp_dm.train_data, temp_dm.val_data, cache_dir


def objective(
    trial: optuna.Trial,
    args: argparse.Namespace,
    processed_train_data: List[HeteroData],
    processed_val_data: List[HeteroData],
) -> float:
    """The Optuna objective function for heterogeneous models."""
    gnn_operator = args.gnn_operator

    hyperparams = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "hidden_channels": trial.suggest_categorical("hidden_channels", [16, 32, 64, 128]),
        "num_layers": trial.suggest_int("num_layers", 2, 10),
        "num_heads": trial.suggest_categorical("num_heads", [1, 2, 4, 8]),
        "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024, 2048]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
    }

    data_module = SPNDataModule(
        label_to_predict=args.label,
        train_data_list=[],
        val_data_list=[],
        batch_size=hyperparams["batch_size"],
        num_workers=args.num_workers,
        heterogeneous=True,
    )
    data_module.train_data = processed_train_data
    data_module.val_data = processed_val_data

    if gnn_operator == "rgat":
        model = RGAT_SPN_Model(
            in_channels=data_module.num_node_features,
            out_channels=1,
            edge_dim=data_module.num_edge_features,
            hidden_channels=hyperparams["hidden_channels"],
            num_layers=hyperparams["num_layers"],
            num_heads=hyperparams["num_heads"],
            learning_rate=hyperparams["learning_rate"],
            weight_decay=hyperparams["weight_decay"],
        )
    elif gnn_operator == "heat":
        # For HEAT, we can add more specific hyperparameters
        node_type_emb_dim = trial.suggest_categorical("node_type_emb_dim", [16, 32, 64])
        edge_type_emb_dim = trial.suggest_categorical("edge_type_emb_dim", [16, 32, 64])
        edge_attr_emb_dim = trial.suggest_categorical("edge_attr_emb_dim", [16, 32, 64])
        model = HEAT_SPN_Model(
            in_channels=data_module.num_node_features,
            out_channels=1,
            edge_dim=data_module.num_edge_features,
            hidden_channels=hyperparams["hidden_channels"],
            num_layers=hyperparams["num_layers"],
            num_heads=hyperparams["num_heads"],
            learning_rate=hyperparams["learning_rate"],
            weight_decay=hyperparams["weight_decay"],
            num_node_types=data_module.num_node_types,
            num_edge_types=data_module.num_edge_types,
            node_type_emb_dim=node_type_emb_dim,
            edge_type_emb_dim=edge_type_emb_dim,
            edge_attr_emb_dim=edge_attr_emb_dim,
        )
    else:
        raise ValueError(f"Unsupported GNN operator: {gnn_operator}")

    logger = TensorBoardLogger(
        save_dir="../optuna_logs", name=f"{args.study_name}_{gnn_operator}", version=f"trial_{trial.number}"
    )
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val/loss")
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=args.patience, verbose=False, mode="min")

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=[early_stop_callback, pruning_callback, TQDMProgressBar(refresh_rate=10)],
        enable_model_summary=False,
        log_every_n_steps=5,
    )

    try:
        trainer.fit(model, datamodule=data_module)
        # Use the unified val/loss metric logged by the base module
        return trainer.callback_metrics["val/loss"].item()
    except (RuntimeError, ValueError) as e:
        print(f"Trial {trial.number} failed with error: {e}")
        raise optuna.exceptions.TrialPruned()


def main():
    """Main function to run the optimization study."""
    args = get_args()
    args.storage_dir.mkdir(parents=True, exist_ok=True)

    operators_to_run = ["rgat", "heat"] if args.all else [args.gnn_operator]

    cache_dir = None
    try:
        processed_train_data, processed_val_data, cache_dir = prepare_and_cache_data(args)

        for gnn_operator in tqdm(operators_to_run, desc="Optimizing Operators"):
            run_args = argparse.Namespace(**vars(args))
            run_args.gnn_operator = gnn_operator

            study_name = f"{run_args.study_name}_{run_args.gnn_operator}"
            storage_name = f"sqlite:///{run_args.storage_dir / study_name}.db"

            study = optuna.create_study(
                study_name=study_name,
                storage=storage_name,
                direction="minimize",
                pruner=optuna.pruners.MedianPruner(),
                load_if_exists=True,
            )
            study.set_user_attr("label", run_args.label)
            study.set_user_attr("gnn_operator", run_args.gnn_operator)

            print(f"\n--- Starting Optuna study '{study_name}' for operator '{run_args.gnn_operator}' ---")
            study.optimize(
                lambda trial: objective(trial, run_args, processed_train_data, processed_val_data),
                n_trials=run_args.n_trials,
                timeout=run_args.timeout,
                n_jobs=1,
            )

            print(f"\n--- Optimization Finished for operator '{run_args.gnn_operator}' ---")
            print("Best trial:")
            best_trial = study.best_trial
            print(f"  Value (val/loss): {best_trial.value}")
            print("  Params: ")
            for key, value in best_trial.params.items():
                print(f"    {key}: {value}")
            print("-" * 80)

    finally:
        if cache_dir:
            print(f"\n--- Cleaning up cache directory: {cache_dir} ---")
            shutil.rmtree(cache_dir)
            print("Cache cleaned up successfully.")

    print("\n--- All optimization studies finished. ---")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
