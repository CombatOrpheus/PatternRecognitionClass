import argparse
import shutil
import tempfile
import typing
from pathlib import Path
from typing import List, Tuple, Dict, Any

import lightning.pytorch as pl
import optuna
import torch
from lightning.pytorch.callbacks import EarlyStopping, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from optuna.integration import PyTorchLightningPruningCallback
from torch_geometric.data import Data
from tqdm import tqdm

from src.HomogeneousModels import GraphGNN_SPN_Model, NodeGNN_SPN_Model
from src.PetriNets import load_spn_data_from_files, SPNAnalysisResultLabel
from src.SPNDataModule import SPNDataModule

# Set a seed for reproducibility of data splits, not for trial-to-trial weights
pl.seed_everything(42, workers=True)


def get_args():
    """Parses command-line arguments for fixed settings during optimization."""
    parser = argparse.ArgumentParser(description="Optimize GNN model hyperparameters using Optuna.")

    # Arguments that are fixed for the entire optimization study
    data_group = parser.add_argument_group("Paths and Data")
    data_group.add_argument("--train_file", type=Path, default=Path("Data/GridData_DS1_train_data.processed"))
    data_group.add_argument("--val_file", type=Path, default=Path("Data/GridData_DS1_test_data.processed"))
    data_group.add_argument(
        "--label", type=str, default="average_tokens_per_place", choices=typing.get_args(SPNAnalysisResultLabel)
    )

    model_group = parser.add_argument_group("Model Settings")
    model_group.add_argument("--prediction_level", type=str, default="node", choices=["node", "graph"])
    model_group.add_argument(
        "--gnn_operator",
        type=str,
        default="gcn",
        choices=["gcn", "tag", "cheb", "sgc", "ssg"],
        help="The GNN operator to optimize.",
    )
    model_group.add_argument(
        "--all", action="store_true", help="If specified, runs optimization for all available GNN operators."
    )

    opt_group = parser.add_argument_group("Optimization Settings")
    opt_group.add_argument("--n_trials", type=int, default=100, help="Number of optimization trials.")
    opt_group.add_argument(
        "--timeout", type=int, default=3600 * 2, help="Timeout for the study in seconds (e.g., 2 hours)."
    )
    opt_group.add_argument(
        "--study_name", type=str, default="gnn_spn_optimization", help="Base name for the Optuna study."
    )
    opt_group.add_argument(
        "--storage_dir", type=Path, default=Path("optuna_studies"), help="Directory to save Optuna study databases."
    )

    training_group = parser.add_argument_group("Fixed Training Hyperparameters")
    training_group.add_argument("--max_epochs", type=int, default=100)
    training_group.add_argument("--patience", type=int, default=5, help="Patience for early stopping in each trial.")
    training_group.add_argument("--num_workers", type=int, default=3)

    return parser.parse_args()


def prepare_and_cache_data(args: argparse.Namespace) -> Tuple[List[Data], List[Data], str]:
    """
    Loads, processes, and caches the training and validation data.
    """
    print("--- Preparing and Caching Data ---")
    cache_dir = tempfile.mkdtemp()
    print(f"Cache directory created at: {cache_dir}")

    raw_train_data = load_spn_data_from_files(args.train_file)
    raw_val_data = load_spn_data_from_files(args.val_file)

    temp_dm = SPNDataModule(
        label_to_predict=args.label,
        train_data_list=raw_train_data,
        val_data_list=raw_val_data,
        batch_size=128,
    )
    temp_dm.setup("fit")

    torch.save(temp_dm.train_data, Path(cache_dir) / "train_data.pt")
    torch.save(temp_dm.val_data, Path(cache_dir) / "val_data.pt")
    print("Processed data has been cached.")

    return temp_dm.train_data, temp_dm.val_data, cache_dir


def objective(
    trial: optuna.Trial,
    args: argparse.Namespace,
    processed_train_data: List[Data],
    processed_val_data: List[Data],
) -> float:
    """The Optuna objective function, using pre-processed data."""
    gnn_operator = args.gnn_operator

    # Define the hyperparameter search space
    hyperparams = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [16, 32, 64, 128, 256]),
        "num_layers_gnn": trial.suggest_int("num_layers_gnn", 2, 20),
        "num_layers_mlp": trial.suggest_int("num_layers_mlp", 1, 5),
        "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024, 2048, 4096]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "gnn_k_hops": trial.suggest_int("gnn_k_hops", 2, 8) if gnn_operator in ["tag", "cheb", "sgc", "ssg"] else 3,
        "gnn_alpha": trial.suggest_float("gnn_alpha", 0.05, 0.5) if gnn_operator == "ssg" else 0.1,
    }

    data_module = SPNDataModule(
        label_to_predict=args.label,
        train_data_list=[],
        val_data_list=[],
        batch_size=hyperparams["batch_size"],
        num_workers=args.num_workers,
    )
    data_module.train_data = processed_train_data
    data_module.val_data = processed_val_data

    model_classes: Dict[str, Any] = {
        "node": NodeGNN_SPN_Model,
        "graph": GraphGNN_SPN_Model,
    }
    model_class = model_classes[args.prediction_level]
    node_features_dim = processed_train_data[0].num_node_features

    model = model_class(
        node_features_dim=node_features_dim,
        out_channels=1,
        hidden_dim=hyperparams["hidden_dim"],
        num_layers=hyperparams["num_layers_gnn"],
        num_layers_mlp=hyperparams["num_layers_mlp"],
        learning_rate=hyperparams["learning_rate"],
        weight_decay=hyperparams["weight_decay"],
        gnn_operator_name=gnn_operator,
        gnn_k_hops=hyperparams["gnn_k_hops"],
        gnn_alpha=hyperparams["gnn_alpha"],
    )

    logger = TensorBoardLogger(
        save_dir="optuna_logs", name=f"{args.study_name}_{gnn_operator}", version=f"trial_{trial.number}"
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=[
            EarlyStopping(monitor="val/loss", patience=args.patience, mode="min"),
            PyTorchLightningPruningCallback(trial, monitor="val/loss"),
            TQDMProgressBar(),
        ],
        enable_model_summary=False,
        log_every_n_steps=5,
    )

    try:
        trainer.fit(model, datamodule=data_module)
    except (RuntimeError, ValueError) as e:
        print(f"Trial {trial.number} failed with error: {e}")
        raise optuna.exceptions.TrialPruned()

    return trainer.callback_metrics["val/loss"].item()


def main():
    """Main function to set up and run the hyperparameter optimization study."""
    args = get_args()
    args.storage_dir.mkdir(parents=True, exist_ok=True)

    all_gnn_operators = ["gcn", "tag", "cheb", "sgc", "ssg"]
    operators_to_run = all_gnn_operators if args.all else [args.gnn_operator]

    if args.all:
        print(f"--- Running optimization for all {len(operators_to_run)} GNN operators ---")

    cache_dir = None
    try:
        processed_train_data, processed_val_data, cache_dir = prepare_and_cache_data(args)

        for gnn_operator in tqdm(operators_to_run, desc="Optimizing GNN Operators"):
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

            study.set_user_attr("train_file", str(run_args.train_file))
            study.set_user_attr("val_file", str(run_args.val_file))
            study.set_user_attr("label", run_args.label)
            study.set_user_attr("prediction_level", run_args.prediction_level)
            study.set_user_attr("gnn_operator", run_args.gnn_operator)

            print(f"\n--- Starting Optuna study '{study_name}' for operator '{run_args.gnn_operator}' ---")
            study.optimize(
                lambda trial: objective(trial, run_args, processed_train_data, processed_val_data),
                n_trials=run_args.n_trials,
                timeout=run_args.timeout,
                n_jobs=1,
            )

            print(f"\n--- Optimization Finished for operator '{run_args.gnn_operator}' ---")
            pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
            complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

            print("Study statistics: ")
            print(f"  Number of finished trials: {len(study.trials)}")
            print(f"  Number of pruned trials: {len(pruned_trials)}")
            print(f"  Number of complete trials: {len(complete_trials)}")

            print("Best trial:")
            best_trial = study.best_trial
            print(f"  Value (val/loss): {best_trial.value:.4f}")
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
