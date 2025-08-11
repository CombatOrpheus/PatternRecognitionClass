import argparse
import typing
from pathlib import Path
from typing import List

import lightning.pytorch as pl
import optuna
import torch
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from optuna.integration import PyTorchLightningPruningCallback

from src.HomogeneousModels import GraphGNN_SPN_Model, NodeGNN_SPN_Model
from src.PetriNets import load_spn_data_from_files, SPNAnalysisResultLabel, SPNData
from src.SPNDataModule import SPNDataModule

# Set a seed for reproducibility of data splits, not for trial-to-trial weights
pl.seed_everything(42, workers=True)


def get_args():
    """Parses command-line arguments for fixed settings during optimization."""
    parser = argparse.ArgumentParser(description="Optimize GNN model hyperparameters using Optuna.")

    # Arguments that are fixed for the entire optimization study
    data_group = parser.add_argument_group("Paths and Data")
    data_group.add_argument(
        "--train_file",
        type=Path,
        default=Path("Data/GridData_DS1_train_data.processed"),
    )
    data_group.add_argument(
        "--val_file",
        type=Path,
        default=Path("Data/GridData_DS1_test_data.processed"),
    )
    data_group.add_argument(
        "--label",
        type=str,
        default="average_tokens_per_place",
        choices=typing.get_args(SPNAnalysisResultLabel),
    )
    data_group.add_argument("--prediction_level", type=str, default="node", choices=["node", "graph"])

    opt_group = parser.add_argument_group("Optimization Settings")
    opt_group.add_argument("--n_trials", type=int, default=100, help="Number of optimization trials.")
    opt_group.add_argument(
        "--timeout", type=int, default=3600 * 2, help="Timeout for the study in seconds (e.g., 2 hours)."
    )
    opt_group.add_argument("--study_name", type=str, default="gnn_spn_optimization", help="Name for the Optuna study.")
    opt_group.add_argument(
        "--storage", type=str, default="sqlite:///gnn_spn_optimization.db", help="Optuna storage URL."
    )

    training_group = parser.add_argument_group("Fixed Training Hyperparameters")
    training_group.add_argument("--max_epochs", type=int, default=100)
    training_group.add_argument("--patience", type=int, default=10, help="Patience for early stopping in each trial.")
    training_group.add_argument("--num_workers", type=int, default=3)

    return parser.parse_args()


def objective(
    trial: optuna.Trial, args: argparse.Namespace, train_spn_list: List[SPNData], val_spn_list: List[SPNData]
) -> float:
    """The Optuna objective function to be minimized."""
    # --- 1. Suggest Hyperparameters ---
    gnn_operator = trial.suggest_categorical("gnn_operator", ["gcn", "tag", "cheb", "sgc", "ssg"])

    # K-hops is only relevant for some operators
    gnn_k_hops = trial.suggest_int("gnn_k_hops", 2, 8) if gnn_operator in ["tag", "cheb", "sgc", "ssg"] else 3

    # Alpha is only for SSGConv
    gnn_alpha = trial.suggest_float("gnn_alpha", 0.05, 0.5) if gnn_operator == "ssg" else 0.1

    hyperparams = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [16, 32, 64, 128]),
        "num_layers_gnn": trial.suggest_int("num_layers_gnn", 2, 10),
        "num_layers_mlp": trial.suggest_int("num_layers_mlp", 1, 3),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "gnn_operator": gnn_operator,
        "gnn_k_hops": gnn_k_hops,
        "gnn_alpha": gnn_alpha,
    }

    # --- 2. Setup Data and Model ---
    data_module = SPNDataModule(
        label_to_predict=args.label,
        train_data_list=train_spn_list,
        val_data_list=val_spn_list,
        batch_size=hyperparams["batch_size"],
        num_workers=args.num_workers,
    )

    model_class = NodeGNN_SPN_Model if args.prediction_level == "node" else GraphGNN_SPN_Model

    model = model_class(
        node_features_dim=train_spn_list[0].num_node_features,
        out_channels=1,
        hidden_dim=hyperparams["hidden_dim"],
        num_layers=hyperparams["num_layers_gnn"],
        num_layers_mlp=hyperparams["num_layers_mlp"],
        learning_rate=hyperparams["learning_rate"],
        weight_decay=hyperparams["weight_decay"],
        gnn_operator_name=hyperparams["gnn_operator"],
        gnn_k_hops=hyperparams["gnn_k_hops"],
        gnn_alpha=hyperparams["gnn_alpha"],
    )

    # --- 3. Setup Trainer and Callbacks ---
    logger = TensorBoardLogger(save_dir="optuna_logs", name=args.study_name, version=f"trial_{trial.number}")
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=args.patience, verbose=False, mode="min")

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=[early_stop_callback, pruning_callback],
        enable_progress_bar=True,
        enable_model_summary=False,
    )

    # --- 4. Train and Evaluate ---
    try:
        trainer.fit(model, datamodule=data_module)
    except (RuntimeError, ValueError) as e:
        print(f"Trial {trial.number} failed with error: {e}")
        raise optuna.exceptions.TrialPruned()  # Prune the trial if it fails

    # --- 5. Return the objective value ---
    # The metric is automatically logged by the callback.
    return trainer.callback_metrics["val_loss"].item()


def main():
    """Main function to run the hyperparameter optimization study."""
    args = get_args()

    # Load data once to avoid I/O in each trial
    print("--- Loading data once for the study ---")
    train_spn_list = load_spn_data_from_files(args.train_file)
    val_spn_list = load_spn_data_from_files(args.val_file)
    if not train_spn_list:
        raise ValueError("Training data is empty. Please check the data files and paths.")

    # Create the study with a pruner
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
        load_if_exists=True,
    )

    print(f"--- Starting Optuna study '{args.study_name}' ---")
    print(f"Storage: {args.storage}")
    print(f"Sampler: {study.sampler.__class__.__name__}, Pruner: {study.pruner.__class__.__name__}")

    # Start optimization
    study.optimize(
        lambda trial: objective(trial, args, train_spn_list, val_spn_list),
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=1,  # Run trials sequentially. Can be > 1 if you have multiple GPUs.
    )

    # --- Print Results ---
    print("--- Optimization Finished ---")
    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    print("Study statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials: {len(pruned_trials)}")
    print(f"  Number of complete trials: {len(complete_trials)}")

    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Value (val_loss): {best_trial.value}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
