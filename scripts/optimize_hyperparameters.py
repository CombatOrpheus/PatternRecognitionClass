import argparse
import typing
from pathlib import Path

import lightning.pytorch as pl
import optuna
import torch
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from optuna.integration import PyTorchLightningPruningCallback
from tqdm import tqdm

from src.HomogeneousModels import GraphGNN_SPN_Model, NodeGNN_SPN_Model, MixedGNN_SPN_Model
from src.PetriNets import SPNAnalysisResultLabel
from src.SPNDataModule import SPNDataModule
from src.SPNDatasets import HomogeneousSPNDataset

pl.seed_everything(42, workers=True)


def get_args() -> argparse.Namespace:
    """Parses command-line arguments for the optimization script."""
    parser = argparse.ArgumentParser(description="Optimize GNN model hyperparameters using Optuna.")
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument(
        "--root", type=str, default="processed_data/SPN_homogeneous", help="Root directory to store processed datasets."
    )
    data_group.add_argument(
        "--raw_data_dir", type=str, default="Data", help="Directory where raw .processed files are located."
    )
    data_group.add_argument(
        "--train_file",
        type=str,
        default="GridData_DS1_train_data.processed",
        help="Filename for training and validation data.",
    )
    data_group.add_argument(
        "--label", type=str, default="average_tokens_network", choices=typing.get_args(SPNAnalysisResultLabel)
    )

    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--prediction_level", type=str, default="graph", choices=["node", "graph"])
    model_group.add_argument(
        "--gnn_operator",
        type=str,
        default="gcn",
        choices=["gcn", "tag", "cheb", "sgc", "ssg", "mixed"],
        help="The GNN operator to optimize.",
    )
    model_group.add_argument("--all", action="store_true", help="If specified, runs optimization for all operators.")

    opt_group = parser.add_argument_group("Optimization Settings")
    opt_group.add_argument("--n_trials", type=int, default=100)
    opt_group.add_argument("--timeout", type=int, default=3600 * 2)
    opt_group.add_argument("--study_name", type=str, default="gnn_spn_optimization")
    opt_group.add_argument("--storage_dir", type=Path, default=Path("optuna_studies"))

    training_group = parser.add_argument_group("Fixed Training Hyperparameters")
    training_group.add_argument("--max_epochs", type=int, default=100)
    training_group.add_argument("--patience", type=int, default=10)
    training_group.add_argument("--num_workers", type=int, default=3)
    training_group.add_argument("--val_split", type=float, default=0.2, help="Fraction of training data to use for validation.")

    return parser.parse_args()


def objective(trial: optuna.Trial, args: argparse.Namespace, train_data: HomogeneousSPNDataset) -> float:
    """The Optuna objective function."""
    gnn_operator = args.gnn_operator
    hyperparams = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [8, 16, 32, 64, 128, 256]),
        "num_layers_mlp": trial.suggest_int("num_layers_mlp", 1, 5),
        "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024, 2048]),
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
        train_data_list=list(train_data),  # Pass the pre-loaded data
        label_to_predict=args.label,
        batch_size=hyperparams["batch_size"],
        num_workers=args.num_workers,
        val_split=args.val_split,
    )
    data_module.setup("fit")

    model_params = hyperparams.copy()
    model_params.pop("batch_size")

    if gnn_operator == "mixed":
        if args.prediction_level != "graph":
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

        model_class = {"graph": GraphGNN_SPN_Model, "node": NodeGNN_SPN_Model}[args.prediction_level]
        model = model_class(
            node_features_dim=data_module.num_node_features,
            out_channels=1,
            **model_params,
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


def main():
    """Main function to run the hyperparameter optimization study."""
    args = get_args()
    args.storage_dir.mkdir(parents=True, exist_ok=True)
    operators_to_run = ["gcn", "tag", "cheb", "sgc", "ssg", "mixed"] if args.all else [args.gnn_operator]

    print("--- Pre-loading and processing data ---")
    train_dataset = HomogeneousSPNDataset(
        root=args.root,
        raw_data_dir=args.raw_data_dir,
        raw_file_name=args.train_file,
        label_to_predict=args.label,
    )
    print("Data loaded successfully.")

    for gnn_operator in tqdm(operators_to_run, desc="Total Optimization Progress"):
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

        for key, value in vars(run_args).items():
            if not isinstance(value, Path):
                study.set_user_attr(key, value)

        study.optimize(
            lambda trial: objective(trial, run_args, train_dataset),
            n_trials=run_args.n_trials,
            timeout=run_args.timeout,
            show_progress_bar=False,
        )

        print(f"\n--- Optimization Finished for operator '{run_args.gnn_operator}' ---")
        print("Best trial:")
        best_trial = study.best_trial
        print(f"  Value (val/loss): {best_trial.value:.4f}")
        print("  Params: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
        print("-" * 80)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
