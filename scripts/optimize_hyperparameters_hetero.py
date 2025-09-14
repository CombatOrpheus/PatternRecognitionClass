import argparse
import logging
import shutil

import lightning.pytorch as pl
import optuna
import torch
from lightning.pytorch.callbacks import EarlyStopping, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from optuna.integration import PyTorchLightningPruningCallback
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.HeterogeneousModels import HEAT_SPN_Model, RGAT_SPN_Model
from src.SPNDataModule import SPNDataModule
from src.SPNDatasets import HeterogeneousSPNDataset
from src.config_utils import load_config
from src.path_utils import PathHandler

pl.seed_everything(42, workers=True)
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.ERROR)


def objective(
    trial: optuna.Trial,
    config: argparse.Namespace,
    train_dataset: HeterogeneousSPNDataset,
    val_dataset: HeterogeneousSPNDataset,
    label_scaler: StandardScaler,
    paths: PathHandler,
) -> float:
    """The Optuna objective function for heterogeneous models."""
    gnn_operator = config.gnn_operator
    hyperparams = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "hidden_channels": trial.suggest_categorical("hidden_channels", [16, 32, 64, 128]),
        "num_layers": trial.suggest_int("num_layers", 2, 10),
        "num_heads": trial.suggest_categorical("num_heads", [1, 2, 4, 8]),
        "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024, 2048]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
    }

    data_module = SPNDataModule(
        label_to_predict=config.label,
        train_data_list=list(train_dataset),
        val_data_list=list(val_dataset),
        batch_size=hyperparams["batch_size"],
        num_workers=config.num_workers,
        heterogeneous=True,
        label_scaler=label_scaler,
    )
    data_module.setup("fit")

    model_args = {
        "in_channels": data_module.num_node_features,
        "out_channels": 1,
        "edge_dim": data_module.num_edge_features,
        "hidden_channels": hyperparams["hidden_channels"],
        "num_layers": hyperparams["num_layers"],
        "num_heads": hyperparams["num_heads"],
        "learning_rate": hyperparams["learning_rate"],
        "weight_decay": hyperparams["weight_decay"],
    }

    if gnn_operator == "rgat":
        model = RGAT_SPN_Model(**model_args)
    elif gnn_operator == "heat":
        model_args.update({
            "num_node_types": data_module.num_node_types,
            "num_edge_types": data_module.num_edge_types,
            "node_type_emb_dim": trial.suggest_categorical("node_type_emb_dim", [16, 32, 64]),
            "edge_type_emb_dim": trial.suggest_categorical("edge_type_emb_dim", [16, 32, 64]),
            "edge_attr_emb_dim": trial.suggest_categorical("edge_attr_emb_dim", [16, 32, 64]),
        })
        model = HEAT_SPN_Model(**model_args)
    else:
        raise ValueError(f"Unsupported GNN operator: {gnn_operator}")

    study_name = f"{config.study_name}_{gnn_operator}"
    logger = TensorBoardLogger(save_dir=paths.get_tensorboard_logger_dir(), name=study_name, version=f"trial_{trial.number}")

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=[
            EarlyStopping(monitor="val/loss", patience=config.patience, verbose=False, mode="min"),
            PyTorchLightningPruningCallback(trial, monitor="val/loss"),
            TQDMProgressBar(refresh_rate=20),
        ],
        enable_model_summary=False,
        log_every_n_steps=5,
    )

    try:
        trainer.fit(model, datamodule=data_module)
        return trainer.callback_metrics["val/loss"].item()
    except (RuntimeError, ValueError) as e:
        print(f"Trial {trial.number} failed with error: {e}")
        raise optuna.exceptions.TrialPruned()


def main():
    """Main function to run the optimization study."""
    config, config_path = load_config()
    paths = PathHandler(config.io)
    paths.io_config.studies_dir.mkdir(parents=True, exist_ok=True)

    operators = ["rgat", "heat"] if config.hetero_optimization.all_operators else [config.hetero_model.gnn_operator]

    print("--- Pre-loading and processing data ---")
    train_dataset = HeterogeneousSPNDataset(
        root=config.io.root, raw_data_dir=config.io.raw_data_dir, raw_file_name=config.io.train_file, label_to_predict=config.model.label
    )
    val_dataset = HeterogeneousSPNDataset(
        root=config.io.root, raw_data_dir=config.io.raw_data_dir, raw_file_name=config.io.val_file, label_to_predict=config.model.label
    )

    labels = torch.cat([data["place"].y for data in train_dataset]).numpy().reshape(-1, 1)
    label_scaler = StandardScaler().fit(labels)
    print("Data loaded and scaler fitted successfully.")

    for gnn_operator in tqdm(operators, desc="Optimizing Operators"):
        run_config = argparse.Namespace(
            **vars(config.io), **vars(config.model), **vars(config.hetero_training), **vars(config.hetero_optimization)
        )
        run_config.gnn_operator = gnn_operator

        study_name = f"{run_config.study_name}_{gnn_operator}"
        study_db_path = paths.get_study_db_path(study_name, "")  # Operator is in the name
        storage_name = paths.get_study_storage_url(study_db_path)
        config_save_path = paths.get_study_config_path(study_name)

        if not config_save_path.exists():
            shutil.copy(config_path, config_save_path)
            print(f"Saved configuration for '{study_name}' to {config_save_path}")

        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(),
            load_if_exists=True,
        )
        study.set_user_attr("label", str(run_config.label))
        study.set_user_attr("gnn_operator", run_config.gnn_operator)

        print(f"\n--- Starting Optuna study '{study_name}' ---")
        study.optimize(
            lambda trial: objective(trial, run_config, train_dataset, val_dataset, label_scaler, paths),
            n_trials=run_config.n_trials,
            timeout=run_config.timeout,
            n_jobs=1,
        )

        print(f"\n--- Optimization Finished for '{study_name}' ---")
        best_trial = study.best_trial
        print(f"  Value (val/loss): {best_trial.value}")
        print("  Params: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
        print("-" * 80)

    print("\n--- All optimization studies finished. ---")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
