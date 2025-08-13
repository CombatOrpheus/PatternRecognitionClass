import argparse
import shutil
import tempfile
import typing
from pathlib import Path

import lightning.pytorch as pl
import optuna
import pandas as pd
import torch
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
)
from lightning.pytorch.loggers import TensorBoardLogger

from src.HomogeneousModels import (
    GraphGNN_SPN_Model,
    NodeGNN_SPN_Model,
    BaseGNN_SPN_Model,
)
from src.PetriNets import load_spn_data_from_files
from src.SPNDataModule import SPNDataModule

# A base seed for reproducibility of the entire experiment suite
BASE_SEED = 42
pl.seed_everything(BASE_SEED, workers=True)


def get_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run statistical analysis on GNN models using optimized hyperparameters from Optuna studies."
    )

    parser.add_argument(
        "--studies_dir", type=Path, default=Path("optuna_studies"), help="Directory containing Optuna study .db files."
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=30,
        help="Number of times to run training for each selected study to gather statistical data.",
    )
    parser.add_argument(
        "--results_file",
        type=Path,
        default=Path("results/statistical_results.parquet"),
        help="Path to save or append aggregated results.",
    )

    parser.add_argument(
        "--test_file",
        type=Path,
        default=Path("Data/GridData_DS1_all_data.processed"),
        help="Path to the test data file (can be overridden by study).",
    )
    parser.add_argument("--num_workers", type=int, default=3)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--log_dir", type=Path, default="lightning_logs")
    parser.add_argument("--exp_name", type=str, default="gnn_spn_statistical_run")
    parser.add_argument("--swa_lrs", type=float, default=None)

    return parser.parse_args()


def select_studies(studies_dir: Path) -> typing.List[Path]:
    """Scans a directory for Optuna studies and prompts the user to select which ones to run."""
    print(f"--- Scanning for studies in '{studies_dir}' ---")
    db_files = sorted(list(studies_dir.glob("*.db")))

    if not db_files:
        print("No Optuna study (.db) files found in the specified directory.")
        return []

    print("\nAvailable studies found:")
    for i, db_path in enumerate(db_files):
        print(f"  [{i + 1}] {db_path.name}")

    while True:
        try:
            selection = input("\nEnter the numbers of the studies to run (e.g., 1,3,4 or 'all'), then press Enter: ")
            if selection.lower() == "all":
                return db_files

            selected_indices = [int(i.strip()) - 1 for i in selection.split(",")]

            if all(0 <= i < len(db_files) for i in selected_indices):
                return [db_files[i] for i in selected_indices]
            else:
                print("Error: One or more selected numbers are out of range.")
        except ValueError:
            print("Invalid input. Please enter numbers separated by commas, or 'all'.")


def load_params_from_study(study_db_path: Path) -> dict:
    """Loads the best hyperparameters and user attributes from an Optuna study."""
    print(f"\n--- Loading best trial from study: {study_db_path.name} ---")
    storage_url = f"sqlite:///{study_db_path}"
    study = optuna.load_study(study_name=None, storage=storage_url)

    best_params = study.best_trial.params
    user_attrs = study.user_attrs
    all_params = {**best_params, **user_attrs}

    for key in ["train_file", "val_file"]:
        if key in all_params:
            all_params[key] = Path(all_params[key])

    return all_params


def prepare_and_cache_data(args: argparse.Namespace) -> tuple[list, list, list, str]:
    """Loads, processes, and caches all data splits based on paths in args."""
    print("\n--- Preparing and Caching Data ---")
    cache_dir = tempfile.mkdtemp()
    print(f"Cache directory created at: {cache_dir}")

    raw_train = load_spn_data_from_files(args.train_file)
    raw_val = load_spn_data_from_files(args.val_file)
    raw_test = load_spn_data_from_files(args.test_file)

    temp_dm = SPNDataModule(
        label_to_predict=args.label,
        train_data_list=raw_train,
        val_data_list=raw_val,
        test_data_list=raw_test,
        batch_size=128,
    )
    temp_dm.setup()

    torch.save(temp_dm.train_data, Path(cache_dir) / "train.pt")
    torch.save(temp_dm.val_data, Path(cache_dir) / "val.pt")
    torch.save(temp_dm.test_data, Path(cache_dir) / "test.pt")
    print("Processed data has been cached.")

    return temp_dm.train_data, temp_dm.val_data, temp_dm.test_data, cache_dir


def setup_model(args: argparse.Namespace, node_features_dim: int) -> BaseGNN_SPN_Model:
    """Sets up the model using pre-processed data."""
    model_class = NodeGNN_SPN_Model if args.prediction_level == "node" else GraphGNN_SPN_Model

    gnn_k_hops = getattr(args, "gnn_k_hops", 3)
    gnn_alpha = getattr(args, "gnn_alpha", 0.1)

    model = model_class(
        node_features_dim=node_features_dim,
        out_channels=1,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers_gnn,
        num_layers_mlp=args.num_layers_mlp,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gnn_operator_name=args.gnn_operator,
        gnn_k_hops=gnn_k_hops,
        gnn_alpha=gnn_alpha,
    )
    return model


def run_single_experiment(
    args: argparse.Namespace, run_id: int, train_data: list, val_data: list, test_data: list
) -> dict:
    """Executes a single training and testing run with a specific seed."""
    seed = BASE_SEED + run_id
    pl.seed_everything(seed, workers=True)

    data_module = SPNDataModule(
        label_to_predict=args.label,
        train_data_list=[],
        val_data_list=[],
        test_data_list=[],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    data_module.train_data, data_module.val_data, data_module.test_data = train_data, val_data, test_data

    model = setup_model(args, train_data[0].num_node_features)

    logger = TensorBoardLogger(
        save_dir=str(args.log_dir), name=args.exp_name, version=f"{args.gnn_operator}_run_{run_id}"
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=[
            ModelCheckpoint(monitor="val_loss", mode="min"),
            EarlyStopping(monitor="val_loss", patience=args.patience),
        ],
        deterministic=True,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    trainer.fit(model, data_module)
    test_results = trainer.test(datamodule=data_module, ckpt_path="best", verbose=False)

    results_with_id = dict(test_results[0])  # Explicitly cast to dict
    results_with_id["run_id"], results_with_id["seed"] = run_id, seed
    return results_with_id


def main() -> None:
    """Main function to orchestrate the selection and execution of training runs."""
    base_args = get_args()
    selected_studies = select_studies(base_args.studies_dir)

    if not selected_studies:
        return

    all_experiments_results = []
    cache_dir = None

    try:
        first_study_params = load_params_from_study(selected_studies[0])
        temp_args = argparse.Namespace(**vars(base_args))
        temp_args.__dict__.update(first_study_params)

        train_data, val_data, test_data, cache_dir = prepare_and_cache_data(temp_args)

        for study_path in selected_studies:
            run_args = argparse.Namespace(**vars(base_args))
            study_params = load_params_from_study(study_path)
            run_args.__dict__.update(study_params)

            print(f"\n--- Running {run_args.num_runs} experiments for operator: {run_args.gnn_operator} ---")

            model_for_summary = setup_model(run_args, train_data[0].num_node_features)
            total_params = sum(p.numel() for p in model_for_summary.parameters())
            trainable_params = sum(p.numel() for p in model_for_summary.parameters() if p.requires_grad)

            for i in range(run_args.num_runs):
                print(f"Starting run {i+1}/{run_args.num_runs}...")
                result = run_single_experiment(run_args, i, train_data, val_data, test_data)

                hparams_to_save = {
                    "gnn_operator": run_args.gnn_operator,
                    "prediction_level": run_args.prediction_level,
                    "label": run_args.label,
                    "learning_rate": run_args.learning_rate,
                    "weight_decay": run_args.weight_decay,
                    "hidden_dim": run_args.hidden_dim,
                    "num_layers_gnn": run_args.num_layers_gnn,
                    "num_layers_mlp": run_args.num_layers_mlp,
                    "batch_size": run_args.batch_size,
                    "total_parameters": total_params,
                    "trainable_parameters": trainable_params,
                }

                result.update(hparams_to_save)
                all_experiments_results.append(result)

        if base_args.results_file:
            print(f"\n--- Saving/Appending results to {base_args.results_file} ---")
            new_results_df = pd.DataFrame(all_experiments_results)

            base_args.results_file.parent.mkdir(parents=True, exist_ok=True)

            if base_args.results_file.exists():
                print("Results file exists. Appending new results.")
                existing_df = pd.read_parquet(base_args.results_file)
                combined_df = pd.concat([existing_df, new_results_df], ignore_index=True)
                combined_df.to_parquet(base_args.results_file, index=False)
            else:
                print("Creating new results file.")
                new_results_df.to_parquet(base_args.results_file, index=False)

            print("Results saved successfully.")

    finally:
        if cache_dir:
            print(f"\n--- Cleaning up cache directory: {cache_dir} ---")
            shutil.rmtree(cache_dir)
            print("Cache cleaned up successfully.")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
