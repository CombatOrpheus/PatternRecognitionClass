import argparse
from pathlib import Path
from typing import List, Dict

import lightning.pytorch as pl
import optuna
import pandas as pd
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from tqdm import tqdm

from src.HomogeneousModels import BaseGNN_SPN_Model, NodeGNN_SPN_Model, GraphGNN_SPN_Model, MixedGNN_SPN_Model
from src.SPNDataModule import SPNDataModule

# Base seed for ensuring reproducibility of statistical runs
BASE_SEED = 42
pl.seed_everything(BASE_SEED, workers=True)


def get_args() -> argparse.Namespace:
    """Parses command-line arguments for the experiment script."""
    parser = argparse.ArgumentParser(
        description="Run a full GNN experiment: training statistical runs and cross-dataset evaluation."
    )
    # --- I/O and Configuration ---
    parser.add_argument(
        "--root",
        type=str,
        default="processed_data/SPN_homogeneous",
        help="Root directory for storing processed datasets.",
    )
    parser.add_argument(
        "--raw_data_dir", type=str, default="Data", help="Directory where raw .processed files are located."
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="GridData_DS1_train_data.processed",
        help="Filename for creating training and validation sets.",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="GridData_DS1_all_data.processed",
        help="Filename for the final, held-out test set.",
    )
    parser.add_argument(
        "--studies_dir", type=Path, default=Path("optuna_studies"), help="Directory containing Optuna study .db files."
    )
    parser.add_argument(
        "--cross_eval_dir", type=Path, default=Path("./Data"), help="Directory of datasets for cross-evaluation."
    )
    parser.add_argument("--stats_results_file", type=Path, default=Path("results/statistical_results.parquet"))
    parser.add_argument(
        "--cross_eval_results_file", type=Path, default=Path("results/cross_dataset_evaluation.parquet")
    )
    parser.add_argument("--log_dir", type=Path, default="lightning_logs")
    parser.add_argument("--exp_name", type=str, default="gnn_spn_experiment")

    # --- Training Settings ---
    parser.add_argument("--num_runs", type=int, default=30, help="Number of statistical runs for each selected study.")
    parser.add_argument("--num_workers", type=int, default=3)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)

    return parser.parse_args()


def select_studies(studies_dir: Path) -> List[Path]:
    """Scans for and allows user to select Optuna studies to run."""
    db_files = sorted(list(studies_dir.glob("*.db")))
    if not db_files:
        print(f"No Optuna study (.db) files found in '{studies_dir}'.")
        return []

    print("\nAvailable studies found:")
    for i, db_path in enumerate(db_files):
        print(f"  [{i + 1}] {db_path.name}")

    while True:
        try:
            selection = input("Enter study numbers to run (e.g., 1,3 or 'all'): ")
            if selection.lower() == "all":
                return db_files
            selected_indices = [int(i.strip()) - 1 for i in selection.split(",")]
            if all(0 <= i < len(db_files) for i in selected_indices):
                return [db_files[i] for i in selected_indices]
            print("Error: Selection out of range.")
        except ValueError:
            print("Invalid input.")


def load_params_from_study(study_db_path: Path) -> dict:
    """Loads the best hyperparameters from an Optuna study."""
    storage_url = f"sqlite:///{study_db_path}"
    study = optuna.load_study(study_name=None, storage=storage_url)
    return {**study.best_trial.params, **study.user_attrs}


def setup_model(args: argparse.Namespace, node_features_dim: int) -> BaseGNN_SPN_Model:
    """Instantiates the model based on the provided arguments."""
    model_classes = {"node": NodeGNN_SPN_Model, "graph": GraphGNN_SPN_Model, "mixed": MixedGNN_SPN_Model}
    model_class = model_classes.get(args.gnn_operator if args.gnn_operator == "mixed" else args.prediction_level)

    # Prepare model init args from the combined run_args namespace
    model_kwargs = vars(args).copy()

    # Handle parameter name difference between optuna study and model constructor
    if "num_layers_gnn" in model_kwargs:
        model_kwargs["num_layers"] = model_kwargs.pop("num_layers_gnn")

    # Filter kwargs to only those accepted by the model's constructor
    accepted_args = model_class.__init__.__code__.co_varnames
    filtered_kwargs = {k: v for k, v in model_kwargs.items() if k in accepted_args}

    return model_class(node_features_dim=node_features_dim, out_channels=1, **filtered_kwargs)


def run_single_training_run(args: argparse.Namespace, run_id: int, data_module: SPNDataModule) -> tuple[str, dict]:
    """Trains one model instance and returns its checkpoint path and results."""
    seed = BASE_SEED + run_id
    pl.seed_everything(seed, workers=True)

    node_features_dim = data_module.train_dataset.dataset[0].num_node_features
    model = setup_model(args, node_features_dim)

    logger = pl.loggers.TensorBoardLogger(
        save_dir=str(args.log_dir), name=args.exp_name, version=f"{args.gnn_operator}_run_{run_id}"
    )
    checkpoint_callback = ModelCheckpoint(monitor="val/loss", mode="min", filename="best")

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        logger=logger,
        callbacks=[checkpoint_callback, EarlyStopping(monitor="val/loss", patience=args.patience)],
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, datamodule=data_module)

    best_model_path = trainer.checkpoint_callback.best_model_path
    if not best_model_path:
        raise FileNotFoundError("Best model checkpoint not found.")

    results = trainer.test(ckpt_path=best_model_path, datamodule=data_module, verbose=False)[0]
    results.update(
        {
            "run_id": run_id,
            "seed": seed,
            "final_train_loss": trainer.callback_metrics.get("train/loss_epoch", torch.tensor(-1.0)).item(),
            "final_val_loss": trainer.callback_metrics.get("val/loss", torch.tensor(-1.0)).item(),
        }
    )
    return best_model_path, results


def perform_cross_evaluation(
    best_model_paths: List[str], hparams_list: List[Dict], cross_eval_dir: Path, node_features_dim: int
) -> List[Dict]:
    """Evaluates a list of trained models against a suite of test datasets, loading data only once."""
    print(f"\n--- Starting Cross-Dataset Evaluation Phase ---")
    test_files = sorted(list(cross_eval_dir.glob("*.processed")))
    if not test_files:
        return []

    print("Pre-loading all cross-evaluation datasets into memory...")
    cross_eval_data_cache = {file: load_spn_data_from_files(file) for file in test_files}
    print(f"Loaded {len(cross_eval_data_cache)} datasets.")

    all_eval_results = []
    for i, model_path in enumerate(tqdm(best_model_paths, desc="Cross-Evaluating Checkpoints")):
        hparams = hparams_list[i]
        model_class = setup_model(argparse.Namespace(**hparams), node_features_dim).__class__
        model = model_class.load_from_checkpoint(model_path)
        model.eval()

        for data_file, raw_test_data in cross_eval_data_cache.items():
            if not raw_test_data:
                continue

            # Use a temporary DataModule for evaluation on this specific dataset
            data_module = SPNDataModule(
                root=hparams["root"],
                raw_data_dir=hparams["raw_data_dir"],
                train_file=data_file.name,
                test_file=data_file.name,  # Use same file for dummy setup
                label_to_predict=hparams["label"],
                batch_size=hparams["batch_size"],
                num_workers=hparams["num_workers"],
            )
            data_module.setup("test")  # Only setup the test set

            trainer = pl.Trainer(accelerator="auto", logger=False, enable_progress_bar=False)
            test_metrics = trainer.test(model, datamodule=data_module, verbose=False)

            if test_metrics:
                result_dict = test_metrics[0]
                result_dict["cross_eval_dataset"] = data_file.name
                result_dict.update(hparams)
                all_eval_results.append(result_dict)

    return all_eval_results


def main():
    """Main function to orchestrate the entire experiment workflow."""
    base_args = get_args()
    selected_studies = select_studies(base_args.studies_dir)
    if not selected_studies:
        return

    all_stats_results = []
    all_best_model_paths = []
    all_hparams_for_eval = []

    # --- PHASE 1: TRAINING ---
    for study_path in selected_studies:
        study_params = load_params_from_study(study_path)
        run_args = argparse.Namespace(**vars(base_args))
        run_args.__dict__.update(study_params)

        print(f"\n--- Training Phase for operator: {run_args.gnn_operator} ---")

        data_module = SPNDataModule(
            root=run_args.root,
            raw_data_dir=run_args.raw_data_dir,
            train_file=run_args.train_file,
            test_file=run_args.test_file,
            label_to_predict=run_args.label,
            batch_size=run_args.batch_size,
            num_workers=run_args.num_workers,
        )
        data_module.setup("fit")

        node_features_dim = data_module.train_dataset.dataset[0].num_node_features
        model_for_summary = setup_model(run_args, node_features_dim)
        total_params = sum(p.numel() for p in model_for_summary.parameters())
        trainable_params = sum(p.numel() for p in model_for_summary.parameters() if p.requires_grad)

        for i in tqdm(range(run_args.num_runs), desc=f"Training {run_args.gnn_operator}"):
            best_model_path, stats_result = run_single_training_run(run_args, i, data_module)

            hparams_to_save = vars(run_args).copy()
            hparams_to_save.update({"total_parameters": total_params, "trainable_parameters": trainable_params})

            stats_result.update(hparams_to_save)
            all_stats_results.append(stats_result)
            all_best_model_paths.append(best_model_path)
            all_hparams_for_eval.append(hparams_to_save)

    # --- PHASE 2: CROSS-EVALUATION ---
    if all_best_model_paths:
        datamodule_hparams = {"batch_size": 512, "num_workers": base_args.num_workers}
        all_cross_eval_results = perform_cross_evaluation(
            all_best_model_paths, all_hparams_for_eval, base_args.cross_eval_dir, node_features_dim
        )

    # --- Final saving ---
    for result_list in [all_stats_results, all_cross_eval_results]:
        for r in result_list:
            for k, v in r.items():
                if isinstance(v, Path):
                    r[k] = str(v)

    if all_stats_results:
        base_args.stats_results_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_stats_results).to_parquet(base_args.stats_results_file, index=False)
        print(f"\nStatistical results saved to {base_args.stats_results_file}")

    if all_cross_eval_results:
        base_args.cross_eval_results_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_cross_eval_results).to_parquet(base_args.cross_eval_results_file, index=False)
        print(f"Cross-evaluation results saved to {base_args.cross_eval_results_file}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
