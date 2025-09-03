import argparse
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict

import lightning.pytorch as pl
import optuna
import pandas as pd
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from tqdm import tqdm

from src.HomogeneousModels import BaseGNN_SPN_Model, NodeGNN_SPN_Model, GraphGNN_SPN_Model
from src.PetriNets import load_spn_data_from_files
from src.SPNDataModule import SPNDataModule

# Base seed for ensuring reproducibility of statistical runs
BASE_SEED = 42
pl.seed_everything(BASE_SEED, workers=True)


def get_args() -> argparse.Namespace:
    """Parses command-line arguments for the experiment script."""
    parser = argparse.ArgumentParser(
        description="Run a full GNN experiment: training statistical runs and cross-dataset evaluation."
    )
    parser.add_argument(
        "--studies_dir", type=Path, default=Path("optuna_studies"), help="Directory containing Optuna study .db files."
    )
    parser.add_argument(
        "--cross_eval_dir", type=Path, default=Path("./Data"), help="Directory of datasets for cross-evaluation."
    )
    parser.add_argument(
        "--stats_results_file",
        type=Path,
        default=Path("results/statistical_results.parquet"),
        help="Path to save primary statistical results.",
    )
    parser.add_argument(
        "--cross_eval_results_file",
        type=Path,
        default=Path("results/cross_dataset_evaluation.parquet"),
        help="Path to save cross-dataset evaluation results.",
    )
    parser.add_argument("--log_dir", type=Path, default="lightning_logs")
    parser.add_argument("--exp_name", type=str, default="gnn_spn_experiment")
    parser.add_argument("--train_file", type=Path, default=None, help="Override the training file path from the study.")
    parser.add_argument("--val_file", type=Path, default=None, help="Override the validation file path from the study.")
    parser.add_argument(
        "--test_file",
        type=Path,
        default=Path("Data/GridData_DS1_all_data.processed"),
        help="Path to the primary test data file.",
    )
    parser.add_argument("--num_runs", type=int, default=30, help="Number of statistical runs for each selected study.")
    parser.add_argument("--num_workers", type=int, default=3)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    return parser.parse_args()


def select_studies(studies_dir: Path) -> List[Path]:
    """Scans a directory for Optuna studies and prompts the user to select which ones to run."""
    db_files = sorted(list(studies_dir.glob("*.db")))
    if not db_files:
        print(f"No Optuna study (.db) files found in '{studies_dir}'.")
        return []
    print("\nAvailable studies found:")
    for i, db_path in enumerate(db_files):
        print(f"  [{i + 1}] {db_path.name}")
    while True:
        try:
            selection = input("Enter study numbers to run (e.g., 1,3,4 or 'all'): ")
            if selection.lower() == "all":
                return db_files
            selected_indices = [int(i.strip()) - 1 for i in selection.split(",")]
            if all(0 <= i < len(db_files) for i in selected_indices):
                return [db_files[i] for i in selected_indices]
            print("Error: Selection out of range.")
        except ValueError:
            print("Invalid input.")


def load_params_from_study(study_db_path: Path) -> dict:
    """Loads the best hyperparameters and user attributes from an Optuna study."""
    storage_url = f"sqlite:///{study_db_path}"
    study = optuna.load_study(study_name=None, storage=storage_url)
    all_params = {**study.best_trial.params, **study.user_attrs}
    for key in ["train_file", "val_file"]:
        if key in all_params:
            all_params[key] = Path(all_params[key])
    return all_params


def load_and_cache_data(args: argparse.Namespace) -> tuple[list, list, list, str]:
    """Loads, processes, and caches all data splits in a temporary directory."""
    print("\n--- Preparing and Caching Data ---")
    cache_dir = tempfile.mkdtemp()
    print(f"Cache directory created at: {cache_dir}")
    try:
        raw_train = load_spn_data_from_files(args.train_file)
        raw_val = load_spn_data_from_files(args.val_file)
        raw_test = load_spn_data_from_files(args.test_file)
        if not all([raw_train, raw_val, raw_test]):
            raise ValueError("One or more data files failed to load or were empty.")
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
        print("Processed data has been cached successfully.")
        return temp_dm.train_data, temp_dm.val_data, temp_dm.test_data, cache_dir
    except Exception as e:
        print(f"FATAL: Could not load or process initial datasets. Error: {e}")
        shutil.rmtree(cache_dir)
        raise


def setup_model(args: argparse.Namespace, node_features_dim: int) -> BaseGNN_SPN_Model:
    """Sets up the model using the provided arguments."""
    model_classes = {"node": NodeGNN_SPN_Model, "graph": GraphGNN_SPN_Model}
    model_class = model_classes[args.prediction_level]
    model = model_class(
        node_features_dim=node_features_dim,
        out_channels=1,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers_gnn,
        num_layers_mlp=args.num_layers_mlp,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gnn_operator_name=args.gnn_operator,
        gnn_k_hops=getattr(args, "gnn_k_hops", 3),
        gnn_alpha=getattr(args, "gnn_alpha", 0.1),
    )
    model.hparams["model_class_path"] = f"{model.__class__.__module__}.{model.__class__.__name__}"
    return model


def run_single_training_run(args: argparse.Namespace, run_id: int, train_data, val_data, test_data) -> tuple[str, dict]:
    """Trains a single model, tests it, and returns its best checkpoint path and test metrics."""
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
    logger = pl.loggers.TensorBoardLogger(
        save_dir=str(args.log_dir), name=args.exp_name, version=f"{args.gnn_operator}_run_{run_id}"
    )
    checkpoint_callback = ModelCheckpoint(monitor="val/loss", mode="min", filename="best")
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        logger=logger,
        callbacks=[checkpoint_callback, EarlyStopping(monitor="val/loss", patience=args.patience)],
        deterministic=True,
        enable_progress_bar=False,
        log_every_n_steps=1,
        enable_model_summary=False,
    )
    trainer.fit(model, datamodule=data_module)

    final_train_loss = trainer.callback_metrics.get("train/loss_epoch", -1.0)
    final_val_loss = trainer.callback_metrics.get("val/loss", -1.0)

    best_model_path = trainer.checkpoint_callback.best_model_path
    if not best_model_path:
        raise FileNotFoundError("Could not find the best model checkpoint.")
    test_results = trainer.test(ckpt_path=best_model_path, datamodule=data_module, verbose=False)

    results_with_id = test_results[0]
    results_with_id["run_id"], results_with_id["seed"] = run_id, seed
    # Add the captured loss values to the results dictionary
    results_with_id["final_train_loss"] = float(final_train_loss)
    results_with_id["final_val_loss"] = float(final_val_loss)

    return best_model_path, results_with_id


def perform_cross_evaluation(
    best_model_paths: List[str],
    hparams_list: List[Dict],
    datamodule_hparams: Dict,
    cross_eval_dir: Path,
    node_features_dim: int,
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
            data_module = SPNDataModule(
                label_to_predict=datamodule_hparams["label_to_predict"],
                train_data_list=[],
                val_data_list=[],
                test_data_list=raw_test_data,
                batch_size=datamodule_hparams["batch_size"],
                num_workers=datamodule_hparams["num_workers"],
            )
            trainer = pl.Trainer(
                accelerator="auto", logger=False, enable_progress_bar=False, enable_model_summary=False
            )
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
    cache_dir = None

    try:
        first_study_params = load_params_from_study(selected_studies[0])
        data_loading_args = argparse.Namespace(**first_study_params)
        data_loading_args.train_file = base_args.train_file or data_loading_args.train_file
        data_loading_args.val_file = base_args.val_file or data_loading_args.val_file
        data_loading_args.test_file = base_args.test_file

        train_data, val_data, test_data, cache_dir = load_and_cache_data(data_loading_args)
        node_features_dim = train_data[0].num_node_features

        # --- PHASE 1: TRAINING ---
        for study_path in selected_studies:
            study_params = load_params_from_study(study_path)
            run_args = argparse.Namespace(**vars(base_args))
            run_args.__dict__.update(study_params)
            run_args.train_file = base_args.train_file or run_args.train_file
            run_args.val_file = base_args.val_file or run_args.val_file

            print(f"\n--- Training Phase for operator: {run_args.gnn_operator} ---")

            model_for_summary = setup_model(run_args, node_features_dim)
            total_params = sum(p.numel() for p in model_for_summary.parameters())
            trainable_params = sum(p.numel() for p in model_for_summary.parameters() if p.requires_grad)

            for i in tqdm(range(run_args.num_runs), desc=f"Training {run_args.gnn_operator}"):
                best_model_path, stats_result = run_single_training_run(run_args, i, train_data, val_data, test_data)

                hparams_to_save = vars(run_args).copy()
                hparams_to_save.update(
                    {
                        "run_id": i,
                        "seed": BASE_SEED + i,
                        "total_parameters": total_params,
                        "trainable_parameters": trainable_params,
                    }
                )

                stats_result.update(hparams_to_save)
                all_stats_results.append(stats_result)
                all_best_model_paths.append(best_model_path)
                all_hparams_for_eval.append(hparams_to_save)

        # --- PHASE 2: CROSS-EVALUATION ---
        if all_best_model_paths:
            datamodule_hparams = {
                "label_to_predict": data_loading_args.label,
                "batch_size": 128,
                "num_workers": base_args.num_workers,
            }
            all_cross_eval_results = perform_cross_evaluation(
                all_best_model_paths,
                all_hparams_for_eval,
                datamodule_hparams,
                base_args.cross_eval_dir,
                node_features_dim,
            )

        # --- Final aggregation and saving of results ---
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

    finally:
        if cache_dir:
            print(f"\n--- Cleaning up cache directory: {cache_dir} ---")
            shutil.rmtree(cache_dir)
            print("Cache cleaned up successfully.")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
