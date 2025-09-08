import argparse
import json
from pathlib import Path
from typing import List

import lightning.pytorch as pl
import optuna
import pandas as pd
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from tqdm import tqdm

from src.HomogeneousModels import BaseGNN_SPN_Model, GraphGNN_SPN_Model, MixedGNN_SPN_Model, NodeGNN_SPN_Model
from src.SPNDataModule import SPNDataModule
from src.SPNDatasets import HomogeneousSPNDataset
from src.config_utils import load_config

# Base seed for ensuring reproducibility of statistical runs
BASE_SEED = 42
pl.seed_everything(BASE_SEED, workers=True)


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


def setup_model(run_config: argparse.Namespace, node_features_dim: int) -> BaseGNN_SPN_Model:
    """Instantiates the model based on the provided arguments."""
    model_classes = {"node": NodeGNN_SPN_Model, "graph": GraphGNN_SPN_Model, "mixed": MixedGNN_SPN_Model}
    model_class = model_classes.get(run_config.gnn_operator if run_config.gnn_operator == "mixed" else run_config.prediction_level)

    model_kwargs = vars(run_config).copy()

    if "num_layers_gnn" in model_kwargs:
        model_kwargs["num_layers"] = model_kwargs.pop("num_layers_gnn")

    accepted_args = model_class.__init__.__code__.co_varnames
    filtered_kwargs = {k: v for k, v in model_kwargs.items() if k in accepted_args}

    return model_class(node_features_dim=node_features_dim, out_channels=1, **filtered_kwargs)


def run_single_training_run(run_config: argparse.Namespace, run_id: int, data_module: SPNDataModule) -> tuple[str, dict]:
    """Trains one model instance and returns its artifact path and test results."""
    seed = BASE_SEED + run_id
    pl.seed_everything(seed, workers=True)

    model = setup_model(run_config, data_module.num_node_features)

    logger = pl.loggers.TensorBoardLogger(
        save_dir=str(run_config.log_dir), name=run_config.exp_name, version=f"{run_config.gnn_operator}_run_{run_id}"
    )
    checkpoint_callback = ModelCheckpoint(monitor="val/loss", mode="min", filename="best")

    trainer = pl.Trainer(
        max_epochs=run_config.max_epochs,
        accelerator="auto",
        logger=logger,
        callbacks=[checkpoint_callback, EarlyStopping(monitor="val/loss", patience=run_config.patience)],
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=1,
    )
    trainer.fit(model, datamodule=data_module)

    best_model_path = trainer.checkpoint_callback.best_model_path
    if not best_model_path:
        raise FileNotFoundError("Best model checkpoint not found.")

    # --- Save model artifacts (state_dict and hparams) ---
    artifact_dir = run_config.state_dict_dir / run_config.exp_name / f"{run_config.gnn_operator}_run_{run_id}_seed_{seed}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    state_dict_path = artifact_dir / "best_model.pt"
    checkpoint = torch.load(best_model_path, map_location="cpu")
    torch.save(checkpoint["state_dict"], state_dict_path)

    hparams_path = artifact_dir / "hparams.json"
    hparams_to_save = {k: v for k, v in vars(run_config).items() if isinstance(v, (str, int, float, bool))}
    with open(hparams_path, "w") as f:
        json.dump(hparams_to_save, f, indent=4)
    # ---

    results = trainer.test(ckpt_path=best_model_path, datamodule=data_module, verbose=False)[0]
    results.update(
        {
            "run_id": run_id,
            "seed": seed,
            "final_train_loss": trainer.callback_metrics.get("train/loss_epoch", torch.tensor(-1.0)).item(),
            "final_val_loss": trainer.callback_metrics.get("val/loss", torch.tensor(-1.0)).item(),
        }
    )
    return str(artifact_dir), results


def main():
    """Main function to orchestrate the entire experiment workflow."""
    config = load_config()

    selected_studies = select_studies(config.io.studies_dir)
    if not selected_studies:
        return

    all_stats_results = []

    for study_path in selected_studies:
        study_params = load_params_from_study(study_path)

        # Create a copy of the config to avoid modification across loops
        run_config = argparse.Namespace(**vars(config.io), **vars(config.model), **vars(config.training), **vars(config.optimization))
        run_config.__dict__.update(study_params)

        print(f"\n--- Training Phase for operator: {run_config.gnn_operator} ---")

        train_dataset = HomogeneousSPNDataset(str(run_config.root), str(run_config.raw_data_dir), str(run_config.train_file), run_config.label)
        test_dataset = HomogeneousSPNDataset(str(run_config.root), str(run_config.raw_data_dir), str(run_config.test_file), run_config.label)

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

        for i in tqdm(range(run_args.num_runs), desc=f"Training {run_args.gnn_operator}"):
            seed = BASE_SEED + i
            artifact_dir_path = run_args.state_dict_dir / run_args.exp_name / f"{run_args.gnn_operator}_run_{i}_seed_{seed}"

            # Check if this run has already been completed
            hparams_path = artifact_dir_path / "hparams.json"
            if artifact_dir_path.exists() and (artifact_dir_path / "best_model.pt").exists() and hparams_path.exists():
                with open(hparams_path, "r") as f:
                    try:
                        existing_hparams = json.load(f)
                        current_hparams = {
                            k: v for k, v in vars(run_args).items() if isinstance(v, (str, int, float, bool))
                        }

                        if existing_hparams == current_hparams:
                            tqdm.write(f"Skipping run {i} for {run_args.gnn_operator}: identical completed run found.")
                            continue
                        else:
                            tqdm.write(f"Re-running run {i} for {run_args.gnn_operator}: hyperparameters have changed.")
                    except json.JSONDecodeError:
                        tqdm.write(f"Re-running run {i} for {run_args.gnn_operator}: corrupted hparams.json found.")

            artifact_dir, stats_result = run_single_training_run(run_args, i, data_module)

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


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
