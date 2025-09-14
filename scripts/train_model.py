import argparse
from pathlib import Path

import lightning.pytorch as pl
import optuna
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm

from src.HomogeneousModels import BaseGNN_SPN_Model, GraphGNN_SPN_Model, MixedGNN_SPN_Model, NodeGNN_SPN_Model
from src.SPNDataModule import SPNDataModule
from src.SPNDatasets import HomogeneousSPNDataset
from src.config_utils import load_config
from src.path_utils import PathHandler

# Base seed for ensuring reproducibility of statistical runs
BASE_SEED = 42
pl.seed_everything(BASE_SEED, workers=True)


def load_params_from_study(study_db_path: Path) -> dict:
    """Loads the best hyperparameters from an Optuna study."""
    storage_url = f"sqlite:///{study_db_path}"
    study = optuna.load_study(study_name=None, storage=storage_url)
    return {**study.best_trial.params, **study.user_attrs}


def setup_model(run_config: argparse.Namespace, node_features_dim: int) -> BaseGNN_SPN_Model:
    """Instantiates the model based on the provided arguments."""
    model_classes = {"node": NodeGNN_SPN_Model, "graph": GraphGNN_SPN_Model, "mixed": MixedGNN_SPN_Model}
    model_class = model_classes.get(
        run_config.gnn_operator_name if run_config.gnn_operator_name == "mixed" else run_config.prediction_level
    )

    model_kwargs = vars(run_config).copy()

    if "num_layers_gnn" in model_kwargs:
        model_kwargs["num_layers"] = model_kwargs.pop("num_layers_gnn")

    accepted_args = model_class.__init__.__code__.co_varnames
    filtered_kwargs = {k: v for k, v in model_kwargs.items() if k in accepted_args}

    return model_class(node_features_dim=node_features_dim, out_channels=1, **filtered_kwargs)


def run_single_training_run(
    run_config: argparse.Namespace,
    run_id: int,
    data_module: SPNDataModule,
    exp_name: str,
    path_handler: PathHandler,
) -> tuple[str, dict]:
    """Trains one model instance and returns its artifact path and test results."""
    seed = BASE_SEED + run_id
    pl.seed_everything(seed, workers=True)

    model = setup_model(run_config, data_module.num_node_features)

    logger = pl.loggers.TensorBoardLogger(
        save_dir=path_handler.get_tensorboard_logger_dir(),
        name=exp_name,
        version=f"{run_config.gnn_operator_name}_run_{run_id}",
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

    best_ckpt_path = trainer.checkpoint_callback.best_model_path
    if not best_ckpt_path or not Path(best_ckpt_path).exists():
        raise FileNotFoundError("Best model checkpoint not found.")

    # --- Save model artifacts (checkpoint and hparams) ---
    artifact_dir = path_handler.get_artifact_dir(exp_name, run_config.gnn_operator_name, run_id, seed)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    final_ckpt_path = path_handler.get_checkpoint_path(artifact_dir)
    Path(best_ckpt_path).rename(final_ckpt_path)

    results = trainer.test(ckpt_path=str(final_ckpt_path), datamodule=data_module, verbose=False)[0]
    results.update(
        {
            "run_id": run_id,
            "seed": seed,
            "final_train_loss": trainer.callback_metrics.get("train/loss_epoch"),
            "final_val_loss": trainer.callback_metrics.get("val/loss", torch.tensor(-1.0)).item(),
        }
    )
    return str(artifact_dir), results


def main():
    """Main function to orchestrate the entire experiment workflow."""
    config, _ = load_config()
    paths = PathHandler(config.io)

    exp_name = paths.get_experiment_name(config.io.train_file, config.io.test_file, config.model.label)

    search_pattern = f"{exp_name}-*.db"
    all_matching_studies = sorted(list(config.io.studies_dir.glob(search_pattern)))
    selected_studies = [
        study_path
        for study_path in all_matching_studies
        if study_path.stem.split("-")[-1] in config.model.gnn_operator
    ]

    if not selected_studies:
        print(f"No Optuna studies found matching the current configuration in '{config.io.studies_dir}'.")
        print(f"  (Searched for pattern: '{search_pattern}' with operators: {config.model.gnn_operator})")
        return

    print("\nFound and training the following studies:")
    for study_path in selected_studies:
        print(f"  - {study_path.name}")

    all_stats_results = []

    for study_path in selected_studies:
        study_params = load_params_from_study(study_path)

        # Create a copy of the config to avoid modification across loops
        run_config = argparse.Namespace(
            **vars(config.io), **vars(config.model), **vars(config.training), **vars(config.optimization)
        )
        run_config.__dict__.update(study_params)
        run_config.gnn_operator_name = run_config.gnn_operator

        print(f"\n--- Training Phase for operator: {run_config.gnn_operator_name} ---")

        train_dataset = HomogeneousSPNDataset(
            str(run_config.root), str(run_config.raw_data_dir), str(run_config.train_file), run_config.label
        )
        test_dataset = HomogeneousSPNDataset(
            str(run_config.root), str(run_config.raw_data_dir), str(run_config.test_file), run_config.label
        )

        data_module = SPNDataModule(
            train_data_list=list(train_dataset),
            test_data_list=list(test_dataset),
            label_to_predict=run_config.label,
            batch_size=run_config.batch_size,
            num_workers=run_config.num_workers,
        )
        data_module.setup()

        model_for_summary = setup_model(run_config, data_module.num_node_features)
        total_params = sum(p.numel() for p in model_for_summary.parameters())
        trainable_params = sum(p.numel() for p in model_for_summary.parameters() if p.requires_grad)

        for i in tqdm(range(run_config.num_runs), desc=f"Training {run_config.gnn_operator_name}"):
            seed = BASE_SEED + i
            artifact_dir = paths.get_artifact_dir(exp_name, run_config.gnn_operator_name, i, seed)
            checkpoint_path = paths.get_checkpoint_path(artifact_dir)

            if checkpoint_path.exists():
                tqdm.write(f"Skipping run {i} for {run_config.gnn_operator_name}: completed run found.")
                continue

            _, stats_result = run_single_training_run(run_config, i, data_module, exp_name, paths)

            hparams_to_save = vars(run_config).copy()
            hparams_to_save.update({"total_parameters": total_params, "trainable_parameters": trainable_params})
            stats_result.update(hparams_to_save)
            all_stats_results.append(stats_result)

    if all_stats_results:
        for r in all_stats_results:
            for k, v in r.items():
                if isinstance(v, Path):
                    r[k] = str(v)

        stats_file = paths.get_stats_results_path()
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_stats_results).to_parquet(stats_file, index=False)
        print(f"\nStatistical results saved to {stats_file}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
