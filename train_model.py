import argparse
import typing
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    DeviceStatsMonitor,
    RichProgressBar,
    ModelSummary,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.tuner import Tuner

from src.HomogeneousModels import FlexibleGNN_SPN_Model
from src.PetriNets import load_spn_data_from_files, SPNAnalysisResultLabel
from src.SPNDataModule import SPNDataModule

# Set a seed for reproducibility
pl.seed_everything(42, workers=True)


def get_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a GNN model on SPN data.")

    # --- Paths and Data ---
    parser.add_argument("--data_dir", type=str, default="./Data", help="Directory containing the data files.")
    parser.add_argument("--train_file", type=str, default="GridData_DS1_train_data.processed",
                        help="Training data filename.")
    parser.add_argument("--val_file", type=str, default="GridData_DS1_test_data.processed",
                        help="Validation data filename.")
    parser.add_argument("--test_file", type=str, default="GridData_DS1_all_data.processed", help="Test data filename.")
    parser.add_argument("--label", type=str, default="average_tokens_per_place", help="The label to predict.",
                        choices=typing.get_args(SPNAnalysisResultLabel))

    # --- Model Hyperparameters ---
    parser.add_argument("--hidden_dim", type=int, default=32, help="Hidden dimension size for GNN layers.")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of GNN layers.")
    parser.add_argument("--prediction_level", type=str, default="node", choices=['node', 'graph'],
                        help="Prediction task type.")

    # --- Training Hyperparameters ---
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training.")
    parser.add_argument("--num_workers", type=int, default=3, help="Number of workers for data loading.")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of training epochs.")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping.")
    parser.add_argument("--log_dir", type=str, default="lightning_logs", help="Directory for logs and checkpoints.")
    parser.add_argument("--exp_name", type=str, default="gnn_spn_experiment", help="Experiment name for the logger.")

    return parser.parse_args()


def main(args):
    """Main function to set up, tune the learning rate, and run training."""

    print("--- Setting up DataModule and Model ---")
    data_dir = Path(args.data_dir)
    train_spn_list = load_spn_data_from_files(data_dir / args.train_file)
    val_spn_list = load_spn_data_from_files(data_dir / args.val_file)
    test_spn_list = load_spn_data_from_files(data_dir / args.test_file)

    data_module = SPNDataModule(
        label_to_predict=args.label,
        train_data_list=train_spn_list,
        val_data_list=val_spn_list,
        test_data_list=test_spn_list,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = FlexibleGNN_SPN_Model(
        node_features_dim=4,  # Assuming this is fixed based on data structure
        out_channels=1,  # Assuming single value regression
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        prediction_level=args.prediction_level,
        learning_rate=1e-5,  # Start with a small LR for the finder
    )

    # --- Add extra parameters to the model's hparams for automatic logging ---
    # By adding these to the model's hparams, the Trainer will automatically
    # log them and, crucially, save them correctly in the checkpoint.
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.hparams.update({
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
    })
    print(f"Model has {total_params:,} total parameters ({trainable_params:,} trainable).")

    # --- LEARNING RATE TUNING PHASE ---
    print("\n--- Finding optimal learning rate ---")
    tuner_trainer = pl.Trainer(accelerator="auto", logger=False, enable_progress_bar=False)
    tuner = Tuner(tuner_trainer)
    lr_finder = tuner.lr_find(model, datamodule=data_module, num_training=100)

    fig = lr_finder.plot(suggest=True)
    fig.savefig("learning_rate_finder.png")
    print("Learning rate finder plot saved to 'learning_rate_finder.png'")

    model.hparams.learning_rate = lr_finder.suggestion()
    print(f"Using suggested learning rate: {model.hparams.learning_rate:.2e}")

    # --- SETUP FOR MAIN TRAINING ---
    print("\n--- Setting up training callbacks and logger ---")
    # Logger
    logger = TensorBoardLogger(args.log_dir, name=args.exp_name)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        filename="best-checkpoint-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        save_last=True,
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        verbose=True,
        mode="min"
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    progress_bar = RichProgressBar(refresh_rate=4)
    model_summary = ModelSummary(max_depth=-1)  # Log the full model

    # --- MAIN TRAINING PHASE ---
    print("\n--- Starting main training loop ---")
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=[
            checkpoint_callback,
            early_stop_callback,
            lr_monitor,
            progress_bar,
            model_summary,
            DeviceStatsMonitor(),
        ],
        deterministic=True,  # For reproducibility
        log_every_n_steps=10,
    )

    # The trainer will automatically log the hyperparameters from the model
    # and datamodule because `save_hyperparameters()` was called in their `__init__`.
    # This is the correct way to ensure all necessary metadata is saved.
    trainer.fit(model, data_module)

    # --- TESTING PHASE ---
    print("\n--- Training finished. Running test set. ---")
    # The trainer will automatically use the best checkpoint for testing
    test_results = trainer.test(datamodule=data_module, ckpt_path="best")
    print("\nTest Results:")
    print(test_results)

    print(f"\n--- Training and testing complete. Logs are in '{logger.log_dir}' ---")


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    arguments = get_args()
    main(arguments)
