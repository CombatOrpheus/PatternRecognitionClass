import argparse
import typing
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    StochasticWeightAveraging,
    LearningRateMonitor,
    DeviceStatsMonitor,
    ModelSummary,
)
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner

from src.HomogeneousModels import (
    GraphGNN_SPN_Model,
    NodeGNN_SPN_Model,
    BaseGNN_SPN_Model,
)
from src.PetriNets import load_spn_data_from_files, SPNAnalysisResultLabel
from src.SPNDataModule import SPNDataModule

# Set a seed for reproducibility
pl.seed_everything(42, workers=True)

# --- Constants ---
DEFAULT_LEARNING_RATE = 1e-3


def get_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a GNN model on SPN data.")

    data_group = parser.add_argument_group("Paths and Data")
    data_group.add_argument(
        "--train_file",
        type=Path,
        default=Path("Data/GridData_DS1_train_data.processed"),
        help="Path to the training data file.",
    )
    data_group.add_argument(
        "--val_file",
        type=Path,
        default=Path("Data/GridData_DS1_test_data.processed"),
        help="Path to the validation data file.",
    )
    data_group.add_argument(
        "--test_file",
        type=Path,
        default=Path("Data/GridData_DS1_all_data.processed"),
        help="Path to the test data file.",
    )
    data_group.add_argument(
        "--label",
        type=str,
        default="average_tokens_per_place",
        help="The label to predict.",
        choices=typing.get_args(SPNAnalysisResultLabel),
    )

    model_group = parser.add_argument_group("Model Hyperparameters")
    model_group.add_argument(
        "--hidden_dim",
        type=int,
        default=32,
        help="Hidden dimension size for GNN layers.",
    )
    model_group.add_argument("--num_layers_gnn", type=int, default=10, help="Number of GNN layers.")
    model_group.add_argument("--num_layers_mlp", type=int, default=3, help="Number of MLP layers.")
    model_group.add_argument(
        "--prediction_level",
        type=str,
        default="node",
        choices=["node", "graph"],
        help="Prediction task type.",
    )
    model_group.add_argument(
        "--gnn_operator",
        type=str,
        default="gcn",
        choices=["gcn", "tag", "cheb", "sgc", "ssg"],
        help="GNN Operator.",
    )

    training_group = parser.add_argument_group("Training Hyperparameters")
    training_group.add_argument("--batch_size", type=int, default=128, help="Batch size for training.")
    training_group.add_argument("--num_workers", type=int, default=3, help="Number of workers for data loading.")
    training_group.add_argument("--max_epochs", type=int, default=100, help="Maximum number of training epochs.")
    training_group.add_argument("--patience", type=int, default=10, help="Patience for early stopping.")
    training_group.add_argument(
        "--log_dir",
        type=Path,
        default="lightning_logs",
        help="Directory for logs and checkpoints.",
    )
    training_group.add_argument(
        "--exp_name",
        type=str,
        default="gnn_spn_experiment",
        help="Experiment name for the logger.",
    )
    training_group.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Set a fixed learning rate. Overrides LR finder.",
    )
    training_group.add_argument(
        "--swa_lrs",
        type=float,
        default=None,
        help="Learning rate for Stochastic Weight Averaging. If set, SWA is enabled.",
    )

    return parser.parse_args()


def setup_data_and_model(
    args: argparse.Namespace,
) -> tuple[SPNDataModule, BaseGNN_SPN_Model]:
    """Sets up the data module and the model."""
    print("--- Setting up DataModule and Model ---")
    train_spn_list = load_spn_data_from_files(args.train_file)
    val_spn_list = load_spn_data_from_files(args.val_file)
    test_spn_list = load_spn_data_from_files(args.test_file)

    if not train_spn_list:
        raise ValueError("Training data is empty. Please check the data files and paths.")

    data_module = SPNDataModule(
        label_to_predict=args.label,
        train_data_list=train_spn_list,
        val_data_list=val_spn_list,
        test_data_list=test_spn_list,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    node_feature_dim = train_spn_list[0].num_node_features
    output_dim = 1  # Single value regression target

    initial_lr = args.learning_rate if args.learning_rate is not None else DEFAULT_LEARNING_RATE

    model_class = NodeGNN_SPN_Model if args.prediction_level == "node" else GraphGNN_SPN_Model

    model = model_class(
        node_features_dim=node_feature_dim,
        out_channels=output_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers_gnn,
        num_layers_mlp=args.num_layers_mlp,
        learning_rate=initial_lr,
        gnn_operator_name=args.gnn_operator,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.hparams.update(
        {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }
    )
    print(f"Model has {total_params:,} total parameters ({trainable_params:,} trainable).")

    return data_module, model


def tune_learning_rate(model: BaseGNN_SPN_Model, data_module: SPNDataModule, args: argparse.Namespace):
    """Finds the optimal learning rate."""
    if args.learning_rate is not None:
        print(f"\n--- Using user-provided learning rate: {args.learning_rate} ---")
        return

    print("\n--- Finding optimal learning rate ---")
    tuner_trainer = pl.Trainer(accelerator="auto", logger=False, enable_progress_bar=False, max_epochs=500)
    tuner = Tuner(tuner_trainer)

    try:
        lr_finder = tuner.lr_find(model, datamodule=data_module, num_training=100)

        exp_log_dir = args.log_dir / args.exp_name
        exp_log_dir.mkdir(parents=True, exist_ok=True)
        plot_path = exp_log_dir / "learning_rate_finder.png"
        fig = lr_finder.plot(suggest=True)
        fig.savefig(plot_path)
        print(f"Learning rate finder plot saved to '{plot_path}'")

        suggested_lr = lr_finder.suggestion()
        if suggested_lr:
            model.hparams.learning_rate = suggested_lr
            print(f"Using suggested learning rate: {model.hparams.learning_rate}")
        else:
            print(f"LR finder did not suggest a rate. Using a default of {DEFAULT_LEARNING_RATE}.")
            model.hparams.learning_rate = DEFAULT_LEARNING_RATE

    except (RuntimeError, ValueError) as e:
        print(f"Error during learning rate finding: {e}")
        print(f"Could not find an optimal learning rate. Using a default of {DEFAULT_LEARNING_RATE}.")
        model.hparams.learning_rate = DEFAULT_LEARNING_RATE


def get_training_callbacks(args: argparse.Namespace) -> typing.List[pl.Callback]:
    """Returns a list of callbacks for training."""
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        filename="best-checkpoint-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        save_last=True,
    )
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=args.patience, verbose=False, mode="min")
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    model_summary = ModelSummary(max_depth=-1)

    callbacks = [
        checkpoint_callback,
        early_stop_callback,
        lr_monitor,
        model_summary,
        DeviceStatsMonitor(),
    ]

    # Conditionally add SWA if a learning rate is provided for it
    if args.swa_lrs:
        swa_callback = StochasticWeightAveraging(swa_lrs=args.swa_lrs)
        callbacks.append(swa_callback)
        print("Stochastic Weight Averaging (SWA) is enabled.")

    return callbacks


def main(args: argparse.Namespace) -> None:
    """Main function to set up, tune the learning rate, and run training."""
    args.log_dir = args.log_dir.resolve()
    args.log_dir.mkdir(parents=True, exist_ok=True)

    data_module, model = setup_data_and_model(args)
    tune_learning_rate(model, data_module, args)

    print("\n--- Setting up training callbacks and logger ---")
    logger = TensorBoardLogger(save_dir=str(args.log_dir), name=args.exp_name)
    callbacks = get_training_callbacks(args)

    print("\n--- Starting main training loop ---")
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=callbacks,
        deterministic=True,
        log_every_n_steps=10,
    )

    trainer.fit(model, data_module)

    print("\n--- Training finished. Running test set. ---")
    test_results = trainer.test(datamodule=data_module, ckpt_path="best")
    print("\nTest Results:")
    print(test_results)

    print(f"\n--- Training and testing complete. Logs are in '{logger.log_dir}' ---")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    arguments = get_args()
    main(arguments)
