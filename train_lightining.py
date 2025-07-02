from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.tuner import Tuner

from src.HomogeneousModels import FlexibleGNN_SPN_Model
from src.PetriNets import load_spn_data_from_files
from src.SPNDataModule import SPNDataModule

# Set a seed for reproducibility
pl.seed_everything(42, workers=True)


def main():
    """Main function to set up, tune the learning rate, and run training."""

    print("--- Setting up DataModule and Model ---")
    # 1. Load data and instantiate DataModule with a fixed batch size
    train_spn_list = load_spn_data_from_files(
        Path("./Data/GridData_DS1_train_data.processed")
    )
    val_spn_list = load_spn_data_from_files(Path("./Data/GridData_DS1_test_data.processed"))
    test_spn_list = load_spn_data_from_files(Path("./Data/GridData_DS2_all_data.processed"))

    data_module = SPNDataModule(
        label_to_predict="average_tokens_network",
        train_data_list=train_spn_list,
        val_data_list=val_spn_list,
        test_data_list=test_spn_list,
        batch_size=128,  # Set a fixed batch size
        num_workers=3,
    )

    # 2. Instantiate your Model with a placeholder learning rate
    model = FlexibleGNN_SPN_Model(
        node_features_dim=4,
        hidden_dim=64,
        out_channels=1,
        num_layers=8,
        prediction_level="graph",
        learning_rate=1e-5,  # Start with a small LR for the finder
    )

    # --- LEARNING RATE TUNING PHASE ---
    # Create a temporary trainer just for the tuning process
    tuner_trainer = pl.Trainer(accelerator="auto", logger=False)
    tuner = Tuner(tuner_trainer)

    print("\n--- Finding optimal learning rate ---")
    lr_finder = tuner.lr_find(model, datamodule=data_module)

    # Save the plot for inspection
    fig = lr_finder.plot(suggest=True)
    fig.savefig("learning_rate_finder.png")
    print("Learning rate finder plot saved to 'learning_rate_finder.png'")

    # Get the suggestion and update the model's learning rate
    suggested_lr = lr_finder.suggestion()
    model.learning_rate = suggested_lr
    print(f"Using suggested learning rate: {model.learning_rate:.2e}")

    # --- TRAINING PHASE ---
    print("\n--- Initializing final trainer for the main run ---")

    # Define callbacks for a robust training loop
    logger = TensorBoardLogger("lightning_logs", name="gnn_spn_tuned_lr")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", mode="min", save_top_k=1, filename="best-checkpoint"
    )
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=10, mode="min")

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="auto",
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
    )

    print("--- Starting Training with tuned learning rate ---")
    trainer.fit(model, datamodule=data_module)

    print("\n--- Starting Testing ---")
    trainer.test(model, datamodule=data_module, ckpt_path="best")


if __name__ == "__main__":
    # Recommended setting for performance on modern GPUs
    torch.set_float32_matmul_precision('high')
    main()
