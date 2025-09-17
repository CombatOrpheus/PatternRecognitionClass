from pathlib import Path

import pytest
from torch.utils.data import Subset

from src.ReachabilityGraphDataModule import ReachabilityGraphDataModule


@pytest.fixture
def reachability_data_dir(tmp_path):
    """Creates a temporary directory structure for reachability graph data."""
    root_dir = tmp_path / "reachability_data"
    raw_dir = root_dir / "raw"
    raw_dir.mkdir(parents=True)

    # Create dummy train, val, and test files
    sample_content = Path("test/sample.processed").read_text()
    (raw_dir / "train.processed").write_text(sample_content)
    (raw_dir / "val.processed").write_text(sample_content)
    (raw_dir / "test.processed").write_text(sample_content)

    return {
        "root": str(root_dir),
        "train_file": str(raw_dir / "train.processed"),
        "val_file": str(raw_dir / "val.processed"),
        "test_file": str(raw_dir / "test.processed"),
    }


def test_reachability_datamodule_init(reachability_data_dir):
    """Tests the initialization of the ReachabilityGraphDataModule."""
    dm = ReachabilityGraphDataModule(
        root=reachability_data_dir["root"],
        train_file=reachability_data_dir["train_file"],
        val_file=reachability_data_dir["val_file"],
        test_file=reachability_data_dir["test_file"],
        label_to_predict="average_tokens_network",
        batch_size=16,
    )
    assert dm.hparams.batch_size == 16
    assert dm.hparams.label_to_predict == "average_tokens_network"


def test_reachability_datamodule_setup(reachability_data_dir):
    """Tests the setup method for fit and test stages."""
    dm = ReachabilityGraphDataModule(
        root=reachability_data_dir["root"],
        train_file=reachability_data_dir["train_file"],
        val_file=reachability_data_dir["val_file"],
        test_file=reachability_data_dir["test_file"],
        label_to_predict="average_tokens_network",
        val_split=0.3,
    )

    # Test 'fit' stage
    dm.setup(stage="fit")
    assert isinstance(dm.train_dataset, Subset)
    assert isinstance(dm.val_dataset, Subset)
    assert dm.label_scaler is not None

    # Test 'test' stage
    dm.setup(stage="test")
    assert dm.test_dataset is not None


def test_reachability_dataloaders(reachability_data_dir):
    """Tests that the dataloaders are created correctly."""
    dm = ReachabilityGraphDataModule(
        root=reachability_data_dir["root"],
        train_file=reachability_data_dir["train_file"],
        val_file=reachability_data_dir["val_file"],
        test_file=reachability_data_dir["test_file"],
        label_to_predict="average_tokens_network",
        batch_size=8,
    )
    dm.setup()
    assert dm.train_dataloader().batch_size == 8
    assert dm.val_dataloader().batch_size == 8
    assert dm.test_dataloader().batch_size == 8
