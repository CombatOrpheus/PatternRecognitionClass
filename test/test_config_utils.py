import argparse
from pathlib import Path

import pytest

from src.config_utils import load_config


def test_load_config_defaults():
    """Tests that the default configuration is loaded correctly."""
    config, config_path = load_config(["--config", "test/dummy_config.toml"])

    assert config_path == Path("test/dummy_config.toml")

    # Check io section
    assert hasattr(config, "io")
    assert isinstance(config.io.input_dir, Path)
    assert config.io.input_dir == Path("/path/to/input")
    assert config.io.output_dir == Path("/path/to/output")

    # Check training section
    assert hasattr(config, "training")
    assert config.training.epochs == 100
    assert config.training.learning_rate == 0.001

    # Check model section
    assert hasattr(config, "model")
    assert config.model.name == "GCN"
    assert config.model.hidden_channels == 64
    assert hasattr(config.model, "specs")
    assert config.model.specs.layers == 2
    assert config.model.specs.dropout == 0.5


def test_load_config_override():
    """Tests that command-line arguments override config file settings."""
    cli_args = [
        "--config",
        "test/dummy_config.toml",
        "--training.epochs",
        "200",
        "--model.specs.dropout",
        "0.7",
    ]
    config, _ = load_config(cli_args)

    assert config.training.epochs == 200
    assert config.model.specs.dropout == 0.7
    # Check that other values remain unchanged
    assert config.training.learning_rate == 0.001
    assert config.model.name == "GCN"


def test_load_config_file_not_found():
    """Tests that a FileNotFoundError is raised for a missing config file."""
    with pytest.raises(FileNotFoundError):
        load_config(["--config", "test/non_existent_config.toml"])


def test_load_config_nested_namespace():
    """Tests that the resulting config is a nested namespace."""
    config, _ = load_config(["--config", "test/dummy_config.toml"])

    assert isinstance(config, argparse.Namespace)
    assert isinstance(config.io, argparse.Namespace)
    assert isinstance(config.training, argparse.Namespace)
    assert isinstance(config.model, argparse.Namespace)
    assert isinstance(config.model.specs, argparse.Namespace)
