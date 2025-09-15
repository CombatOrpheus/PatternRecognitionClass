"""This module provides utilities for loading and managing configurations.

It allows for loading settings from a TOML file and overriding them with
command-line arguments, providing a flexible configuration system. The loaded
configuration is returned as a nested namespace for easy access.
"""
import argparse
from pathlib import Path
from typing import Optional, Tuple

import toml


def _add_args_from_config(parser: argparse.ArgumentParser, config_dict: dict, parent_key: str = ""):
    """Recursively adds arguments to an argparse parser from a nested dictionary.

    This function traverses a dictionary and adds each key-value pair as a
    command-line argument. Nested keys are flattened with a dot notation
    (e.g., `parent.child.key`).

    Args:
        parser: The argparse.ArgumentParser instance to which arguments will be added.
        config_dict: The dictionary of configuration values.
        parent_key: The base key for creating nested argument names.
    """
    for key, value in config_dict.items():
        full_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            _add_args_from_config(parser, value, full_key)
        else:
            parser.add_argument(f"--{full_key}", type=type(value), default=value)


def load_config(cli_args: Optional[list] = None) -> Tuple[argparse.Namespace, Path]:
    """Loads configuration from a TOML file and overrides with command-line arguments.

    This function first parses the `--config` argument to find the TOML file.
    It then loads the TOML file, creates command-line arguments for all its
    settings, and re-parses the arguments to allow overrides. Finally, it
    converts all paths in the 'io' section to `pathlib.Path` objects.

    Args:
        cli_args: A list of command-line arguments to parse. If None, `sys.argv`
                  is used.

    Returns:
        A tuple containing:
        - A nested argparse.Namespace with the final configuration.
        - The path to the loaded configuration file.

    Raises:
        FileNotFoundError: If the specified configuration file does not exist.
    """
    parser = argparse.ArgumentParser(description="Configuration loader with command-line override.")
    parser.add_argument(
        "--config", type=str, default="configs/default_config.toml", help="Path to the TOML configuration file."
    )

    args, unknown = parser.parse_known_args(cli_args)

    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r") as f:
        config_dict = toml.load(f)

    # Create a new parser that includes all options from the config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=args.config, help="Path to the TOML configuration file.")
    _add_args_from_config(parser, config_dict)

    # Parse the remaining arguments, which will override the defaults from the config
    final_args = parser.parse_args(unknown)

    # Convert the flat namespace into a nested namespace
    config_ns = argparse.Namespace()
    for key, value in vars(final_args).items():
        parts = key.split(".")
        d = config_ns
        for part in parts[:-1]:
            if not hasattr(d, part):
                setattr(d, part, argparse.Namespace())
            d = getattr(d, part)
        setattr(d, parts[-1], value)

    # Convert all string paths in the 'io' section to Path objects
    if hasattr(config_ns, "io"):
        io_dict = vars(config_ns.io)
        new_io_dict = {k: Path(v) if isinstance(v, str) else v for k, v in io_dict.items()}
        config_ns.io = argparse.Namespace(**new_io_dict)

    return config_ns, config_path
