import argparse
import toml
from pathlib import Path
from typing import Any, Dict

def _convert_to_path(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively converts dictionary values that are strings and represent paths to Path objects.
    """
    path_keys = ["root", "raw_data_dir", "studies_dir", "state_dict_dir", "stats_results_file", "log_dir", "cross_eval_results_file", "output_dir", "val_file", "experiment_dir", "data_dir", "output_file"]
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = _convert_to_path(v)
        elif k in path_keys and isinstance(v, str):
            d[k] = Path(v)
    return d

def _add_args_from_config(parser, config_dict, parent_key=''):
    """
    Recursively add arguments to the parser from a nested dictionary.
    """
    for key, value in config_dict.items():
        full_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            _add_args_from_config(parser, value, full_key)
        else:
            # Note: The type of the argument is inferred from the type of the default value.
            # This is a simple way to handle basic types like str, int, float, bool.
            parser.add_argument(f'--{full_key}', type=type(value), default=value)

def load_config() -> argparse.Namespace:
    """
    Loads configuration from a TOML file and allows overriding with command-line arguments.
    """
    # First pass: get the config file path
    parser = argparse.ArgumentParser(description="Configuration loader with command-line override.")
    parser.add_argument("--config", type=str, default="configs/default_config.toml", help="Path to the TOML configuration file.")

    # Use parse_known_args to avoid errors on unrecognized args
    args, unknown = parser.parse_known_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r") as f:
        config_dict = toml.load(f)

    # Second pass: create a new parser and add arguments from the config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=args.config, help="Path to the TOML configuration file.")
    _add_args_from_config(parser, config_dict)

    # This will parse all arguments, with CLI args taking precedence over defaults from config
    final_args = parser.parse_args(unknown)

    # Convert the flat namespace to a nested namespace
    nested_args = argparse.Namespace()
    for key, value in vars(final_args).items():
        parts = key.split('.')
        d = nested_args
        for part in parts[:-1]:
            if not hasattr(d, part):
                setattr(d, part, argparse.Namespace())
            d = getattr(d, part)
        setattr(d, parts[-1], value)

    # Convert path strings to Path objects
    config_with_paths = _convert_to_path(vars(nested_args))

    def to_namespace(d):
        if isinstance(d, dict):
            for key, value in d.items():
                d[key] = to_namespace(value)
            return argparse.Namespace(**d)
        return d

    return to_namespace(config_with_paths)
