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

def load_config(config_path: Path) -> argparse.Namespace:
    """
    Loads a TOML configuration file and returns it as a namespace.
    Converts path-like strings to Path objects.
    """
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r") as f:
        config_dict = toml.load(f)

    # Recursively convert nested dictionaries to namespaces for attribute access
    def to_namespace(d):
        if isinstance(d, dict):
            for key, value in d.items():
                d[key] = to_namespace(value)
            return argparse.Namespace(**d)
        return d

    config_with_paths = _convert_to_path(config_dict)
    return to_namespace(config_with_paths)
