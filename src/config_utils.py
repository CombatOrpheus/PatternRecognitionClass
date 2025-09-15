import argparse
from pathlib import Path

import toml


def _add_args_from_config(parser, config_dict, parent_key=""):
    """
    Recursively add arguments to the parser from a nested dictionary.
    """
    for key, value in config_dict.items():
        full_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            _add_args_from_config(parser, value, full_key)
        else:
            parser.add_argument(f"--{full_key}", type=type(value), default=value)


def load_config(cli_args: list = None) -> tuple[argparse.Namespace, Path]:
    """
    Loads configuration from a TOML file, allows overriding with command-line arguments,
    and converts all parameters in the 'io' section to pathlib.Path objects.
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=args.config, help="Path to the TOML configuration file.")
    _add_args_from_config(parser, config_dict)

    final_args = parser.parse_args(unknown) if unknown else parser.parse_args([])

    config_ns = argparse.Namespace()
    for key, value in vars(final_args).items():
        parts = key.split(".")
        d = config_ns
        for part in parts[:-1]:
            if not hasattr(d, part):
                setattr(d, part, argparse.Namespace())
            d = getattr(d, part)
        setattr(d, parts[-1], value)

    # Convert all values in the 'io' section to Path objects
    if hasattr(config_ns, "io"):
        io_dict = vars(config_ns.io)
        new_io_dict = {k: Path(v) if isinstance(v, str) else v for k, v in io_dict.items()}
        config_ns.io = argparse.Namespace(**new_io_dict)

    return config_ns, config_path
