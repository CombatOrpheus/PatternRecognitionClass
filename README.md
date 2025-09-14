# GNN for SPN Analysis

This project uses Graph Neural Networks (GNNs) to analyze and predict properties of Stochastic Petri Nets (SPNs).

## Configuration

All scripts in this project are configured using a centralized TOML configuration file. The default configuration is located at `configs/default_config.toml`.

For a detailed explanation of all available parameters, please see the [`configs/CONFIG_OPTIONS.md`](configs/CONFIG_OPTIONS.md) file.

### Custom Configuration

You can create your own configuration file by copying and modifying the default one. To run a script with a custom configuration, use the `--config` argument:

```bash
python scripts/train_model.py --config /path/to/your/custom_config.toml
```

### Overriding Parameters via Command Line

You can also override any parameter from the configuration file directly from the command line. The command-line arguments are named based on their section and key in the TOML file.

For example, to override the number of epochs in the `training` section, you can use the `--training.max_epochs` argument:

```bash
python scripts/train_model.py --training.max_epochs 200
```

This will run the training for 200 epochs, while using the other parameters from the default configuration file.

## Usage

Here are some examples of how to run the different scripts:

### Train a Model

To run the training script with the default configuration:
```bash
python scripts/train_model.py
```

### Optimize Hyperparameters

To run the hyperparameter optimization for the homogeneous models:
```bash
python scripts/optimize_hyperparameters.py
```

To run the hyperparameter optimization for the heterogeneous models:
```bash
python scripts/optimize_hyperparameters_hetero.py
```

### Test a Model

To evaluate trained models on a directory of test data:
```bash
python scripts/test_model.py
```
