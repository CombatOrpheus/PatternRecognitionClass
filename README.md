# GNN for SPN Analysis

This project uses Graph Neural Networks (GNNs) to analyze and predict properties of Stochastic Petri Nets (SPNs).

## Configuration

All scripts in this project are configured using a centralized TOML configuration file. The default configuration is located at `configs/default_config.toml`.

### Custom Configuration

You can create your own configuration file by copying and modifying the default one. To run a script with a custom configuration, use the `--config` argument:

```bash
python scripts/train_model.py --config /path/to/your/custom_config.toml
```

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

### Analyze Results

To analyze the results and generate plots:
```bash
python scripts/analyze_and_plot.py
```

### Test a Model

To evaluate trained models on a directory of test data:
```bash
python scripts/test_model.py
```
