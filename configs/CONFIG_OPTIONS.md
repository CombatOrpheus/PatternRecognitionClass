# Configuration Options

This file provides a detailed explanation of all the parameters available in the `default_config.toml` file.

## `[io]` - Input/Output and Paths

This section contains all parameters related to file paths and experiment naming.

- `root`: The root directory for storing processed datasets.
  - Default: `"processed_data/SPN_homogeneous"`
- `raw_data_dir`: The directory where the raw `.processed` files are located.
  - Default: `"Data"`
- `train_file`: The filename of the training data.
  - Default: `"GridData_DS1_train_data.processed"`
- `test_file`: The filename for the final, held-out test set.
  - Default: `"GridData_DS1_all_data.processed"`
- `val_file`: The filename of the validation data.
  - Default: `"../Data/GridData_DS1_test_data.processed"`
- `studies_dir`: The directory containing Optuna study `.db` files.
  - Default: `"optuna_studies"`
- `state_dict_dir`: The directory to save model artifacts (state dictionaries).
  - Default: `"results/state_dicts"`
- `stats_results_file`: The path to the output Parquet file for statistical results.
  - Default: `"results/statistical_results.parquet"`
- `cross_eval_results_file`: The path to the output Parquet file for cross-dataset evaluation results.
  - Default: `"results/cross_dataset_evaluation.parquet"`
- `output_dir`: The directory to save analysis plots.
  - Default: `"results/analysis_plots"`
- `log_dir`: The directory to save Lightning logs.
  - Default: `"lightning_logs"`
- `exp_name`: The name of the experiment.
  - Default: `"gnn_spn_experiment"`
- `experiment_dir`: The path to the main experiment log directory for testing models.
  - Default: `"lightning_logs/gnn_spn_experiment"`
- `data_dir`: The directory containing the `.processed` test files for the test script.
  - Default: `"../Data"`
- `output_file`: The path to the output Parquet file for the test script results.
  - Default: `"../results/cross_dataset_evaluation.parquet"`

## `[model]` - Homogeneous Model Configuration

This section contains parameters for the homogeneous GNN models.

- `label`: The label to predict.
  - Default: `"average_tokens_network"`
  - Choices: `["average_firing_rates", "steady_state_probabilities", "token_probability_density_function", "average_tokens_per_place", "average_tokens_network"]`
- `prediction_level`: The prediction level, either "node" or "graph".
  - Default: `"graph"`
  - Choices: `["node", "graph"]`
- `gnn_operator`: The GNN operator to use.
  - Default: `"gcn"`
  - Choices: `["gcn", "tag", "cheb", "sgc", "ssg", "mixed"]`

## `[training]` - Homogeneous Model Training

This section contains parameters for training the homogeneous GNN models.

- `num_runs`: The number of statistical runs for each selected study.
  - Default: `30`
- `num_workers`: The number of workers for the DataLoader.
  - Default: `3`
- `max_epochs`: The maximum number of epochs for training.
  - Default: `100`
- `patience`: The patience for early stopping.
  - Default: `10`
- `val_split`: The fraction of training data to use for validation.
  - Default: `0.2`

## `[optimization]` - Homogeneous Model Optimization

This section contains parameters for hyperparameter optimization of the homogeneous GNN models.

- `all_operators`: If `true`, runs optimization for all operators.
  - Default: `false`
- `n_trials`: The number of optimization trials.
  - Default: `100`
- `timeout`: The timeout for the study in seconds.
  - Default: `7200`
- `study_name`: The base name for the Optuna study.
  - Default: `"gnn_spn_optimization"`

## `[hetero_model]` - Heterogeneous Model Configuration

This section contains parameters for the heterogeneous GNN models.

- `gnn_operator`: The Heterogeneous GNN operator to use.
  - Default: `"rgat"`
  - Choices: `["rgat", "heat"]`

## `[hetero_training]` - Heterogeneous Model Training

This section contains parameters for training the heterogeneous GNN models.

- `max_epochs`: The maximum number of epochs for training.
  - Default: `100`
- `patience`: The patience for early stopping.
  - Default: `10`
- `num_workers`: The number of workers for the DataLoader.
  - Default: `3`

## `[hetero_optimization]` - Heterogeneous Model Optimization

This section contains parameters for hyperparameter optimization of the heterogeneous GNN models.

- `all_operators`: If `true`, runs optimization for all available operators.
  - Default: `false`
- `n_trials`: The number of optimization trials.
  - Default: `100`
- `timeout`: The timeout for the study in seconds.
  - Default: `7200`
- `study_name`: The base name for the Optuna study.
  - Default: `"hetero_gnn_spn_optimization"`
