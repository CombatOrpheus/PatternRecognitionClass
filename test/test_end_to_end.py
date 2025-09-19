import subprocess
import sys
import pytest
import os
import shutil

CONFIG_CONTENT = """
# Test configuration for the GNN experiments

[io]
root = "test"
raw_data_dir = ""
train_file = "sample.processed"
test_file = "sample.processed"
val_file = "sample.processed"
studies_dir = "test/optuna_studies"
state_dict_dir = "test/results/state_dicts"
stats_results_file = "test/results/statistical_results.parquet"
cross_eval_results_file = "test/results/cross_dataset_evaluation.parquet"
output_dir = "test/results/analysis_plots"
log_dir = "test/lightning_logs"

[model]
label = "average_tokens_network"
prediction_level = "graph"
gnn_operator = ["gcn"]

[training]
num_runs = 1
num_workers = 0
max_epochs = 2
patience = 2
val_split = 0.5
to_be_compiled = false

[optimization]
n_trials = 2
timeout = 60

[analysis]
main_metrics = ["test/rmse", "test/mae", "val/rmse"]
cross_val_metrics = ["test/rmse", "test/mae"]
generate_critical_diagram = false
generate_performance_complexity_plot = false
generate_cross_eval_heatmap = false

[cross_validation]
datasets = []
"""

@pytest.fixture(scope="module")
def setup_and_teardown():
    """
    This fixture sets up the necessary environment for the end-to-end test
    and tears it down afterward.
    """
    # Create the config file
    with open("test/test_config.toml", "w") as f:
        f.write(CONFIG_CONTENT)

    # Yield to the test function
    yield

    # Teardown: Clean up created directories and files
    os.remove("test/test_config.toml")
    dirs_to_remove = [
        "test/optuna_studies",
        "test/results",
        "test/lightning_logs",
        "optuna_logs",
        "test/processed",
    ]
    for dir_path in dirs_to_remove:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)

def test_end_to_end(setup_and_teardown):
    """
    Runs the main script with a custom configuration for a quick end-to-end test.
    """
    config_path = "test/test_config.toml"

    command = [
        sys.executable,
        "scripts/main.py",
        "--config",
        config_path,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "."

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, f"Script failed with output:\n{result.stdout}\n{result.stderr}"
    assert os.path.exists("test/results/statistical_results.parquet")
