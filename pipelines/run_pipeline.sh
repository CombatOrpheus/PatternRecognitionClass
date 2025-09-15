#!/bin/bash

# ===================================================================================
# Pipeline Script for Homogeneous GNN SPN Experiments
# This script orchestrates the entire workflow:
# 1. Runs hyperparameter optimization based on the operators listed in the config.
# 2. Kicks off final model training and evaluation for the same operators.
# 3. Runs cross-validation on the trained models against all datasets.
# 4. Generates final analysis plots and statistical summaries.
#
# The entire pipeline is non-interactive and driven by the specified config file.
#
# Usage:
#   bash pipelines/run_pipeline.sh [path/to/config.toml]
#
# If no config file is provided, it defaults to 'configs/default_config.toml'.
# ===================================================================================

# --- Configuration ---
# Exit immediately if a command exits with a non-zero status.
set -e

# Use the provided config file or default to 'configs/default_config.toml'
CONFIG_FILE="${1:-configs/default_config.toml}"

echo "--- Using Configuration File: $CONFIG_FILE ---"

# --- Step 1: Hyperparameter Optimization (Homogeneous) ---
echo -e "\n\n--- STEP 1: Running Homogeneous Hyperparameter Optimization ---"
uv run python -m scripts.optimize_hyperparameters --config "$CONFIG_FILE"
echo "--- Homogeneous Optimization Complete ---"

# --- Step 2: Train Final Models ---
echo -e "\n\n--- STEP 2: Starting Final Model Training ---"
echo "The following script will automatically find and train the studies matching the config."
uv run python -m scripts.train_model --config "$CONFIG_FILE"
echo "--- Final Model Training Complete ---"


# --- Step 3: Cross-Validate Final Models ---
echo -e "\n\n--- STEP 3: Running Cross-Validation on Trained Models ---"
uv run python -m scripts.cross_validate_model --config "$CONFIG_FILE"
echo "--- Cross-Validation Complete ---"

# --- Step 4: Analyze and Plot Results ---
echo -e "\n\n--- STEP 4: Generating Analysis and Plots ---"
uv run python -m scripts.analyze_and_plot --config "$CONFIG_FILE"
echo "--- Analysis Complete ---"

echo -e "\n\n--- Pipeline Finished Successfully! ---"