# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Interactive Exploration of SPN Datasets
#
# This notebook provides an interactive environment for exploring and comparing the characteristics of various Stochastic Petri Net (SPN) datasets.
#
# ## Instructions
#
# 1. **Run the setup cells**: Execute the cells under the "Setup and Data Loading" section to load the necessary libraries and process the datasets.
# 2. **Select datasets**: Use the interactive widget to choose the datasets you want to compare.
# 3. **View the plots**: The visualizations will automatically update to reflect your selection.
#

# %% [markdown]
# ---
# ## 1. Setup and Data Loading
#
# This section imports the required libraries and defines the functions to extract and process the SPN data from the `Reduced Data.zip` archive.

# %%
# General utilities
import os
import zipfile
from pathlib import Path
import shutil

# Data manipulation
import pandas as pd
import numpy as np

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Interactive widgets
import ipywidgets as widgets
from ipywidgets import interact

# Import custom modules from the project
# We need to add the project's root directory to the Python path
import sys
sys.path.append('..')
from src.PetriNets import SPNData, load_spn_data_lazily

print("Libraries imported successfully.")

# %%
# Define the path to the zip file and the temporary extraction directory
DATA_ZIP_PATH = Path("../Reduced Data.zip")
TEMP_EXTRACT_DIR = Path("./temp_data")

# Create the temporary directory if it doesn't exist
if TEMP_EXTRACT_DIR.exists():
    shutil.rmtree(TEMP_EXTRACT_DIR)
TEMP_EXTRACT_DIR.mkdir()

# List of all dataset files to process from the zip archive
# We focus on the '_all_data' files for a complete overview
DATASET_FILES = [
    "GridData_DS1_all_data.processed",
    "GridData_DS2_all_data.processed",
    "GridData_DS3_all_data.processed",
    "GridData_DS4_all_data.processed",
    "GridData_DS5_all_data.processed",
    "RandDataDS1_all_data.processed",
    # For RandData_DS2, we'll combine train and test sets
    "RandData_DS2_train_data.processed",
    "RandData_DS2_test_data.processed",
]

# A manually combined dataset name for RandData_DS2
COMBINED_RANDDATA_2_NAME = "RandData_DS2_all_data"

# Dictionary to hold the raw SPNData objects for each dataset
all_spn_data = {}

print(f"Starting data extraction from {DATA_ZIP_PATH}...")

with zipfile.ZipFile(DATA_ZIP_PATH, 'r') as zip_ref:
    # Get a list of all files in the zip archive
    zip_file_list = zip_ref.namelist()

    # Create a mapping for the combined RandData_DS2
    dataset_groups = {
        "GridData_DS1_all_data": ["GridData_DS1_all_data.processed"],
        "GridData_DS2_all_data": ["GridData_DS2_all_data.processed"],
        "GridData_DS3_all_data": ["GridData_DS3_all_data.processed"],
        "GridData_DS4_all_data": ["GridData_DS4_all_data.processed"],
        "GridData_DS5_all_data": ["GridData_DS5_all_data.processed"],
        "RandDataDS1_all_data": ["RandDataDS1_all_data.processed"],
        COMBINED_RANDDATA_2_NAME: [
            "RandData_DS2_train_data.processed",
            "RandData_DS2_test_data.processed",
        ],
    }

    for name, files in dataset_groups.items():
        print(f"Processing dataset: {name}")

        # Check if the required files exist in the zip archive
        missing_files = [f for f in files if f not in zip_file_list]
        if missing_files:
            print(f"Warning: Missing files {missing_files} for dataset {name}. Skipping.")
            continue

        # Extract and load data for the current group
        extracted_paths = []
        for file in files:
            zip_ref.extract(file, TEMP_EXTRACT_DIR)
            extracted_paths.append(TEMP_EXTRACT_DIR / file)

        # Load the data using the lazy loader and convert to a list
        spn_list = list(load_spn_data_lazily(extracted_paths))
        all_spn_data[name] = spn_list

        # Clean up the extracted files to save space
        for path in extracted_paths:
            path.unlink()

# Clean up the temporary directory
shutil.rmtree(TEMP_EXTRACT_DIR)

print("Data loading complete. The following datasets are available:")
for name, data in all_spn_data.items():
    print(f"- {name}: {len(data)} SPNs")

# %%
# Now, let's compute the summary statistics for each dataset
summary_data = []

for name, spns in all_spn_data.items():
    if not spns:
        continue

    # Get structural properties from the first SPN (assuming they are uniform)
    num_places = [spn.spn.shape[0] for spn in spns]
    num_transitions = [spn.spn.shape[1] // 2 for spn in spns]

    # Get analysis results (stochastic properties)
    avg_firing_rates = np.concatenate([spn.average_firing_rates for spn in spns if spn.average_firing_rates.ndim > 0])
    avg_tokens_per_place = np.concatenate([spn.average_tokens_per_place for spn in spns if spn.average_tokens_per_place.ndim > 0])
    avg_tokens_network = [spn.average_tokens_network for spn in spns]

    # Reachability graph properties
    num_reachability_nodes = [len(spn.reachability_graph_nodes) for spn in spns]
    num_reachability_edges = [len(spn.reachability_graph_edges) for spn in spns]

    summary_data.append({
        "Dataset": name,
        "Number of SPNs": len(spns),
        "Mean Places": np.mean(num_places),
        "Mean Transitions": np.mean(num_transitions),
        "Mean Firing Rate": np.mean(avg_firing_rates),
        "Std Firing Rate": np.std(avg_firing_rates),
        "Mean Avg Tokens per Place": np.mean(avg_tokens_per_place),
        "Mean Total Tokens": np.mean(avg_tokens_network),
        "Std Total Tokens": np.std(avg_tokens_network),
        "Mean Reachability Nodes": np.mean(num_reachability_nodes),
        "Mean Reachability Edges": np.mean(num_reachability_edges),
    })

summary_df = pd.DataFrame(summary_data)
print("\nSummary statistics computed:")
display(summary_df)

# %% [markdown]
# ---
# ## 2. Interactive Dataset Comparison
#
# Use the widget below to select the datasets you wish to analyze. The plots will update automatically.

# %%
# Create the interactive widget for dataset selection
dataset_selector = widgets.SelectMultiple(
    options=summary_df['Dataset'].tolist(),
    value=[summary_df['Dataset'].tolist()[0]],  # Default to the first dataset
    description='Datasets',
    disabled=False,
    layout=widgets.Layout(width='50%')
)

# Create a dropdown to select the type of plot
plot_type_selector = widgets.Dropdown(
    options=['Summary Statistics', 'Firing Rate Distribution', 'Total Tokens Distribution'],
    value='Summary Statistics',
    description='Plot Type:',
    disabled=False,
)

# Create a container for the output plots
plot_output = widgets.Output()

def plot_selected_data(datasets, plot_type):
    """
    This function is triggered on selection change.
    It clears the previous output and generates new plots based on the selection.
    """
    plot_output.clear_output(wait=True)

    if not datasets:
        with plot_output:
            print("Please select at least one dataset to visualize.")
        return

    with plot_output:
        if plot_type == 'Summary Statistics':
            # Bar charts for comparing summary metrics
            filtered_df = summary_df[summary_df['Dataset'].isin(datasets)]

            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Mean Places and Transitions",
                    "Mean Total Tokens in Network",
                    "Mean Reachability Graph Nodes",
                    "Mean Firing Rate"
                ),
                vertical_spacing=0.3
            )

            # Plot 1: Places and Transitions
            fig.add_trace(go.Bar(name='Places', x=filtered_df['Dataset'], y=filtered_df['Mean Places']), row=1, col=1)
            fig.add_trace(go.Bar(name='Transitions', x=filtered_df['Dataset'], y=filtered_df['Mean Transitions']), row=1, col=1)

            # Plot 2: Total Tokens
            fig.add_trace(go.Bar(name='Total Tokens', x=filtered_df['Dataset'], y=filtered_df['Mean Total Tokens']), row=1, col=2)

            # Plot 3: Reachability Nodes
            fig.add_trace(go.Bar(name='RG Nodes', x=filtered_df['Dataset'], y=filtered_df['Mean Reachability Nodes']), row=2, col=1)

            # Plot 4: Firing Rate
            fig.add_trace(go.Bar(name='Firing Rate', x=filtered_df['Dataset'], y=filtered_df['Mean Firing Rate']), row=2, col=2)

            fig.update_layout(height=700, title_text="Summary Statisics Comparison", showlegend=False)
            fig.show()

        elif plot_type == 'Firing Rate Distribution':
            # Box plots for distribution of average firing rates
            fig = go.Figure()
            for dataset_name in datasets:
                rates = np.concatenate([spn.average_firing_rates for spn in all_spn_data[dataset_name] if spn.average_firing_rates.ndim > 0])
                fig.add_trace(go.Box(y=rates, name=dataset_name))

            fig.update_layout(title_text="Distribution of Average Firing Rates per Transition")
            fig.show()

        elif plot_type == 'Total Tokens Distribution':
            # Box plots for distribution of average total tokens
            fig = go.Figure()
            for dataset_name in datasets:
                tokens = [spn.average_tokens_network for spn in all_spn_data[dataset_name]]
                fig.add_trace(go.Box(y=tokens, name=dataset_name))

            fig.update_layout(title_text="Distribution of Average Total Tokens per Network")
            fig.show()

# Link the widgets to the plotting function
def on_selection_change(change):
    plot_selected_data(dataset_selector.value, plot_type_selector.value)

dataset_selector.observe(on_selection_change, names='value')
plot_type_selector.observe(on_selection_change, names='value')

# Display the widgets and initial plot
print("Select datasets and a plot type to begin analysis.")
display(widgets.VBox([widgets.HBox([dataset_selector, plot_type_selector]), plot_output]))

# Trigger the first plot draw
plot_selected_data(dataset_selector.value, plot_type_selector.value)
