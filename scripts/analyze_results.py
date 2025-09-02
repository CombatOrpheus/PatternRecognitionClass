import argparse
import itertools  # Import itertools for pairwise combinations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scikit_posthocs as sp
import scipy.stats as ss
import seaborn as sns

# Define the full list of test metrics available from the model
ALL_TEST_METRICS = [
    "test_mae",
    "test_mse",
    "test_rmse",
    "test_rse",
    "test_rrse",
    "test_r2",
    "test_mape",
]


def analyze_results(results_file: Path, output_dir: Path, metric: str = "test_mae"):
    """
    Loads experimental results from a Parquet file, performs statistical analysis,
    and generates summary tables and plots.

    Args:
        results_file (Path): Path to the input .parquet file containing the results.
        output_dir (Path): Directory to save the analysis outputs (tables and plots).
        metric (str): The primary performance metric to use for statistical testing.
    """
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found at: {results_file}")

    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"--- Analysis for metric '{metric}' will be saved to: {output_dir} ---")

    # Load the results from the Parquet file
    df = pd.read_parquet(results_file)
    print("--- Results data loaded successfully ---")

    # 1. Generate and save the descriptive statistics table for ALL available metrics
    print(f"\n--- Generating descriptive statistics (grouped by 'gnn_operator') ---")

    # Find which of the possible metrics are actually in the dataframe
    available_metrics = [m for m in ALL_TEST_METRICS if m in df.columns]
    if not available_metrics:
        print("No recognized metric columns found in the results file. Skipping descriptive statistics.")
    else:
        # Create the aggregation dictionary dynamically
        agg_dict = {}
        for m in available_metrics:
            metric_suffix = m.replace("test_", "")  # e.g., "mae"
            agg_dict[f"mean_{metric_suffix}"] = (m, "mean")
            agg_dict[f"std_{metric_suffix}"] = (m, "std")

        summary_stats = df.groupby("gnn_operator").agg(**agg_dict).reset_index()

        summary_table_path = output_dir / "summary_statistics.csv"
        summary_stats.to_csv(summary_table_path, index=False, float_format="%.4f")
        print(f"Descriptive statistics table for all metrics saved to: {summary_table_path}")
        print(summary_stats)

    # 2. Perform statistical tests for the SPECIFIED metric
    print(f"\n--- Performing statistical tests on primary metric: '{metric}' ---")

    if metric not in df.columns:
        print(f"Primary metric '{metric}' not found in results file. Skipping statistical tests and plots.")
        return

    # Prepare data for Friedman test (list of arrays, one for each group)
    model_groups = [group[metric].values for name, group in df.groupby("gnn_operator")]
    model_names = df["gnn_operator"].unique()

    if len(model_groups) < 2:
        print("Need at least two models to perform statistical tests. Skipping.")
        return

    # Friedman test
    try:
        friedman_stat, p_value = ss.friedmanchisquare(*model_groups)
        print(f"Friedman Test: statistic={friedman_stat:.4f}, p-value={p_value:.4g}")
        if p_value >= 0.05:
            print("Friedman test is not significant (p >= 0.05). Post-hoc tests may not be meaningful.")
    except ValueError as e:
        print(f"Could not perform Friedman test. Reason: {e}")
        return

    # Wilcoxon signed-rank test for pairwise comparisons
    print(f"\n--- Performing Wilcoxon signed-rank tests on metric: '{metric}' ---")
    print("This test checks for significant differences between pairs of models.")

    # Create a DataFrame to store the p-values
    wilcoxon_results = pd.DataFrame(index=model_names, columns=model_names)

    # Get the performance data for each model
    model_data = {name: data[metric].values for name, data in df.groupby("gnn_operator")}

    # Perform pairwise Wilcoxon tests
    for model1, model2 in itertools.combinations(model_names, 2):
        try:
            stat, p_value = ss.wilcoxon(model_data[model1], model_data[model2])
            wilcoxon_results.loc[model1, model2] = p_value
            wilcoxon_results.loc[model2, model1] = p_value
            print(f"  {model1} vs {model2}: statistic={stat:.4f}, p-value={p_value:.4g}")
        except ValueError as e:
            print(f"  Could not perform Wilcoxon test for {model1} vs {model2}. Reason: {e}")

    # Save the Wilcoxon results to a CSV file
    wilcoxon_table_path = output_dir / f"wilcoxon_pvalues_{metric}.csv"
    wilcoxon_results.to_csv(wilcoxon_table_path, float_format="%.4g")
    print(f"Wilcoxon p-value table saved to: {wilcoxon_table_path}")

    # Post-hoc Conover test
    posthoc_results = sp.posthoc_conover_friedman(
        df, melted=True, y_col=metric, group_col="gnn_operator", block_col="run_id", block_id_col="run_id"
    )

    # 3. Generate and save plots for the SPECIFIED metric
    print(f"\n--- Generating and saving plots for '{metric}' ---")

    # Critical Difference Diagram
    avg_rank = df.groupby("run_id")[metric].rank().groupby(df["gnn_operator"]).mean()

    plt.figure(figsize=(10, max(4, len(model_names) * 0.5)))  # Adjust height based on number of models
    sp.critical_difference_diagram(
        ranks=avg_rank,
        sig_matrix=posthoc_results,
        label_fmt_left="{label} [{rank:.3f}]  ",
        label_fmt_right="  [{rank:.3f}] {label}",
    )
    plt.title(f"Critical Difference Diagram for {metric}")
    cd_diagram_path = output_dir / f"cd_diagram_{metric}.svg"
    plt.savefig(cd_diagram_path, bbox_inches="tight")
    plt.close()
    print(f"Critical difference diagram saved to: {cd_diagram_path}")

    # Box Plots for Performance Distribution
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=df, x="gnn_operator", y=metric)
    sns.stripplot(data=df, x="gnn_operator", y=metric, color=".25", size=3)
    plt.title(f"Performance Distribution of Models ({metric})")
    plt.ylabel(metric.replace("_", " ").title())
    plt.xlabel("GNN Operator")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    box_plot_path = output_dir / f"boxplot_{metric}.svg"
    plt.savefig(box_plot_path)
    plt.close()
    print(f"Performance distribution box plot saved to: {box_plot_path}")

    print(f"\n--- Analysis for '{metric}' complete. ---")


def get_analysis_args():
    """Parses command-line arguments for the analysis script."""
    parser = argparse.ArgumentParser(description="Analyze GNN model performance statistics from experimental results.")
    parser.add_argument(
        "results_file",
        type=Path,
        help="Path to the .parquet file containing the aggregated experimental results.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("../results/analysis"),
        help="Directory to save the generated tables and plots.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="test_mae",
        choices=ALL_TEST_METRICS,
        help="The primary metric to use for statistical comparisons and ranking.",
    )
    parser.add_argument(
        "--all-metrics",
        action="store_true",
        help="If specified, runs the analysis for all available metrics, creating a subdirectory for each.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_analysis_args()

    if args.all_metrics:
        print("--- Running analysis for all available metrics ---")
        # Load the dataframe once to see which metrics are available
        if not args.results_file.exists():
            raise FileNotFoundError(f"Results file not found at: {args.results_file}")
        df = pd.read_parquet(args.results_file)

        metrics_to_run = [m for m in ALL_TEST_METRICS if m in df.columns]
        print(f"Found metrics in file: {metrics_to_run}")

        for metric in metrics_to_run:
            # Create a dedicated output directory for each metric's analysis
            metric_output_dir = args.output_dir / metric
            analyze_results(
                results_file=args.results_file,
                output_dir=metric_output_dir,
                metric=metric,
            )
            print("-" * 80)
    else:
        # Run analysis only for the single, specified metric
        analyze_results(
            results_file=args.results_file,
            output_dir=args.output_dir,
            metric=args.metric,
        )
