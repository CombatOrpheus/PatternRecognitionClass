import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scikit_posthocs as sp
import scipy.stats as ss
import seaborn as sns


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
    print(f"--- Analysis outputs will be saved to: {output_dir} ---")

    # Load the results from the Parquet file
    df = pd.read_parquet(results_file)
    print("--- Results data loaded successfully ---")
    print(df.info())

    # 1. Generate and save the descriptive statistics table
    print(f"\n--- Generating descriptive statistics (grouped by 'gnn_operator') ---")
    summary_stats = (
        df.groupby("gnn_operator")
        .agg(
            mean_mae=("test_mae", "mean"),
            std_mae=("test_mae", "std"),
            mean_rmse=("test_rmse", "mean"),
            std_rmse=("test_rmse", "std"),
            #mean_mape=("test_mape", "mean"),
            #std_mape=("test_mape", "std"),
        )
        .reset_index()
    )

    summary_table_path = output_dir / "summary_statistics.csv"
    summary_stats.to_csv(summary_table_path, index=False, float_format="%.4f")
    print(f"Descriptive statistics table saved to: {summary_table_path}")
    print(summary_stats)

    # 2. Perform statistical tests
    print(f"\n--- Performing statistical tests on metric: '{metric}' ---")

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

    # Post-hoc Conover test
    posthoc_results = sp.posthoc_conover_friedman(df, melted=True, y_col=metric, block_col="run_id", block_id_col="run_id", group_col="gnn_operator")

    # 3. Generate and save plots
    print("\n--- Generating and saving plots ---")

    # Critical Difference Diagram
    avg_rank = df.groupby("run_id")[metric].rank(pct=True).groupby(df["gnn_operator"]).mean()

    plt.figure(figsize=(10, max(4, len(model_names) * 0.5)))  # Adjust height based on number of models
    sp.critical_difference_diagram(
        ranks=avg_rank,
        sig_matrix=posthoc_results,
        label_fmt_left="{label} [{rank:.3f}]  ",
        label_fmt_right="  [{rank:.3f}] {label}",
    )
    plt.title(f"Critical Difference Diagram for {metric}")
    cd_diagram_path = output_dir / "critical_difference_diagram.svg"
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
    box_plot_path = output_dir / "performance_distribution_boxplot.svg"
    plt.savefig(box_plot_path)
    plt.close()
    print(f"Performance distribution box plot saved to: {box_plot_path}")

    print("\n--- Analysis complete. ---")


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
        default=Path("results/analysis"),
        help="Directory to save the generated tables and plots.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="test_mae",
        choices=["test_mae", "test_rmse", "test_mape", "test_medae", "test_r2"],
        help="The primary metric to use for statistical comparisons and ranking.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_analysis_args()
    analyze_results(
        results_file=args.results_file,
        output_dir=args.output_dir,
        metric=args.metric,
    )
