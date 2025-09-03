import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scikit_posthocs as sp
import scipy.stats as ss
import seaborn as sns


def analyze_and_plot_results(
    stats_results_file: Path, cross_eval_results_file: Path, output_dir: Path, metric: str = "test/rmse"
):
    """
    Loads all experimental results, performs statistical analysis, and generates
    a comprehensive set of summary tables and plots.
    """
    # --- Setup ---
    if not stats_results_file.exists():
        raise FileNotFoundError(f"Statistical results file not found at: {stats_results_file}")
    if not cross_eval_results_file.exists():
        raise FileNotFoundError(f"Cross-evaluation results file not found at: {cross_eval_results_file}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"--- Analysis outputs will be saved to: {output_dir} ---")

    stats_df = pd.read_parquet(stats_results_file)
    cross_df = pd.read_parquet(cross_eval_results_file)
    sns.set_theme(style="whitegrid")

    # --- 1. Main Performance Analysis (from stats_results_file) ---
    print("\n--- Analyzing Main Statistical Results ---")

    # Summary Table
    summary_df = (
        stats_df.groupby("gnn_operator")
        .agg(
            mean_metric=(metric, "mean"),
            std_metric=(metric, "std"),
            trainable_parameters=("trainable_parameters", "first"),
        )
        .reset_index()
    )
    print("Model Performance Summary:")
    print(summary_df)
    summary_df.to_csv(output_dir / "main_performance_summary.csv", index=False, float_format="%.4f")

    # Statistical Tests (Friedman + Conover)
    model_groups = [group[metric].values for _, group in stats_df.groupby("gnn_operator")]
    if len(model_groups) > 1:
        friedman_stat, p_value = ss.friedmanchisquare(*model_groups)
        print(f"\nFriedman Test on '{metric}': statistic={friedman_stat:.4f}, p-value={p_value:.4g}")
        if p_value < 0.05:
            posthoc_results = sp.posthoc_conover_friedman(
                stats_df, melted=True, y_col=metric, block_col="run_id", group_col="gnn_operator"
            )

            # Critical Difference Diagram
            avg_rank = stats_df.groupby("run_id")[metric].rank().groupby(stats_df["gnn_operator"]).mean()
            fig, ax = plt.subplots(figsize=(10, max(4, len(avg_rank) * 0.5)))
            sp.critical_difference_diagram(ranks=avg_rank, sig_matrix=posthoc_results, ax=ax)
            ax.set_title(f"Critical Difference Diagram for {metric.replace('/', ' ').title()}")
            plt.savefig(output_dir / "critical_difference_diagram.svg", bbox_inches="tight")
            plt.close()
            print("Critical difference diagram saved.")

    # Performance vs. Complexity Plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.boxplot(ax=axes[0], data=stats_df, x="gnn_operator", y=metric, palette="viridis")
    axes[0].set_title("Model Performance Distribution", fontsize=14, weight="bold")
    axes[0].tick_params(axis="x", rotation=45)

    sns.scatterplot(
        ax=axes[1],
        data=summary_df,
        x="trainable_parameters",
        y="mean_metric",
        hue="gnn_operator",
        s=200,
        palette="viridis",
        legend=False,
    )
    for _, row in summary_df.iterrows():
        axes[1].text(row["trainable_parameters"] * 1.01, row["mean_metric"], row["gnn_operator"])
    axes[1].set_title("Performance vs. Model Complexity", fontsize=14, weight="bold")
    axes[1].set_xlabel("Trainable Parameters")
    axes[1].set_ylabel(f"Mean {metric.replace('/', ' ').title()}")
    plt.tight_layout()
    plt.savefig(output_dir / "performance_vs_complexity.svg")
    plt.close()
    print("Performance vs. complexity plots saved.")

    # --- 2. Cross-Evaluation Analysis (from cross_eval_results_file) ---
    print("\n--- Analyzing Cross-Evaluation Results ---")

    # Heatmap of performance across datasets
    pivot_table = cross_df.pivot_table(
        index="gnn_operator", columns="cross_eval_dataset", values=metric, aggfunc="mean"
    )
    plt.figure(figsize=(max(10, pivot_table.shape[1]), max(6, pivot_table.shape[0])))
    sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="viridis_r", linewidths=0.5)
    plt.title(f"Cross-Dataset Generalization ({metric.replace('/', ' ').title()})")
    plt.xlabel("Test Dataset")
    plt.ylabel("GNN Operator")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "cross_eval_heatmap.svg")
    plt.close()
    print("Cross-evaluation heatmap saved.")

    print("\n--- Analysis complete. ---")


def get_analysis_args() -> argparse.Namespace:
    """Parses command-line arguments for the analysis script."""
    parser = argparse.ArgumentParser(description="Analyze and plot GNN model performance and characteristics.")
    parser.add_argument(
        "--stats_results_file",
        type=Path,
        default=Path("results/statistical_results.parquet"),
    )
    parser.add_argument(
        "--cross_eval_results_file",
        type=Path,
        default=Path("results/cross_dataset_evaluation.parquet"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/analysis_plots"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_analysis_args()
    analyze_and_plot_results(
        stats_results_file=args.stats_results_file,
        cross_eval_results_file=args.cross_eval_results_file,
        output_dir=args.output_dir,
    )
