"""
This module provides a comprehensive analysis suite for evaluating the results
of GNN model training and cross-validation.

It includes capabilities for:
- Calculating summary statistics (mean, std) for performance metrics.
- Performing statistical tests (Friedman test) to compare model performances.
- Generating visualizations such as critical difference diagrams, performance vs.
  complexity plots, and cross-evaluation heatmaps.

The analysis is driven by a configuration object and operates on data from
Parquet files generated during the MLOps pipeline.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scikit_posthocs as sp
import scipy.stats as ss
import seaborn as sns

from src.config_utils import load_config


class Analysis:
    """
    A class to perform analysis of model performance and generate visualizations.
    """

    def __init__(self, config):
        """
        Initializes the Analysis class with the given configuration.

        Args:
            config: A configuration object containing paths and analysis settings.
        """
        self.config = config
        self.output_dir = self.config.io.output_dir
        self.analysis_config = self.config.analysis

    def run(self):
        """
        Executes the full analysis pipeline.

        This method reads the statistical and cross-evaluation results, sets up
        the output directories, and then calls the appropriate methods to
        generate summaries and plots based on the configuration.

        Raises:
            FileNotFoundError: If the results files are not found.
        """
        stats_results_file = self.config.io.stats_results_file
        cross_eval_results_file = self.config.io.cross_eval_results_file

        if not stats_results_file.exists():
            raise FileNotFoundError(f"Statistical results file not found at: {stats_results_file}")
        if not cross_eval_results_file.exists():
            raise FileNotFoundError(f"Cross-evaluation results file not found at: {cross_eval_results_file}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"--- Analysis outputs will be saved to: {self.output_dir} ---")

        stats_df = pd.read_parquet(stats_results_file)
        cross_df = pd.read_parquet(cross_eval_results_file)
        sns.set_theme(style="whitegrid")

        # --- Main Analysis ---
        print("\n--- Running Main Analysis ---")
        for metric in self.analysis_config.main_metrics:
            print(f"\n--- Analyzing Metric: {metric} ---")
            metric_output_dir = self.output_dir / "main" / metric.replace('/', '_')
            metric_output_dir.mkdir(parents=True, exist_ok=True)

            summary_df = self._calculate_summary(stats_df, metric, metric_output_dir)
            if self.analysis_config.generate_critical_diagram:
                self._plot_critical_difference(stats_df, metric, metric_output_dir)
            if self.analysis_config.generate_performance_complexity_plot:
                self._plot_performance_complexity(stats_df, summary_df, metric, metric_output_dir)

        # --- Cross-Validation Analysis ---
        if self.analysis_config.generate_cross_eval_heatmap:
            print("\n--- Running Cross-Validation Analysis ---")
            for metric in self.analysis_config.cross_val_metrics:
                print(f"\n--- Analyzing Metric for Heatmap: {metric} ---")
                metric_output_dir = self.output_dir / "cross_val" / metric.replace('/', '_')
                metric_output_dir.mkdir(parents=True, exist_ok=True)
                self._plot_cross_eval_heatmap(cross_df, metric, metric_output_dir)

        print("\n--- Analysis complete. ---")

    def _calculate_summary(self, stats_df: pd.DataFrame, metric: str, output_dir: Path) -> pd.DataFrame:
        """
        Calculates and saves a summary of model performance metrics.

        Args:
            stats_df: DataFrame with statistical results from training runs.
            metric: The performance metric to summarize.
            output_dir: The directory to save the summary CSV file.

        Returns:
            A DataFrame containing the summary statistics.
        """
        summary_df = (
            stats_df.groupby("gnn_operator")
            .agg(
                mean_metric=(metric, "mean"),
                std_metric=(metric, "std"),
                trainable_parameters=("trainable_parameters", "first"),
            )
            .reset_index()
            .rename(columns={"mean_metric": f"mean_{metric}", "std_metric": f"std_{metric}"})
        )
        print("Model Performance Summary:")
        print(summary_df)
        summary_df.to_csv(output_dir / "main_performance_summary.csv", index=False, float_format="%.4f")
        return summary_df

    def _plot_critical_difference(self, stats_df: pd.DataFrame, metric: str, output_dir: Path):
        """
        Performs a Friedman test and plots a critical difference diagram.

        Args:
            stats_df: DataFrame with statistical results.
            metric: The metric to compare models on.
            output_dir: The directory to save the plot.
        """
        model_groups = [group[metric].values for _, group in stats_df.groupby("gnn_operator")]
        if len(model_groups) <= 1:
            print("Skipping Friedman test and critical difference diagram: only one model group.")
            return

        friedman_stat, p_value = ss.friedmanchisquare(*model_groups)
        print(f"\nFriedman Test on '{metric}': statistic={friedman_stat:.4f}, p-value={p_value:.4g}")

        if p_value < 0.05:
            posthoc_results = sp.posthoc_conover_friedman(
                stats_df, melted=True, y_col=metric, block_col="run_id", group_col="gnn_operator"
            )
            avg_rank = stats_df.groupby("run_id")[metric].rank().groupby(stats_df["gnn_operator"]).mean()

            fig, ax = plt.subplots(figsize=(10, max(4, len(avg_rank) * 0.5)))
            sp.critical_difference_diagram(ranks=avg_rank, sig_matrix=posthoc_results, ax=ax)
            ax.set_title(f"Critical Difference Diagram for {metric.replace('/', ' ').title()}")
            plt.savefig(output_dir / "critical_difference_diagram.svg", bbox_inches="tight")
            plt.close()
            print("Critical difference diagram saved.")
        else:
            print("No significant difference found; skipping post-hoc test and diagram.")

    def _plot_performance_complexity(self, stats_df: pd.DataFrame, summary_df: pd.DataFrame, metric: str, output_dir: Path):
        """
        Generates plots comparing model performance against complexity.

        Args:
            stats_df: DataFrame with detailed statistical results.
            summary_df: DataFrame with summarized performance metrics.
            metric: The performance metric to plot.
            output_dir: The directory to save the plot.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        sns.boxplot(ax=axes[0], data=stats_df, x="gnn_operator", y=metric, palette="viridis")
        axes[0].set_title("Model Performance Distribution", fontsize=14, weight="bold")
        axes[0].tick_params(axis="x", rotation=45)

        mean_metric_col = f"mean_{metric}"
        sns.scatterplot(
            ax=axes[1],
            data=summary_df,
            x="trainable_parameters",
            y=mean_metric_col,
            hue="gnn_operator",
            s=200,
            palette="viridis",
            legend=False,
        )
        for _, row in summary_df.iterrows():
            axes[1].text(row["trainable_parameters"] * 1.01, row[mean_metric_col], row["gnn_operator"])
        axes[1].set_title("Performance vs. Model Complexity", fontsize=14, weight="bold")
        axes[1].set_xlabel("Trainable Parameters")
        axes[1].set_ylabel(f"Mean {metric.replace('/', ' ').title()}")
        plt.tight_layout()
        plt.savefig(output_dir / "performance_vs_complexity.svg")
        plt.close()
        print("Performance vs. complexity plots saved.")

    def _plot_cross_eval_heatmap(self, cross_df: pd.DataFrame, metric: str, output_dir: Path):
        """
        Plots a heatmap of cross-dataset generalization performance.

        Args:
            cross_df: DataFrame with cross-evaluation results.
            metric: The performance metric to visualize.
            output_dir: The directory to save the heatmap.
        """
        if "cross_eval_dataset" not in cross_df.columns:
            print("Skipping cross-evaluation heatmap: 'cross_eval_dataset' column not found.")
            return

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


def main():
    """Main function to run the analysis as a standalone script."""
    config, _ = load_config()
    analysis = Analysis(config)
    analysis.run()


if __name__ == "__main__":
    main()
