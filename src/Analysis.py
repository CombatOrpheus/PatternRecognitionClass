import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scikit_posthocs as sp
import scipy.stats as ss
import seaborn as sns


class Analysis:
    """
    Loads all experimental results, performs statistical analysis, and generates
    a comprehensive set of summary tables and plots.
    """

    def __init__(self, stats_results_file: Path, cross_eval_results_file: Path, output_dir: Path):
        if not stats_results_file.exists():
            raise FileNotFoundError(f"Statistical results file not found at: {stats_results_file}")
        if not cross_eval_results_file.exists():
            raise FileNotFoundError(f"Cross-evaluation results file not found at: {cross_eval_results_file}")

        self.stats_df = pd.read_parquet(stats_results_file)
        self.cross_df = pd.read_parquet(cross_eval_results_file)
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"--- Analysis outputs will be saved to: {self.output_dir} ---")

    def run(self, metric: str):
        """
        Runs the full analysis and plotting pipeline.
        """
        sns.set_theme(style="whitegrid")
        self._analyze_main_performance(metric)
        self._analyze_cross_evaluation(metric)
        print("\n--- Analysis complete. ---")

    def _analyze_main_performance(self, metric: str):
        """
        Performs analysis on the main statistical results file.
        """
        print("\n--- Analyzing Main Statistical Results ---")

        # Summary Table
        summary_df = (
            self.stats_df.groupby("gnn_operator")
            .agg(
                mean_metric=(metric, "mean"),
                std_metric=(metric, "std"),
                trainable_parameters=("trainable_parameters", "first"),
            )
            .reset_index()
        )
        print("Model Performance Summary:")
        print(summary_df)
        summary_df.to_csv(self.output_dir / "main_performance_summary.csv", index=False, float_format="%.4f")

        # Statistical Tests
        self._perform_statistical_tests(metric)

        # Performance vs. Complexity Plots
        self._plot_performance_vs_complexity(metric, summary_df)

    def _perform_statistical_tests(self, metric: str):
        """
        Performs Friedman and Conover tests and generates a critical difference diagram.
        """
        model_groups = [group[metric].values for _, group in self.stats_df.groupby("gnn_operator")]
        if len(model_groups) <= 1:
            return

        friedman_stat, p_value = ss.friedmanchisquare(*model_groups)
        print(f"\nFriedman Test on '{metric}': statistic={friedman_stat:.4f}, p-value={p_value:.4g}")

        if p_value < 0.05:
            posthoc_results = sp.posthoc_conover_friedman(
                self.stats_df, melted=True, y_col=metric, block_col="run_id", group_col="gnn_operator"
            )

            avg_rank = self.stats_df.groupby("run_id")[metric].rank().groupby(self.stats_df["gnn_operator"]).mean()
            fig, ax = plt.subplots(figsize=(10, max(4, len(avg_rank) * 0.5)))
            sp.critical_difference_diagram(ranks=avg_rank, sig_matrix=posthoc_results, ax=ax)
            ax.set_title(f"Critical Difference Diagram for {metric.replace('/', ' ').title()}")
            plt.savefig(self.output_dir / "critical_difference_diagram.svg", bbox_inches="tight")
            plt.close()
            print("Critical difference diagram saved.")

    def _plot_performance_vs_complexity(self, metric: str, summary_df: pd.DataFrame):
        """
        Generates and saves plots for performance distribution and performance vs. complexity.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        sns.boxplot(ax=axes[0], data=self.stats_df, x="gnn_operator", y=metric, palette="viridis")
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
        plt.savefig(self.output_dir / "performance_vs_complexity.svg")
        plt.close()
        print("Performance vs. complexity plots saved.")

    def _analyze_cross_evaluation(self, metric: str):
        """
        Performs analysis on the cross-evaluation results file.
        """
        print("\n--- Analyzing Cross-Evaluation Results ---")
        pivot_table = self.cross_df.pivot_table(
            index="gnn_operator", columns="cross_eval_dataset", values=metric, aggfunc="mean"
        )
        plt.figure(figsize=(max(10, pivot_table.shape[1]), max(6, pivot_table.shape[0])))
        sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="viridis_r", linewidths=0.5)
        plt.title(f"Cross-Dataset Generalization ({metric.replace('/', ' ').title()})")
        plt.xlabel("Test Dataset")
        plt.ylabel("GNN Operator")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(self.output_dir / "cross_eval_heatmap.svg")
        plt.close()
        print("Cross-evaluation heatmap saved.")