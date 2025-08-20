import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as ss
import seaborn as sns


def analyze_cross_evaluation_results(results_file: Path, output_dir: Path, metric: str = 'test_mae'):
    """
    Loads cross-evaluation results, analyzes model generalization across different
    datasets, and generates summary tables and a performance heatmap.

    Args:
        results_file (Path): Path to the input .parquet file containing the cross-evaluation results.
        output_dir (Path): Directory to save the analysis outputs.
        metric (str): The performance metric to use for the analysis.
    """
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found at: {results_file}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"--- Cross-evaluation analysis outputs will be saved to: {output_dir} ---")

    df = pd.read_parquet(results_file)
    print("--- Cross-evaluation results loaded successfully ---")

    # 1. Create a pivot table for the heatmap
    performance_pivot = df.pivot_table(
        index='gnn_operator',
        columns='cross_eval_dataset',
        values=metric,
        aggfunc='mean'
    )

    # 2. Generate and save the performance heatmap
    print(f"\n--- Generating performance heatmap for metric: '{metric}' ---")
    plt.figure(figsize=(max(12, performance_pivot.shape[1] * 0.8), max(8, performance_pivot.shape[0] * 0.6)))
    sns.heatmap(
        performance_pivot,
        annot=True,
        fmt=".4f",
        cmap="viridis_r",
        linewidths=.5,
        cbar_kws={'label': f'Mean {metric.replace("_", " ").title()}'}
    )
    plt.title(f'Model Performance ({metric.replace("_", " ").title()}) Across Datasets')
    plt.xlabel('Test Dataset')
    plt.ylabel('GNN Operator')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    # --- MODIFIED: Save in high-quality vector and raster formats ---
    heatmap_path_svg = output_dir / "cross_eval_performance_heatmap.svg"
    plt.savefig(heatmap_path_svg, format='svg', bbox_inches='tight')
    plt.close()
    print(f"Performance heatmap saved to: {heatmap_path_svg}")

    # 3. Generate and save summary statistics with robustness score
    print("\n--- Generating summary statistics ---")
    model_robustness = performance_pivot.std(axis=1).rename('robustness_std')

    summary_by_model = df.groupby('gnn_operator')[metric].agg(['mean', 'std']).reset_index()
    summary_by_model = summary_by_model.merge(model_robustness, on='gnn_operator')

    summary_by_model_path = output_dir / "summary_by_model.csv"
    summary_by_model.to_csv(summary_by_model_path, index=False, float_format='%.4f')
    print("\nAverage performance and robustness by model (across all datasets):")
    print(summary_by_model)

    summary_by_dataset = df.groupby('cross_eval_dataset')[metric].agg(['mean', 'std']).reset_index()
    summary_by_dataset_path = output_dir / "summary_by_dataset.csv"
    summary_by_dataset.to_csv(summary_by_dataset_path, index=False, float_format='%.4f')
    print("\nAverage difficulty by dataset (across all models):")
    print(summary_by_dataset)

    # 4. Statistical Test for Generalization
    print(f"\n--- Performing Friedman test on model ranks across datasets ---")
    df['rank'] = df.groupby(['run_id', 'cross_eval_dataset'])[metric].rank()
    rank_pivot = df.pivot_table(index=['run_id', 'cross_eval_dataset'], columns='gnn_operator', values='rank')

    model_rank_groups = [rank_pivot[col].dropna().values for col in rank_pivot.columns]

    if len(model_rank_groups) >= 2:
        friedman_stat, p_value = ss.friedmanchisquare(*model_rank_groups)
        print(f"Friedman Test on Ranks: statistic={friedman_stat:.4f}, p-value={p_value:.4g}")
        if p_value < 0.05:
            print("Result is significant (p < 0.05): There is a statistically significant difference in model generalization performance.")
        else:
            print("Result is not significant (p >= 0.05): There is no statistically significant difference in model generalization.")

    # 5. Performance Stability Visualization
    print(f"\n--- Generating performance stability plots (boxplots) ---")
    num_datasets = df['cross_eval_dataset'].nunique()
    g = sns.FacetGrid(df, col="cross_eval_dataset", col_wrap=min(4, num_datasets), sharey=True, height=4)
    g.map_dataframe(sns.boxplot, x='gnn_operator', y=metric)
    g.fig.suptitle('Model Performance Stability Across Datasets', y=1.03)
    g.set_axis_labels("GNN Operator", metric.replace("_", " ").title())
    g.set_xticklabels(rotation=45, ha='right')
    g.tight_layout()
    # --- MODIFIED: Save in high-quality vector and raster formats ---
    stability_plot_path_svg = output_dir / "cross_eval_stability_boxplots.svg"
    plt.savefig(stability_plot_path_svg, format='svg')
    plt.close()
    print(f"Stability boxplots saved to: {stability_plot_path_svg}")

    print("\n--- Cross-evaluation analysis complete. ---")


def get_analysis_args():
    """Parses command-line arguments for the analysis script."""
    parser = argparse.ArgumentParser(description="Analyze GNN model cross-dataset evaluation results.")
    parser.add_argument(
        "results_file",
        type=Path,
        help="Path to the .parquet file with aggregated cross-evaluation results.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/cross_eval_analysis"),
        help="Directory to save the generated tables and plots.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="test_mae",
        choices=["test_mae", "test_rmse", "test_mape", "test_medae", "test_r2"],
        help="The primary metric to use for the analysis.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_analysis_args()
    analyze_cross_evaluation_results(
        results_file=args.results_file,
        output_dir=args.output_dir,
        metric=args.metric,
    )
