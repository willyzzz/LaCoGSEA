#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unified Visualization for Figure 2 (per-dataset).
Generates (by default):
1) Target Pathway Ranking Heatmap  (mask rank>=100 as Not Detected)
3) Specificity Bar (Num_Hits_Strict)
4) Redundancy Bar (Redundancy_Jaccard)

IMPORTANT: Method names are NOT renamed (kept exactly as in input CSV).
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Try to use scienceplots style if available, but fallback gracefully
try:
    import scienceplots
    plt.style.use(['science', 'nature', 'no-latex'])
    # Override some scienceplots defaults that might be too small
    plt.rcParams.update({
        "font.size": 18,
        "axes.titlesize": 22,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 16,
        "figure.figsize": (10, 8)
    })
except ImportError:
    # Use seaborn as fallback but with larger context
    sns.set_theme(style="white", context="talk")
    plt.rcParams.update({
        "axes.titlesize": 22,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 16,
    })

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# Keep your original 8 methods and order
METHODS_ORDER = [
    "Standard_DE",
    "PCA_Corr",
    "PCA_Weights",
    "AE_DeepLIFT_Zero",
    "AE_DeepLIFT_Mean",
    "AE_SHAP_Zero",
    "AE_SHAP_Mean",
    "AE_Correlation",
]

METHOD_DISPLAY_NAMES = {
    "AE_Correlation": "LaCoGSEA",
    "AE_Corr": "LaCoGSEA"
}

def get_method_display(name: str) -> str:
    # Force names to use underscores for consistency and fix potential dot issues
    # Use '$\\_$' to ensure underscores render correctly in mathtext
    display = METHOD_DISPLAY_NAMES.get(name, name)
    return str(display).replace(".", "_").replace("_", r"$\_$")

NOT_DETECTED_RANK = 100  # your convention

# Mapping for dataset descriptions
DATASET_DESCRIPTIONS = {
    "GSE10846": "GSE10846\n(DLBCL)",
    "GSE48350": "GSE48350\n(Alzheimer)",
    "GSE11375": "GSE11375\n(Trauma)",
    "GSE126848": "GSE126848\n(Liver)",
    "GSE116250": "GSE116250\n(Heart)",
    # Fallback for others if any
    "SCAN-B": "SCAN-B",
    "METABRIC": "METABRIC",
    "TCGA Lung": "TCGA Lung"
}

def get_dataset_desc(name: str) -> str:
    """Return descriptive name if available, else original."""
    return DATASET_DESCRIPTIONS.get(name, name)


def clean_pathway_name(name: str) -> str:
    # keep readable; does not change method names
    return str(name).replace("KEGG_", "").replace("_", " ").title()


def _ordered_methods(existing_methods) -> list:
    existing_methods = set(existing_methods)
    return [m for m in METHODS_ORDER if m in existing_methods]


def plot_rank_heatmap(df: pd.DataFrame, out_dir: Path, dataset: str, primary_sort_method: str = "AE_Correlation"):
    logging.info("[%s] Generating ranking heatmap...", dataset)
    rank_df = df[df["Metric_Type"] == "Target_Rank"].copy()
    if rank_df.empty:
        logging.warning("[%s] No Target_Rank data found; skip heatmap.", dataset)
        return

    # Pivot -> pathways x methods
    pivot_df = rank_df.pivot(index="Pathway", columns="Method", values="Value")

    # Ensure consistent method columns and numeric values
    pivot_df = pivot_df.reindex(columns=_ordered_methods(pivot_df.columns))
    pivot_df = pivot_df.apply(pd.to_numeric, errors="coerce")

    # Fill missing with NOT_DETECTED_RANK so they will be masked
    pivot_df = pivot_df.fillna(NOT_DETECTED_RANK)

    # Optional: order pathways by a key method (default AE_Correlation) ascending
    if primary_sort_method in pivot_df.columns:
        pivot_df = pivot_df.sort_values(by=primary_sort_method, ascending=True)
    else:
        # fallback: order by mean rank
        pivot_df = pivot_df.loc[pivot_df.mean(axis=1).sort_values().index]

    # Clean pathway display names
    pivot_df.index = [clean_pathway_name(p) for p in pivot_df.index]
    
    # Rename columns for display
    pivot_df.columns = [get_method_display(c) for c in pivot_df.columns]

    # Prepare display values (mask for colors, manual text for ND)
    mask = pivot_df >= NOT_DETECTED_RANK
    
    # Determine vmax for the color scale (max rank to show color for, cap at 50)
    detected_values = pivot_df.values[pivot_df.values < NOT_DETECTED_RANK]
    vmax_val = min(50, int(np.nanmax(detected_values))) if detected_values.size > 0 else 50

    # Compact Figure
    plt.figure(figsize=(12, 10), layout='constrained')
    ax = plt.gca()
    ax.set_facecolor("white")  # Background color for masked (ND) cells

    sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".0f",
        cmap="Blues_r",
        vmin=1,
        vmax=vmax_val,
        mask=mask,
        cbar_kws={"label": "Rank (Lower is Better)"},
        linewidths=0.8,
        linecolor="#D3D3D3", 
        annot_kws={"fontsize": 16, "fontweight": "bold"}, # Increased fontsize
    )
    
    # Manually draw grid lines to ensure even masked (ND) cells have visible borders
    for i in range(len(pivot_df.index) + 1):
        ax.axhline(i, color='#D3D3D3', lw=0.8)
    for j in range(len(pivot_df.columns) + 1):
        ax.axvline(j, color='#D3D3D3', lw=0.8)

    dataset_display = get_dataset_desc(dataset).replace("\n", " ") # Single line for title
    ax.set_title(f"{dataset_display} | Target Pathway Ranking", fontsize=24, fontweight='bold', pad=20)
    ax.set_xlabel("", fontsize=20, fontweight='bold')
    ax.set_ylabel("", fontsize=20, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=18, fontweight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=18, fontweight='bold')
    
    # Increase colorbar font size
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label("Rank (Lower is Better)", fontsize=18, fontweight='bold')

    # plt.tight_layout() # constrained_layout is used
    out_path = out_dir / f"{dataset}_ranking_heatmap.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    logging.info("[%s] Saved: %s", dataset, out_path)

def plot_average_rank_summary(df, output_dir):
    """
    Journal-style Dot Plot for Method comparison across datasets.
    Y-axis: Method (sorted by global mean rank).
    X-axis: Mean Penalized Rank.
    Dots: Individual datasets.
    Mean Marker: Thick vertical line segment.
    """
    logging.info("Generating journal-style Dot Plot for average rank summary...")

    rank_df = df[df['Metric_Type'] == 'Target_Rank'].copy()
    if rank_df.empty:
        logging.warning("No Target_Rank data found; skip summary plot.")
        return

    rank_df['Value'] = pd.to_numeric(rank_df['Value'], errors='coerce')
    penalty_value = 100
    
    # 1. Pre-process and calculate metrics per [Method, Dataset]
    rank_df['Value'] = rank_df['Value'].fillna(penalty_value)
    rank_df.loc[rank_df['Value'] >= penalty_value, 'Value'] = penalty_value
    rank_df['Is_Detected'] = rank_df['Value'] < penalty_value

    summary = (
        rank_df.groupby(['Method', 'Dataset']).agg(
            Mean_Penalized=('Value', 'mean'),
            Coverage=('Is_Detected', 'mean')
        ).reset_index()
    )

    # Map dataset names to descriptions
    summary['Dataset_Display'] = summary['Dataset'].apply(get_dataset_desc)

    # 2. Calculate Global Stats per Method for sorting and marking
    global_stats = (
        summary.groupby('Method').agg(
            Global_Mean=('Mean_Penalized', 'mean'),
            Global_Coverage=('Coverage', 'mean')
        ).reset_index()
    ).sort_values('Global_Mean', ascending=True)

    # Reorder methods based on Global_Mean
    sorted_methods = global_stats['Method'].tolist()
    sorted_display_methods = [get_method_display(m) for m in sorted_methods]
    summary['Method_Display'] = summary['Method'].apply(get_method_display)

    # 3. Plotting
    plt.figure(figsize=(14, 10), layout='constrained')
    sns.set_style("white")
    ax = plt.gca()

    # Stripplot for datasets (Dots)
    sns.stripplot(
        data=summary,
        y='Method_Display',
        x='Mean_Penalized',
        order=sorted_display_methods,
        hue='Dataset_Display', # Use descriptive names
        jitter=0.2,
        size=15, # Larger dots
        alpha=0.8,
        palette='tab10',
        ax=ax
    )


    # Add Mean Lines (Vertical segments at each row)
    for i, method in enumerate(sorted_methods):
        mean_val = global_stats.loc[global_stats['Method'] == method, 'Global_Mean'].values[0]
        avg_cov = global_stats.loc[global_stats['Method'] == method, 'Global_Coverage'].values[0]
        is_ae_corr = (method == "AE_Correlation" or method == "LaCoGSEA")
        ax.vlines(
            x=mean_val,
            ymin=i - 0.35, ymax=i + 0.35,
            color='#111111',
            linewidth=5 if is_ae_corr else 3,
            linestyles='--', # Using dashed line for mean as in Fig 4/5
            alpha=0.8,
            zorder=4
        )
        
        # Add text annotation on the right: "Mean (Cov%)"
        ax.text(
            105, # Positioned slightly beyond the 100 limit
            i,
            f"{mean_val:.1f} ({avg_cov*100:.0f}%)",
            va='center',
            ha='left',
            fontsize=16, # Increased font size
            fontweight='bold',
            color='black'
        )

    # Final touches
    ax.set_title('Method Performance Across All Datasets', fontsize=24, fontweight='bold', pad=20)
    ax.set_xlabel('Mean Target Pathway Rank (Penalty=100)', fontsize=20, fontweight='bold')
    ax.set_ylabel('')
    ax.set_xlim(0, 130) # Room for annotations
    ax.set_xticks([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    
    # Increase font sizes for better readability
    plt.xticks(fontsize=18, fontweight='bold')
    plt.yticks(fontsize=18, fontweight='bold') # Method names on Y-axis
    
    # Grid and borders
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    sns.despine(left=True)
    
    # Legend - larger font sizes
    plt.legend(title='Dataset', bbox_to_anchor=(1.0, 0.5), loc='center left', 
               frameon=False, fontsize=16, title_fontsize=20)

    # plt.tight_layout() # constrained_layout is used
    out_path = output_dir / 'all_datasets_dotplot_summary.png'
    plt.savefig(out_path, dpi=300)
    plt.close()

    logging.info(f"Journal-style dot plot saved to: {out_path}")


def plot_specificity_bar(df: pd.DataFrame, out_dir: Path, dataset: str):
    logging.info("[%s] Generating specificity bar...", dataset)
    summary_df = df[df["Metric_Type"].isna()].copy()
    if summary_df.empty or "Num_Hits_Strict" not in summary_df.columns:
        logging.warning("[%s] No summary Num_Hits_Strict found; skip specificity.", dataset)
        return

    # Ensure order
    summary_df["Method"] = summary_df["Method"].astype(str)
    order = _ordered_methods(summary_df["Method"].unique())

    plt.figure(figsize=(12, 6), layout='constrained')
    ax = sns.barplot(
        data=summary_df,
        x="Method",
        y="Num_Hits_Strict",
        order=order,
    )
    ax.set_title(f"{dataset} | Number of Significant Pathways (Strict)", fontsize=22, fontweight='bold')
    ax.set_ylabel("Significant Pathway Count", fontsize=20, fontweight='bold')
    ax.set_xlabel("")
    ax.set_xticklabels([get_method_display(t.get_text()) for t in ax.get_xticklabels()], rotation=45, ha="right", fontsize=16)
    ax.tick_params(axis="y", labelsize=16)

    # plt.tight_layout()
    out_path = out_dir / f"{dataset}_specificity_counts.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    logging.info("[%s] Saved: %s", dataset, out_path)


def plot_redundancy_bar(df: pd.DataFrame, out_dir: Path, dataset: str):
    logging.info("[%s] Generating redundancy bar...", dataset)
    summary_df = df[df["Metric_Type"].isna()].copy()
    if summary_df.empty or "Redundancy_Jaccard" not in summary_df.columns:
        logging.warning("[%s] No summary Redundancy_Jaccard found; skip redundancy.", dataset)
        return

    summary_df["Method"] = summary_df["Method"].astype(str)
    order = _ordered_methods(summary_df["Method"].unique())

    plt.figure(figsize=(12, 6), layout='constrained')
    ax = sns.barplot(
        data=summary_df,
        x="Method",
        y="Redundancy_Jaccard",
        order=order,
    )
    ax.set_title(f"{dataset} | Redundancy (Avg Pairwise Jaccard)", fontsize=22, fontweight='bold')
    ax.set_ylabel("Avg Pairwise Jaccard Index", fontsize=20, fontweight='bold')
    ax.set_xlabel("")
    ax.set_xticklabels([get_method_display(t.get_text()) for t in ax.get_xticklabels()], rotation=45, ha="right", fontsize=16)
    ax.tick_params(axis="y", labelsize=16)

    # plt.tight_layout()
    out_path = out_dir / f"{dataset}_redundancy_score.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    logging.info("[%s] Saved: %s", dataset, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="GSE10846", help="Dataset name, e.g., GSE10846")
    parser.add_argument(
        "--root",
        default=None,
        help="Root dir that contains deep_metrics_*.csv (default: PROJECT_ROOT/results/figure2/figure2_deep_analytics)",
    )
    parser.add_argument("--no_heatmap", action="store_true", help="Skip ranking heatmap")
    parser.add_argument("--no_specificity", action="store_true", help="Skip specificity bar")
    parser.add_argument("--no_redundancy", action="store_true", help="Skip redundancy bar")
    parser.add_argument("--primary_sort_method", default="AE_Correlation", help="Pathway row sorting key in heatmap")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2].absolute()
    root = Path(args.root) if args.root else (project_root / "results" / "figure2" / "figure2_deep_analytics")
    root.mkdir(parents=True, exist_ok=True)
 
    # Load multi-dataset file if possible
    all_datasets_file = root / "deep_metrics_all_datasets.csv"
    if all_datasets_file.exists():
        df_all = pd.read_csv(all_datasets_file)
    else:
        # Prefer per-dataset file if global not found
        summary_file = root / f"deep_metrics_{args.dataset}.csv"
        if not summary_file.exists():
            summary_file = root / "deep_metrics_summary.csv"
        if not summary_file.exists():
            raise FileNotFoundError(f"Summary file not found: {summary_file}")
        df_all = pd.read_csv(summary_file)
 
    # Filter for single dataset plots
    df_single = df_all.copy()
    if "Dataset" in df_single.columns:
        df_single = df_single[df_single["Dataset"] == args.dataset]

    if df_single.empty: # Changed from df.empty to df_single.empty
        logging.warning("[%s] No rows found after filtering; nothing to plot.", args.dataset)
        return

    # Plot single dataset metrics
    if not df_single.empty:
        if not args.no_heatmap:
            plot_rank_heatmap(df_single, root, args.dataset, primary_sort_method=args.primary_sort_method)
        # if not args.no_specificity:
        #     plot_specificity_bar(df_single, root, args.dataset)
        # if not args.no_redundancy:
        #     plot_redundancy_bar(df_single, root, args.dataset)
    else:
        logging.warning("[%s] No rows found for single dataset; skip per-dataset plots.", args.dataset)
 
    # Multi-dataset summary (always run if df_all is valid)
    plot_average_rank_summary(df_all, root)


    logging.info("[%s] Done. Outputs in: %s", args.dataset, root.resolve())


if __name__ == "__main__":
    main()
