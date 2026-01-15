from __future__ import annotations
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union
import logging

LOGGER = logging.getLogger(__name__)

def plot_top_pathways_heatmap(top_df: pd.DataFrame, output_path: Union[str, Path]):
    """
    Creates a heatmap for the top pathways.
    - Repetitive pathways (appearing in multiple dims) are at the top.
    """
    if top_df.empty:
        LOGGER.warning("No significant pathways found (FDR < 0.05). Skipping heatmap.")
        return

    # 1. Row Selection/Ordering:
    # Pathways that appear in most dimensions go to the top
    presence_count = top_df.notna().sum(axis=1)
    
    # Identify the first dimension each pathway appears in
    first_dim = top_df.notna().idxmax(axis=1).str.replace('dim_', '').astype(int)
    
    # Within those with same presence count, sort by first appearance and then absolute NES
    max_abs_nes = top_df.abs().max(axis=1)
    
    order_df = pd.DataFrame({
        'count': presence_count,
        'first_dim': first_dim,
        'max_abs': max_abs_nes
    }).sort_values(['count', 'first_dim', 'max_abs'], ascending=[False, True, False])
    
    plot_df = top_df.loc[order_df.index].fillna(0)
    
    # 2. Figure size scaling
    fig_height = max(6, len(plot_df) * 0.3)
    fig_width = max(8, len(plot_df.columns) * 0.8)
    
    plt.figure(figsize=(fig_width, fig_height))
    
    # 3. Plot
    sns.heatmap(
        plot_df,
        cmap="RdBu_r",
        center=0,
        cbar_kws={'label': 'NES'},
        annot=False,
        linewidths=.5
    )
    
    plt.title("Top Significant Pathways per Latent Dimension (FDR < 0.05)", pad=20)
    plt.xlabel("Latent Dimensions")
    plt.ylabel("Pathways")
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    LOGGER.info(f"Heatmap saved to: {output_path}")
