from __future__ import annotations
import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path

def calculate_pearson_correlation(embedding_df, gene_expression_df):
    """
    Calculate Pearson correlation between each dimension of the embedding and gene expression data.
    """
    # Align samples
    common_samples = embedding_df.index.intersection(gene_expression_df.index)
    if len(common_samples) == 0:
        raise ValueError("No common samples between embedding and expression data.")
    
    emb = embedding_df.loc[common_samples]
    expr = gene_expression_df.loc[common_samples]
    
    # Standardize
    emb_std = (emb - emb.mean()) / emb.std()
    expr_std = (expr - expr.mean()) / expr.std()
    
    n = len(common_samples)
    correlation_lists = []
    
    for col in emb_std.columns:
        # Vectorized correlation calculation for one dimension vs all genes
        corrs = (expr_std.T @ emb_std[col]) / (n - 1)
        
        corr_df = pd.DataFrame({
            'Gene': expr.columns,
            'Correlation': corrs.values
        })
        # Sort by Correlation (descending)
        corr_df = corr_df.sort_values(by='Correlation', ascending=False)
        correlation_lists.append(corr_df)
        
    return correlation_lists

def save_correlation_lists(correlation_lists, result_dir):
    """
    Save correlation lists to RNK files (GSEA format).
    """
    output_dir = Path(result_dir) / "correlations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, corr_df in enumerate(correlation_lists):
        output_file = output_dir / f"dimension_{i}.rnk"
        corr_df.to_csv(output_file, sep='\t', index=False, header=False)
