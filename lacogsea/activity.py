from __future__ import annotations
from pathlib import Path
from typing import Union
import pandas as pd
import numpy as np
import logging

LOGGER = logging.getLogger(__name__)

def calculate_sample_pathway_activity(
    embedding_df: pd.DataFrame,
    nes_df: pd.DataFrame,
    output_path: Union[str, Path, None] = None
) -> pd.DataFrame:
    """
    Calculate sample-level pathway activity by multiplying embeddings with the NES matrix.
    
    Activity = Embedding (Samples x Dims) * NES (Dims x Pathways)
    
    Args:
        embedding_df: DataFrame of latent embeddings (Samples x Dims).
        nes_df: DataFrame of GSEA NES scores (Dims x Pathways).
        output_path: Optional path to save the resulting activity matrix.
        
    Returns:
        DataFrame of pathway activity scores (Samples x Pathways).
    """
    # Align dimensions
    # embedding_df columns should be dim_0, dim_1...
    # nes_df index should be dim_0, dim_1...
    
    common_dims = embedding_df.columns.intersection(nes_df.index)
    if len(common_dims) == 0:
        # Try to clean names (e.g., 'dimension_0' vs 'dim_0')
        emb_cols = [c.replace('dimension_', 'dim_') for c in embedding_df.columns]
        nes_idx = [i.replace('dimension_', 'dim_') for i in nes_df.index]
        
        embedding_df.columns = emb_cols
        nes_df.index = nes_idx
        common_dims = embedding_df.columns.intersection(nes_df.index)
        
    if len(common_dims) == 0:
        raise ValueError(f"No common dimensions found between embedding ({list(embedding_df.columns[:3])}...) "
                         f"and NES matrix ({list(nes_df.index[:3])}...).")
    
    if len(common_dims) < len(embedding_df.columns) or len(common_dims) < len(nes_df.index):
        LOGGER.warning(f"Aligning on {len(common_dims)} common dimensions.")
        
    E = embedding_df[common_dims].values
    N = nes_df.loc[common_dims].values
    
    # Activity = E * N
    activity_values = E @ N
    
    activity_df = pd.DataFrame(
        activity_values,
        index=embedding_df.index,
        columns=nes_df.columns
    )
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        activity_df.to_csv(output_path, sep="\t")
        LOGGER.info(f"Saved pathway activity matrix to {output_path}")
        
    return activity_df
