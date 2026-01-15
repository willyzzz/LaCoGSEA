"""
LaCoGSEA core package.
"""

from .pipeline import train_autoencoder, run_gsea, summarize_gsea, run_full_pipeline, compute_activity
from .evaluation import calculate_pearson_correlation, save_correlation_lists
from .summarize import (
    build_nes_matrix_from_gsea_results_dir,
    build_nes_plus_minus_from_gsea_results_dir,
)
from .activity import calculate_sample_pathway_activity

__all__ = [
    "train_autoencoder",
    "run_gsea",
    "summarize_gsea",
    "run_full_pipeline",
    "compute_activity",
    "calculate_pearson_correlation",
    "save_correlation_lists",
    "build_nes_matrix_from_gsea_results_dir",
    "build_nes_plus_minus_from_gsea_results_dir",
    "calculate_sample_pathway_activity",
]
