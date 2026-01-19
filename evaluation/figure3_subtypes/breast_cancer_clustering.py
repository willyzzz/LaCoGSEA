#!/usr/bin/env python
from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
from core.model_structure import Encoder

import argparse
import logging
import re
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from core.gene_mapping import convert_ensembl_to_symbol, get_ensembl_to_symbol_mapping, is_ensembl_id
from core.io_utils import (
    PAM50_CANONICAL,
    RunDirSelection,
    find_latest_ae_run_dir,
    find_pca_run_dir,
    load_embedding_csv,
    load_labels_with_pam50,
)
from core.pathway_activity_r_gsva import (
    find_gmt_fallback,
    load_or_compute_pathway_activity,
    parse_gmt,
    union_genes,
)
from core.nes_from_gsea_reports import (
    build_nes_matrix_from_gsea_results_dir,
    build_nes_matrix_from_scanb_dim_dirs,
)


METHOD_DISPLAY_NAMES = {
    "AE": "LaCoGSEA",
}

def get_method_display(name: str) -> str:
    """Helper to get plotting names (e.g., AE -> LaCoGSEA)."""
    display = METHOD_DISPLAY_NAMES.get(name, name)
    return display.replace(".", "_")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def _dataset_defaults(dataset: str) -> Dict[str, str]:
    d = dataset.lower()
    # Path(__file__).resolve().parents[2] is GSEA-PLUS
    # Path(__file__).resolve().parents[3] is barlow_twins (where data resides)
    data_root = Path(__file__).resolve().parents[3]
    
    if d == "scanb":
        return {
            "bulk_train_path": str(data_root / "scanb_data/scanb_2022/scanb_trainingset_original.txt"),
            "bulk_test_path": str(data_root / "scanb_data/scanb_2022/scanb_testset_original.txt"),
            "bulk_normal_train_path": str(data_root / "scanb_data/scanb_2022/scanb_normal_bulk_train_original.txt"),
            "bulk_normal_test_path": str(data_root / "scanb_data/scanb_2022/scanb_normal_bulk_test_original.txt"),
            "label_path": str(data_root / "scanb_data/scanb_2022/test_labels.csv"),
            "train_label_path": str(data_root / "scanb_data/scanb_2022/train_labels.csv"),
        }
    if d == "metabric":
        return {
            "bulk_train_path": str(data_root / "pyega3/EGA_metabric_discovery_fullset.csv"),
            "bulk_test_path": str(data_root / "pyega3/EGA_metabric_validation_fullset.csv"),
            "bulk_normal_train_path": str(data_root / "pyega3/EGA_metabric_normal_discovery_fullset.csv"),
            "bulk_normal_test_path": str(data_root / "pyega3/EGA_metabric_normal_validation_fullset.csv"),
            "label_path": str(data_root / "original_paper_metabric_data/clinical_data.csv"),
        }
    raise ValueError(f"Unknown dataset={dataset}. Expected scanb or Metabric.")


def load_expression_matrix(path: str | Path, dataset: str, *, log_prefix: str, include_normal: bool = True, test_ratio: float = 0.2, seed: int = 42) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Expression file not found: {path}")

    logging.info(f"Loading {log_prefix} expression: {path}")
    df = pd.read_csv(path, sep=",", index_col=0, low_memory=False)
    
    if include_normal and dataset.lower() in ["scanb", "metabric"]:
        ds_key = dataset.lower()
        defaults = _dataset_defaults(ds_key)
        norm_train = Path(defaults["bulk_normal_train_path"])
        norm_test = Path(defaults["bulk_normal_test_path"])

        norm_dfs = []
        if norm_train.exists():
            norm_dfs.append(pd.read_csv(norm_train, index_col=0))
        if norm_test.exists():
            norm_dfs.append(pd.read_csv(norm_test, index_col=0))

        if norm_dfs:
            logging.info(f"Including Normal bulk samples for {dataset.upper()} (splitting all normal by {test_ratio})...")
            df_norm_all = pd.concat(norm_dfs)
            df_norm_all = df_norm_all.loc[~df_norm_all.index.duplicated()]
            
            # Remove any normal samples already present in the primary file to ensure valid split
            original_normal_count = df.index.isin(df_norm_all.index).sum()
            if original_normal_count > 0:
                logging.info(f"Removing {original_normal_count} existing normal samples from primary file to re-apply split.")
                df = df.loc[~df.index.isin(df_norm_all.index)]

            from sklearn.model_selection import train_test_split
            norm_train_ids, norm_test_ids = train_test_split(df_norm_all.index, test_size=test_ratio, random_state=seed)
            
            if log_prefix.lower() == "test":
                df = pd.concat([df, df_norm_all.loc[norm_test_ids]])
            else:
                df = pd.concat([df, df_norm_all.loc[norm_train_ids]])
            logging.info(f"Added {len(norm_test_ids) if log_prefix.lower() == 'test' else len(norm_train_ids)} normal samples to {log_prefix} set.")

    if dataset.lower() == "scanb" and "GEX.assay" in df.columns:
        df = df.drop(columns=["GEX.assay"])

    # Fill missing per-sample (row) mean, vectorized.
    row_means = df.mean(axis=1)
    df = df.T.fillna(row_means).T
    df = df.loc[:, ~df.columns.duplicated()]
    df.index = df.index.astype(str)
    df.sort_index(inplace=True)

    # Ensembl -> symbol if needed (inspect first 10 columns)
    sample_cols = list(df.columns[:10])
    if any(is_ensembl_id(str(c)) for c in sample_cols):
        logging.info("Detected Ensembl IDs; converting to gene symbols...")
        df = convert_ensembl_to_symbol(df)
        df = df.loc[:, ~df.columns.duplicated()]

    # log2(x+1) as in dataset_input.py
    df_vals = np.log2(df.values.astype(np.float32) + 1.0)
    df = pd.DataFrame(df_vals, index=df.index, columns=df.columns)
    return df


_ENSEMBL_TO_SYMBOL_CACHE: Optional[Dict[str, str]] = None


def _get_ensembl_to_symbol() -> Dict[str, str]:
    global _ENSEMBL_TO_SYMBOL_CACHE
    if _ENSEMBL_TO_SYMBOL_CACHE is None:
        _ENSEMBL_TO_SYMBOL_CACHE = get_ensembl_to_symbol_mapping(force_download=False)
    return _ENSEMBL_TO_SYMBOL_CACHE


def load_gene_list_from_expression_header(path: str | Path, dataset: str) -> set[str]:
    """
    Load only gene column names from an expression CSV header (no full data read),
    and map Ensembl IDs to gene symbols if needed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Expression file not found: {path}")

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline().strip("\n\r")

    parts = header.split(",")
    if not parts:
        return set()

    # Drop index column header (empty string for many CSVs)
    if parts[0] == "":
        parts = parts[1:]

    # SCAN-B may have an extra first column 'GEX.assay'
    if dataset.lower() == "scanb" and parts and parts[0] == "GEX.assay":
        parts = parts[1:]

    genes = [p.strip() for p in parts if p.strip()]
    if not genes:
        return set()

    # Map Ensembl -> symbol using cached mapping (fast, no full data load).
    if any(is_ensembl_id(g) for g in genes[:50]):  # check first chunk
        mapping = _get_ensembl_to_symbol()
        mapped = []
        for g in genes:
            g_clean = str(g).split(".")[0]
            mapped.append(mapping.get(g_clean, g_clean))
        genes = mapped

    # De-dup
    return set(genes)


def compute_pca_test_embedding(
    *,
    train_expr: pd.DataFrame,
    test_expr: pd.DataFrame,
    n_components: int,
    seed: int,
) -> pd.DataFrame:
    from sklearn.decomposition import PCA

    # Align genes
    common_genes = list(set(train_expr.columns).intersection(test_expr.columns))
    if len(common_genes) < 100:
        raise ValueError(f"Too few common genes between train/test for PCA: {len(common_genes)}")

    train_df = train_expr[common_genes]
    test_df = test_expr[common_genes]

    scaler = StandardScaler(with_mean=True, with_std=True)
    train_scaled = scaler.fit_transform(train_df.values)
    test_scaled = scaler.transform(test_df.values)

    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(train_scaled)
    test_pca = pca.transform(test_scaled)

    return pd.DataFrame(
        test_pca,
        index=test_expr.index,
        columns=[f"dim_{i}" for i in range(n_components)],
    )


def _align_by_index(
    embedding_ae: pd.DataFrame,
    embedding_pca: pd.DataFrame,
    expr: pd.DataFrame,
    pam50: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    common = embedding_ae.index.intersection(embedding_pca.index)
    common = common.intersection(expr.index)
    common = common.intersection(pam50.index)
    common = common.sort_values()

    if len(common) < 10:
        raise ValueError(f"Too few common samples after alignment: {len(common)}")

    return (
        embedding_ae.loc[common],
        embedding_pca.loc[common],
        expr.loc[common],
        pam50.loc[common],
    )


def _standardize(X: np.ndarray) -> np.ndarray:
    return StandardScaler(with_mean=True, with_std=True).fit_transform(X)


def _kmeans_ari(X: np.ndarray, y_true: np.ndarray, k: int, seed: int) -> float:
    km = KMeans(n_clusters=k, n_init=50, random_state=seed)
    y_pred = km.fit_predict(X)
    return float(adjusted_rand_score(y_true, y_pred))


def _tsne_2d(X: np.ndarray, seed: int) -> np.ndarray:
    n = X.shape[0]
    # Consistent parameters across all methods for fair comparison
    perp = min(30, max(2, (n - 1) // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        init="pca",
        learning_rate="auto",
        n_iter=1000,
        random_state=seed,
    )
    return tsne.fit_transform(X)


def _plot_tsne(
    out_png: Path,
    out_pdf: Path,
    coords: np.ndarray,
    labels: pd.Series,
    title: str,
    ax=None,
    show_legend: bool = False, # Changed default to False
) -> None:
    import matplotlib.pyplot as plt

    palette = {
        "Basal": "#D62728",
        "LumA": "#1F77B4",
        "LumB": "#FF7F0E",
        "HER2": "#2CA02C",
        "Normal": "#999999",
    }

    created_fig = False
    if ax is None:
        # Standard square-ish ratio since legend is gone
        fig, ax = plt.subplots(figsize=(7, 6), dpi=200)
        created_fig = True
    
    # Plot all subtypes
    for subtype in PAM50_CANONICAL:
        mask = labels.values == subtype
        if mask.sum() == 0:
            continue
        
        # Large points as requested
        s_size = 40 if subtype == "Basal" else 35
        alpha_val = 0.85 if subtype == "Basal" else 0.75
        
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=s_size,
            c=palette.get(subtype, "#333333"),
            label=f"{subtype} (n={int(mask.sum())})",
            alpha=alpha_val,
            linewidths=0.3, 
            edgecolors='white',
        )

    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("t-SNE 1", fontsize=12, fontweight='bold')
    ax.set_ylabel("t-SNE 2", fontsize=12, fontweight='bold')
    
    if show_legend:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    
    ax.grid(False)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    if created_fig:
        fig.tight_layout()
        fig.savefig(out_png, dpi=300, bbox_inches='tight')
        fig.savefig(out_pdf, bbox_inches='tight')
        plt.close(fig)

def _plot_shared_legend(out_png: Path, out_pdf: Path, subtype_counts: Dict[str, int]) -> None:
    import matplotlib.pyplot as plt

    # More vibrant color versions for publication impact
    palette = {
        "Basal": "#D62728",
        "LumA": "#1F77B4",
        "LumB": "#FF7F0E",
        "HER2": "#2CA02C",
        "Normal": "#999999",
    }

    fig, ax = plt.subplots(figsize=(4, 6), facecolor='white')

    for subtype in PAM50_CANONICAL:
        count = subtype_counts.get(subtype, 0)
        label = f"{subtype} (n={count})"
        ax.scatter(
            [], [],
            c=palette[subtype],
            label=label,
            s=160,
            alpha=1.0,
            edgecolors="white",   # White border makes it look brighter
            linewidths=0.8
        )

    legend = ax.legend(
        loc="center",
        frameon=False,
        fontsize=14,
        title="PAM50 Subtypes",
        title_fontsize=16,
        markerscale=1.0,
        labelspacing=1.2 # More space between items
    )

    # Force legend handles to be fully opaque
    for h in legend.legendHandles:
        h.set_alpha(1.0)

    ax.axis("off")

    # Important: Save with transparent=False to keep the white background
    fig.savefig(out_png, dpi=300, bbox_inches="tight", transparent=False, facecolor='white')
    fig.savefig(out_pdf, bbox_inches="tight", transparent=False, facecolor='white')
    plt.close(fig)



def _plot_ari_bar(
    out_png: Path,
    out_pdf: Path,
    ari_scores: Dict[str, list[float]],
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    methods_internal = ["AE", "PCA", "GSVA", "ssGSEA"]
    display_names = [get_method_display(m) for m in methods_internal]
    
    # Calculate means and stds
    means = [np.mean(ari_scores[m]) for m in methods_internal]
    stds = [np.std(ari_scores[m], ddof=1) for m in methods_internal]
    
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]

    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
    bars = ax.bar(display_names, means, yerr=stds, color=colors, edgecolor='black', 
                  alpha=0.9, capsize=8, error_kw={'elinewidth': 1.5, 'capthick': 1.5})
    
    # Dynamic ylim
    max_val = max([m + s for m, s in zip(means, stds)]) if means else 0.5
    ax.set_ylim(0, max_val * 1.25) # Give space for text labels
    
    ax.set_ylabel("ARI (Adjusted Rand Index)", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    
    for i, v in enumerate(means):
        s = stds[i]
        text = f"{v:.3f}\n+/-{s:.3f}" if s > 0.0001 else f"{v:.3f}"
        ax.text(i, v + s + (max_val * 0.02), text, ha="center", va="bottom", 
                fontsize=10, fontweight='bold')
    
    # Stylize
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.xticks(fontsize=12, fontweight='bold')
    
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)

def _plot_tsne_panel_2x2(
    out_png: Path,
    out_pdf: Path,
    coords_by_method: Dict[str, np.ndarray],
    labels: pd.Series,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    order = ["AE", "PCA", "GSVA", "ssGSEA"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 11), dpi=200)
    axes = axes.flatten()
    
    # Calculate global min/max for unified axes
    all_coords = np.concatenate(list(coords_by_method.values()), axis=0)
    x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
    y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
    
    # Add small padding
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05
    xlim = (x_min - x_pad, x_max + x_pad)
    ylim = (y_min - y_pad, y_max + y_pad)
    
    for i, method in enumerate(order):
        ax = axes[i]
        _plot_tsne(
            out_png=out_png,
            out_pdf=out_pdf,
            coords=coords_by_method[method],
            labels=labels,
            title=get_method_display(method),
            ax=ax,
            show_legend=False, # Shared legend instead
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
    # Shared title (Simplified)
    fig.suptitle("Pathway Activity Scores Separate PAM50 Subtypes", fontsize=18, fontweight='bold', y=0.98)
    
    # Create shared legend
    handles, fig_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, fig_labels, loc='center left', bbox_to_anchor=(1.02, 0.5), 
               title="PAM50 Subtypes", fontsize=12, title_fontsize=14, frameon=False)
    
    fig.tight_layout(rect=[0, 0, 1.0, 0.95])
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)


def _default_nes_source_dirs(dataset_key: str, dim: int, gene_set: str) -> Dict[str, Path]:
    """
    Return default directories where we can parse NES matrices for AE and PCA.
    Improved with fallback logic for inconsistent directory naming.
    """
    gene_set_upper = gene_set.upper()
    results_gsea = Path(__file__).resolve().parents[2] / "results" / "gsea_outputs"
    
    if dataset_key == "scanb":
        if gene_set_upper == "C6":
            ae_base = results_gsea / "SCANB_result" / "GSEA_autoencoder_C6" / f"gsea_c6_scanb_dim{dim}"
            # Fallback for folders like GSEA_PCA_C6 instead of GSEA_PCA_C6_dim64
            pca_parent = results_gsea / "SCANB_result" / f"GSEA_PCA_C6_dim{dim}"
            if not pca_parent.exists():
                pca_parent = results_gsea / "SCANB_result" / "GSEA_PCA_C6"
            pca_gsea = pca_parent / "gsea_results"
        elif gene_set_upper == "KEGG":
            ae_base = results_gsea / "SCANB_result" / "GSEA_autoencoder_kegg" / f"dim{dim}"
            pca_gsea = results_gsea / "SCANB_result" / f"GSEA_PCA_kegg_dim{dim}" / "gsea_results"
        else:
            raise ValueError(f"Unsupported gene_set={gene_set}. Expected 'C6' or 'KEGG'.")
        return {"ae": ae_base, "pca": pca_gsea}

    if dataset_key == "metabric":
        if gene_set_upper == "C6":
            ae_base = results_gsea / "Metabric_result" / "GSEA_autoencoder_C6" / f"gsea_c6_Metabric_dim{dim}" / "gsea_results"
            pca_parent = results_gsea / "Metabric_result" / f"GSEA_PCA_C6_dim{dim}"
            if not pca_parent.exists():
                pca_parent = results_gsea / "Metabric_result" / "GSEA_PCA_C6"
            pca_gsea = pca_parent / "gsea_results"
        elif gene_set_upper == "KEGG":
            ae_base = results_gsea / "Metabric_result" / "GSEA_autoencoder_kegg" / f"dim{dim}" / "gsea_results"
            pca_gsea = results_gsea / "Metabric_result" / f"GSEA_PCA_KEGG_dim{dim}" / "gsea_results"
        else:
            raise ValueError(f"Unsupported gene_set={gene_set}. Expected 'C6' or 'KEGG'.")
        return {"ae": ae_base, "pca": pca_gsea}
    
    raise ValueError(f"Unsupported dataset_key={dataset_key}")


def _load_or_build_nes(
    *,
    out_dir: Path,
    dataset_key: str,
    dim: int,
    method: str,  # 'ae' or 'pca'
    source_dir: Path,
    gene_set: str,
) -> pd.DataFrame:
    cache_path = out_dir / f"cache_nes_{method}_{dataset_key}_{gene_set}_dim{dim}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    if dataset_key == "scanb" and method == "ae":
        nes = build_nes_matrix_from_scanb_dim_dirs(source_dir, dims=dim)
    else:
        nes = build_nes_matrix_from_gsea_results_dir(source_dir, dims=dim)

    nes.to_parquet(cache_path, index=True)
    return nes


def _compute_activity(
    Z: pd.DataFrame,
    NES: pd.DataFrame,
) -> pd.DataFrame:
    """
    Z: samples x dim with columns dim_0..dim_{D-1}
    NES: dim x pathways with index dim_0..dim_{D-1}
    returns A: samples x pathways
    """
    Z_cols = list(Z.columns)
    NES_rows = list(NES.index)
    # Enforce alignment by dim names
    common_dims = [d for d in Z_cols if d in set(NES_rows)]
    if not common_dims:
        raise ValueError("No common dims between Z and NES")
    Z_aligned = Z[common_dims]
    NES_aligned = NES.loc[common_dims]

    # Original/simple version: direct multiplication (no rank/L2/cosine).
    A = Z_aligned.values @ NES_aligned.values
    return pd.DataFrame(A, index=Z.index, columns=NES_aligned.columns)


def _compute_activity_posneg(
    Z: pd.DataFrame,
    NES_plus: pd.DataFrame,
    NES_minus: pd.DataFrame,
) -> pd.DataFrame:
    """
    Positive/negative decomposition scoring (net-only):

      Score = Z_plus @ NES_plus - Z_minus @ NES_minus

    Z: samples x dim (columns dim_0..dim_{D-1})
    NES_plus: dim x pathways (index dim_0..dim_{D-1})
    NES_minus: dim x pathways (index dim_0..dim_{D-1}), magnitude (>=0)
    returns Score: samples x pathways
    """
    Z_cols = list(Z.columns)
    common_dims = [d for d in Z_cols if d in set(NES_plus.index) and d in set(NES_minus.index)]
    if not common_dims:
        raise ValueError("No common dims between Z and NES_plus/NES_minus")

    Z_aligned = Z[common_dims].astype(float)
    NESp = NES_plus.loc[common_dims]
    NESm = NES_minus.loc[common_dims]

    # Ensure pathway columns match (use intersection to be safe)
    pathways = sorted(set(NESp.columns).intersection(NESm.columns))
    if not pathways:
        raise ValueError("No common pathways between NES_plus and NES_minus")
    NESp = NESp[pathways]
    NESm = NESm[pathways]

    Z_vals = Z_aligned.values
    Z_plus = np.clip(Z_vals, 0.0, None)
    Z_minus = np.clip(-Z_vals, 0.0, None)

    score = (Z_plus @ NESp.values) - (Z_minus @ NESm.values)
    return pd.DataFrame(score, index=Z.index, columns=pathways)


def _resolve_gmt(gmt_path: Optional[str], gene_set: str) -> Path:
    if gmt_path:
        p = Path(gmt_path)
        if not p.exists():
            raise FileNotFoundError(f"Provided GMT path not found: {p}")
        return p

    gene_set_upper = gene_set.upper()
    
    # Prefer user home gsea_home gene_sets if available (via run_full_pipeline helper)
    try:
        from scripts.run_full_pipeline import find_c6_gene_set_file, find_kegg_gene_set_file

        if gene_set_upper == "C6":
            p = find_c6_gene_set_file()
        elif gene_set_upper == "KEGG":
            p = find_kegg_gene_set_file()
        else:
            raise ValueError(f"Unsupported gene_set={gene_set}. Expected 'C6' or 'KEGG'.")
        
        if p and Path(p).exists():
            return Path(p)
    except Exception:
        pass

    # Fallback: search in result directories
    fallback = find_gmt_fallback(Path(__file__).resolve().parents[2] / "results" / "gsea_outputs", gene_set)
    if fallback:
        return fallback

    raise FileNotFoundError(
        f"Could not locate {gene_set} GMT. Provide `--gmt_path` or install gene sets under "
        f"`~/gsea_home/gene_sets` (or `C:\\Users\\zhengzh\\gsea_home\\gene_sets`)."
    )


def run_one_dataset(
    *,
    dataset: str,
    dim: int,
    seed: int,
    out_root: Path,
    gene_set: str,
    c6_gmt_path: Optional[str],
    pam50_col: Optional[str],
    ae_run_dir: Optional[str],
    pca_run_dir: Optional[str],
    bulk_test_path: Optional[str],
    bulk_train_path: Optional[str],
    label_path: Optional[str],
    force_gsva: bool,
    no_scale: bool,
    save_activity_scores: bool,
    n_seeds: int = 5,
) -> None:
    # Use SCAN-B for display if the dataset is scanb
    display_name = "SCAN-B" if dataset.lower() == "scanb" else "METABRIC"
    display_gene_set = gene_set.upper()  # Ensure KEGG, C6, etc. are uppercase
    dataset_key = dataset.lower()
    defaults = _dataset_defaults(dataset_key)
    bulk_train_path = bulk_train_path or defaults["bulk_train_path"]
    bulk_test_path = bulk_test_path or defaults["bulk_test_path"]
    label_path = label_path or defaults["label_path"]

    out_dir = out_root / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)
    scale_tag = "_noscale" if no_scale else ""

    # Locate embeddings
    if ae_run_dir:
        ae_sel = RunDirSelection(run_dir=Path(ae_run_dir), test_embedding_csv=Path(ae_run_dir) / "test_embedding.csv")
    else:
        ae_sel = find_latest_ae_run_dir(dataset=dataset_key, dim=dim, result_root=Path(__file__).resolve().parents[2] / "results" / "gsea_outputs")
    if pca_run_dir:
        pca_sel = RunDirSelection(
            run_dir=Path(pca_run_dir), test_embedding_csv=Path(pca_run_dir) / "test_embedding.csv"
        )
    else:
        pca_sel = find_pca_run_dir(dataset=dataset_key, dim=dim, gene_set=gene_set, result_root=Path(__file__).resolve().parents[2] / "results" / "gsea_outputs")

    logging.info(f"[{display_name}] AE run dir:  {ae_sel.run_dir}")
    logging.info(f"[{display_name}] PCA run dir: {pca_sel.run_dir}")

    emb_ae = load_embedding_csv(ae_sel.test_embedding_csv)
    emb_pca = load_embedding_csv(pca_sel.test_embedding_csv)

    # Calculate dataset-specific test ratio based on actual cancer samples (Train vs Test count)
    try:
        def count_samples(path):
            with open(path, "r", encoding='utf-8', errors='ignore') as f:
                return len(f.readlines()) - 1
        
        n_train = count_samples(bulk_train_path)
        n_test = count_samples(bulk_test_path)
        test_ratio = n_test / (n_train + n_test)
        logging.info(f"[{display_name}] Detected dataset split: {n_train} train / {n_test} test (test_ratio={test_ratio:.4f})")
    except Exception as e:
        logging.warning(f"[{display_name}] Error calculating test_ratio: {e}. Defaulting to 0.2")
        test_ratio = 0.2

    # [CHANGE] For Metabric, we only want clinical Normal subtype (144 samples), 
    # not the added Healthy Normal (external).
    include_ext_normal = True if dataset_key.lower() == "scanb" else False
    expr = load_expression_matrix(bulk_test_path, dataset=dataset_key, log_prefix="test", include_normal=include_ext_normal, seed=seed, test_ratio=test_ratio)
    
    # Load and combine labels to ensure all Normal samples are included
    pam50 = load_labels_with_pam50(label_path=label_path, pam50_col=pam50_col)
    if dataset_key.lower() in ["scanb", "metabric"]:
        defaults = _dataset_defaults(dataset_key.lower())
        if "train_label_path" in defaults:
            train_pam50 = load_labels_with_pam50(label_path=defaults["train_label_path"], pam50_col=pam50_col)
            pam50 = pd.concat([pam50, train_pam50])
            pam50 = pam50.loc[~pam50.index.duplicated()]
        
        # [RESTORED] Keep clinical "Normal" as "Normal" for Metabric.
        # [CHANGE] Only add external normals to labels if for SCAN-B.
        if dataset_key.lower() == "scanb":
            # Explicitly label normal bulk samples as "Normal" if they are in expr but missing/NaN in pam50
            norm_train_path = Path(defaults["bulk_normal_train_path"])
            norm_test_path = Path(defaults["bulk_normal_test_path"])

            norm_dfs = []
            if norm_train_path.exists():
                norm_dfs.append(pd.read_csv(norm_train_path, index_col=0))
            if norm_test_path.exists():
                norm_dfs.append(pd.read_csv(norm_test_path, index_col=0))

            if norm_dfs:
                norm_ids = pd.concat(norm_dfs).index
                # Ensure these are marked as Normal in the labels
                combined_norm_labels = pd.Series("Normal", index=norm_ids)
                pam50 = pd.concat([pam50, combined_norm_labels])
                pam50 = pam50.loc[~pam50.index.duplicated()]

    # If AE embedding is missing samples from expr, project them.
    missing_ae = expr.index.difference(emb_ae.index)
    if not missing_ae.empty:
        logging.info(f"[{display_name}] Projecting {len(missing_ae)} missing samples through AE encoder...")
        encoder_path = ae_sel.run_dir / "encoder.pt"
        if encoder_path.exists():
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                input_dim = expr.shape[1]
                # Try to infer hidden dims from file if possible, or use a robust way.
                # Assuming standard encoder structure from training logs or model_structure.py
                # Actually, can't easily infer hidden_dims without more info.
                # Let's try to load the state dict and check weight shapes.
                sd = torch.load(encoder_path, map_location=device)
                
                # Robustly infer structure from state_dict
                sd = torch.load(encoder_path, map_location=device)
                
                # Filter for weights of Linear layers (at indices 0, 3, 6...)
                weight_keys = sorted([k for k in sd.keys() if k.endswith(".weight") and "layers" in k and int(re.search(r'layers\.(\d+)', k).group(1)) % 3 == 0], 
                                     key=lambda x: int(re.search(r'layers\.(\d+)', x).group(1)))
                
                if not weight_keys:
                    raise ValueError("No linear layer weights found in state_dict")
                
                weights_shapes = [sd[k].shape for k in weight_keys]
                input_dim = weights_shapes[0][1]
                hidden_dims = [s[0] for s in weights_shapes[:-1]]
                out_dim = weights_shapes[-1][0]
                
                logging.info(f"[{display_name}] Reconstructed Encoder: {input_dim} -> {hidden_dims} -> {out_dim}")
                
                # IMPORTANT: Ensure gene order matches AE run
                corr_dir = ae_sel.run_dir / "correlations"
                rnk_files = list(corr_dir.glob("*.rnk"))
                if rnk_files:
                    ae_genes = pd.read_csv(rnk_files[0], sep="\t", header=None)[0].tolist()
                    X_expr = expr.loc[missing_ae].reindex(columns=ae_genes).fillna(0)
                else:
                    X_expr = expr.loc[missing_ae]
                
                # Robust padding/truncation
                if X_expr.shape[1] < input_dim:
                    logging.info(f"[{display_name}] Padding features {X_expr.shape[1]} -> {input_dim}")
                    pad_cols = [f"pad_{i}" for i in range(input_dim - X_expr.shape[1])]
                    X_expr = pd.concat([X_expr, pd.DataFrame(0, index=X_expr.index, columns=pad_cols)], axis=1)
                elif X_expr.shape[1] > input_dim:
                    X_expr = X_expr.iloc[:, :input_dim]

                encoder = Encoder(input_dim, hidden_dims, out_dim).to(device)
                encoder.load_state_dict(sd)
                encoder.eval()
                with torch.no_grad():
                    Z_missing = encoder(torch.tensor(X_expr.values.astype(np.float32)).to(device)).cpu().numpy()
                
                emb_missing = pd.DataFrame(Z_missing, index=missing_ae, columns=emb_ae.columns)
                emb_ae = pd.concat([emb_ae, emb_missing])
                logging.info(f"[{display_name}] Successfully projected {len(missing_ae)} missing samples.")
            except Exception as e:
                logging.warning(f"[{display_name}] Failed to project missing AE samples: {e}")

    # Keep only test samples & labeled
    pam50 = pam50[pam50.notna()]

    # Similarly for PCA if missing normals
    missing_pca = expr.index.difference(emb_pca.index)
    if not missing_pca.empty:
        logging.info(f"[{display_name}] PCA embedding missing {len(missing_pca)} samples; recomputing PCA test embedding...")
        train_expr = load_expression_matrix(bulk_train_path, dataset=dataset_key, log_prefix="train", include_normal=True, seed=seed)
        emb_pca = compute_pca_test_embedding(train_expr=train_expr, test_expr=expr, n_components=dim, seed=seed)

    # Keep only test samples & labeled
    pam50 = pam50[pam50.notna()]

    logging.info(f"[{display_name}] Samples before alignment: emb_ae={len(emb_ae)}, emb_pca={len(emb_pca)}, expr={len(expr)}, pam50={len(pam50)}")
    emb_ae, emb_pca, expr, pam50 = _align_by_index(emb_ae, emb_pca, expr, pam50)

    subtype_counts = pam50.value_counts().to_dict()
    logging.info(f"[{display_name}] Samples after alignment: {len(pam50)}")
    logging.info(f"[{display_name}] PAM50 distribution: {subtype_counts}")

    # Determine k by actual present subtypes in test set
    k = int(pam50.nunique())
    if k < 2:
        raise ValueError(f"[{display_name}] Need at least 2 PAM50 classes to compute ARI; got {k}.")

    # Locate gene set
    gmt_path = _resolve_gmt(c6_gmt_path, gene_set=gene_set)
    logging.info(f"[{display_name}] Using GMT: {gmt_path}")

    # Ensure GSVA/ssGSEA sees the SAME gene universe as AE/PCA embedding generation:
    # use common genes between train and test (then intersect with gene set gene-union).
    try:
        train_genes = load_gene_list_from_expression_header(bulk_train_path, dataset=dataset_key)
        common_genes = sorted(set(expr.columns).intersection(train_genes))
        if common_genes:
            logging.info(f"[{display_name}] Common genes (train intersection test) for GSVA input: {len(common_genes)}")
            expr = expr[common_genes]
        else:
            logging.warning(f"[{display_name}] train intersection test common gene set is empty; GSVA input may be inconsistent.")
    except Exception as e:
        logging.warning(f"[{display_name}] Failed to enforce train intersection test gene alignment for GSVA input: {type(e).__name__}: {e}")

    # Subset expression to gene set gene-union to reduce R memory
    gene_sets = parse_gmt(gmt_path)
    keep_genes = sorted(set(expr.columns).intersection(union_genes(gene_sets)))
    if len(keep_genes) < 100:
        logging.warning(f"[{display_name}] Only {len(keep_genes)} genes overlap between expr and GMT; GSVA may be unstable.")
    expr_subset = expr[keep_genes]

    # GSVA/ssGSEA via R, with caching (pathway space)
    gsva_scores = load_or_compute_pathway_activity(
        expr_df=expr_subset,
        gmt_path=gmt_path,
        dataset=dataset_key,
        gene_set=gene_set,
        dim=dim,
        out_dir=out_dir,
        method="gsva",
        force=force_gsva,
        verbose=False,
    )
    ssgsea_scores = load_or_compute_pathway_activity(
        expr_df=expr_subset,
        gmt_path=gmt_path,
        dataset=dataset_key,
        gene_set=gene_set,
        dim=dim,
        out_dir=out_dir,
        method="ssgsea",
        force=force_gsva,
        verbose=False,
    )

    # Align GSVA/ssGSEA with current aligned sample set
    if not gsva_scores.index.equals(pam50.index):
        logging.info(f"[{display_name}] Aligning GSVA/ssGSEA scores with current sample set...")
        missing_count = len(pam50.index.difference(gsva_scores.index))
        if missing_count > 0:
            logging.warning(f"[{display_name}] Cache is missing {missing_count} samples! ARI results for GSVA/ssGSEA will contain NaNs. Consider using --force_gsva to recompute.")
        
        gsva_scores = gsva_scores.reindex(pam50.index)
        ssgsea_scores = ssgsea_scores.reindex(pam50.index)

    y_true = pam50.values

    # Build NES matrix for AE/PCA and compute pathway activity
    nes_dirs = _default_nes_source_dirs(dataset_key=dataset_key, dim=dim, gene_set=gene_set)
    nes_ae = _load_or_build_nes(
        out_dir=out_dir,
        dataset_key=dataset_key,
        dim=dim,
        method="ae",
        source_dir=nes_dirs["ae"],
        gene_set=gene_set,
    )
    nes_pca = _load_or_build_nes(
        out_dir=out_dir,
        dataset_key=dataset_key,
        dim=dim,
        method="pca",
        source_dir=nes_dirs["pca"],
        gene_set=gene_set,
    )

    A_ae = _compute_activity(emb_ae, nes_ae)
    A_pca = _compute_activity(emb_pca, nes_pca)

    # Optional: persist the pre-ARI pathway activity matrices (unscaled).
    # Note: GSVA/ssGSEA raw scores are already cached in cache_gsva_*.pkl.gz and cache_ssgsea_*.pkl.gz.
    if save_activity_scores:
        A_ae.to_pickle(out_dir / f"activity_scores_AE_{dataset_key}_{gene_set}_dim{dim}.pkl.gz", compression="gzip")
        A_pca.to_pickle(out_dir / f"activity_scores_PCA_{dataset_key}_{gene_set}_dim{dim}.pkl.gz", compression="gzip")
        gsva_scores.to_pickle(out_dir / f"activity_scores_GSVA_{dataset_key}_{gene_set}_dim{dim}.pkl.gz", compression="gzip")
        ssgsea_scores.to_pickle(out_dir / f"activity_scores_ssGSEA_{dataset_key}_{gene_set}_dim{dim}.pkl.gz", compression="gzip")

    # Standardize each feature space before clustering/t-SNE
    if no_scale:
        logging.info(f"[{display_name}] no_scale enabled: skipping StandardScaler for all methods.")
        X_ae = A_ae.values
        X_pca = A_pca.values
        X_gsva = gsva_scores.fillna(0.0).values
        X_ssgsea = ssgsea_scores.fillna(0.0).values
    else:
        X_ae = _standardize(A_ae.values)
        X_pca = _standardize(A_pca.values)
        X_gsva = _standardize(gsva_scores.fillna(0.0).values)
        X_ssgsea = _standardize(ssgsea_scores.fillna(0.0).values)

    # For ARI calculation:
    # - Metabric: Keep "Normal" (it's the clinical 144 samples).
    # - SCAN-B: Exclude "Normal" (it's the added healthy noise).
    if dataset_key == "metabric":
        ari_mask = np.ones(len(pam50), dtype=bool)
        log_msg = f"[{display_name}] Computing ARI over {n_seeds} seeds (including all clinical subtypes, {len(pam50)} samples)..."
    else:
        ari_mask = (pam50.values != "Normal")
        log_msg = f"[{display_name}] Computing ARI over {n_seeds} seeds (excluding 'Normal', {ari_mask.sum()} samples)..."

    X_ae_ari = X_ae[ari_mask]
    X_pca_ari = X_pca[ari_mask]
    X_gsva_ari = X_gsva[ari_mask]
    X_ssgsea_ari = X_ssgsea[ari_mask]
    y_true_ari = pam50.values[ari_mask]
    k_ari = int(pd.Series(y_true_ari).nunique())

    # Run K-means clustering multiple times for each method
    logging.info(log_msg)
    ari_scores = {
        "AE": [_kmeans_ari(X_ae_ari, y_true=y_true_ari, k=k_ari, seed=seed + i) for i in range(n_seeds)],
        "PCA": [_kmeans_ari(X_pca_ari, y_true=y_true_ari, k=k_ari, seed=seed + i) for i in range(n_seeds)],
        "GSVA": [_kmeans_ari(X_gsva_ari, y_true=y_true_ari, k=k_ari, seed=seed + i) for i in range(n_seeds)],
        "ssGSEA": [_kmeans_ari(X_ssgsea_ari, y_true=y_true_ari, k=k_ari, seed=seed + i) for i in range(n_seeds)],
    }
    
    ari_means = {get_method_display(m): np.mean(v) for m, v in ari_scores.items()}
    logging.info(f"[{display_name}] ARI Means: {ari_means}")

    # Save all seed scores
    all_scores = []
    for i in range(n_seeds):
        row = {"seed_idx": i}
        for m in ari_scores:
            row[m] = ari_scores[m][i]
        all_scores.append(row)
    
    scores_df = pd.DataFrame(all_scores)
    scores_path = out_dir / f"ari_scores_all_seeds_{dataset_key}_{gene_set}_dim{dim}{scale_tag}.csv"
    scores_df.to_csv(scores_path, index=False)

    # Compute t-SNE coordinates for all methods
    coords_by_method = {
        "AE": _tsne_2d(X_ae, seed=seed),
        "PCA": _tsne_2d(X_pca, seed=seed),
        "GSVA": _tsne_2d(X_gsva, seed=seed),
        "ssGSEA": _tsne_2d(X_ssgsea, seed=seed),
    }

    # 1) Save a single shared legend file for the entire dataset
    _plot_shared_legend(
        out_png=out_dir / f"tsne_shared_legend_{dataset_key}.png",
        out_pdf=out_dir / f"tsne_shared_legend_{dataset_key}.pdf",
        subtype_counts=subtype_counts
    )

    # 2) Save individual t-SNE plots without legends
    for method, coords in coords_by_method.items():
        _plot_tsne(
            out_png=out_dir / f"tsne_{method}_{dataset_key}_{gene_set}_dim{dim}{scale_tag}.png",
            out_pdf=out_dir / f"tsne_{method}_{dataset_key}_{gene_set}_dim{dim}{scale_tag}.pdf",
            coords=coords,
            labels=pam50,
            title=f"{get_method_display(method)} Pathway Activities Scores ({display_name}, {display_gene_set})",
            show_legend=False,
        )

    # Individual ARI plot with dataset name in title
    _plot_ari_bar(
        out_png=out_dir / f"ari_{dataset_key}_{gene_set}_dim{dim}{scale_tag}.png",
        out_pdf=out_dir / f"ari_{dataset_key}_{gene_set}_dim{dim}{scale_tag}.pdf",
        ari_scores=ari_scores,
        title=f"ARI Performance: {display_name} ({display_gene_set})", 
    )

    # Simple trend check (non-fatal)
    if not (ari_scores["AE"] >= ari_scores["PCA"]):
        logging.warning(f"[{display_name}] Unexpected: {get_method_display('AE')} ARI < PCA ARI. Check class balance / seed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Figure 3: Breast Cancer PAM50 clustering comparison (AE/PCA/GSVA/ssGSEA).")
    parser.add_argument("--datasets", nargs="+", default=["scanb", "Metabric"])
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--gene_set", type=str, default="C6")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_root", type=str, default=str(Path(__file__).resolve().parents[2] / "results" / "figure3" / "figure3_outputs_breast_cancer_clustering"))

    parser.add_argument("--gmt_path", type=str, default=None, help="Path to GMT file (overrides auto-detection). Can also use --c6_gmt_path for backward compatibility.")
    parser.add_argument("--c6_gmt_path", type=str, default=None, help="[Deprecated] Use --gmt_path instead. Kept for backward compatibility.")
    parser.add_argument("--pam50_col", type=str, default=None, help="Optional explicit PAM50 column name in label file.")

    # Override per-dataset paths / run dirs
    parser.add_argument("--ae_run_dir_scanb", type=str, default=None)
    parser.add_argument("--ae_run_dir_metabric", type=str, default=None)
    parser.add_argument("--pca_run_dir_scanb", type=str, default=None)
    parser.add_argument("--pca_run_dir_metabric", type=str, default=None)
    parser.add_argument("--scanb_test_path", type=str, default=None)
    parser.add_argument("--scanb_train_path", type=str, default=None)
    parser.add_argument("--scanb_label_path", type=str, default=None)
    parser.add_argument("--metabric_test_path", type=str, default=None)
    parser.add_argument("--metabric_train_path", type=str, default=None)
    parser.add_argument("--metabric_label_path", type=str, default=None)

    parser.add_argument("--force_gsva", action="store_true", help="Recompute GSVA/ssGSEA even if cache exists.")
    parser.add_argument("--no_scale", action="store_true", help="Skip StandardScaler for all methods before KMeans/t-SNE.")
    parser.add_argument(
        "--save_activity_scores",
        action="store_true",
        help="Save the pre-ARI pathway activity matrices for AE/PCA/GSVA/ssGSEA into the output directory.",
    )
    parser.add_argument("--n_seeds", type=int, default=20, help="Number of seeds for K-means (default: 20)")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    _setup_logging(args.verbose)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for ds in args.datasets:
        key = ds.lower()
        if key == "scanb":
            run_one_dataset(
                dataset=ds,
                dim=args.dim,
                seed=args.seed,
                out_root=out_root,
                gene_set=args.gene_set,
                c6_gmt_path=args.gmt_path or args.c6_gmt_path,
                pam50_col=args.pam50_col,
                ae_run_dir=args.ae_run_dir_scanb,
                pca_run_dir=args.pca_run_dir_scanb,
                bulk_test_path=args.scanb_test_path,
                bulk_train_path=args.scanb_train_path,
                label_path=args.scanb_label_path,
                force_gsva=args.force_gsva,
                no_scale=args.no_scale,
                save_activity_scores=args.save_activity_scores,
                n_seeds=args.n_seeds,
            )
        elif key == "metabric":
            run_one_dataset(
                dataset=ds,
                dim=args.dim,
                seed=args.seed,
                out_root=out_root,
                gene_set=args.gene_set,
                c6_gmt_path=args.gmt_path or args.c6_gmt_path,
                pam50_col=args.pam50_col,
                ae_run_dir=args.ae_run_dir_metabric,
                pca_run_dir=args.pca_run_dir_metabric,
                bulk_test_path=args.metabric_test_path,
                bulk_train_path=args.metabric_train_path,
                label_path=args.metabric_label_path,
                force_gsva=args.force_gsva,
                no_scale=args.no_scale,
                save_activity_scores=args.save_activity_scores,
                n_seeds=args.n_seeds,
            )
        else:
            raise ValueError(f"Unsupported dataset: {ds}")

    logging.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


