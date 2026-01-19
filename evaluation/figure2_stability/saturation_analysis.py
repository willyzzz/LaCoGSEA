#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

# -*- coding: utf-8 -*-
"""
Figure 2A: Saturation Analysis - Using FDR
Supports METABRIC and SCAN-B datasets, reads data from GSEA output directory and generates charts.
Uses FDR < 0.05/dimension as threshold.

Python 3.8 compatible.
"""

import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

import scienceplots

# =========================
# Global plotting style (BIGGER TEXT + SciencePlots)
# =========================
plt.style.use(['science', 'nature', 'no-latex'])

plt.rcParams.update({
    "axes.titlesize": 26,
    "axes.labelsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "font.size": 18,
})


# Windows Console Encoding Fix
if sys.platform == 'win32':
    import io
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass


BASE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
GSEA_OUTPUT_DIR = PROJECT_ROOT / "results" / "gsea_outputs"


def _win_long_path(path: str) -> str:
    """Windows long path handling"""
    if sys.platform == 'win32' and not path.startswith('\\\\?\\'):
        try:
            return '\\\\?\\' + os.path.abspath(path)
        except:
            pass
    return path


def count_significant_pathways_from_gsea(
    gsea_results_dir: Path,
    dim: int,
    dataset: str,
    gene_set: str,
    fdr_alpha: float = 0.05,
    divide_by_dims: bool = True,
    verbose: bool = False,
) -> int:
    """
    Count significant pathways from GSEA results directory (FDR q-val < fdr_alpha/D).
    
    Args:
        gsea_results_dir: GSEA results directory
        dim: Number of dimensions (used to calculate D)
        dataset: Dataset name (METABRIC or SCANB)
        gene_set: Gene set name (KEGG, C6, GO)
        fdr_alpha: FDR threshold
        divide_by_dims: Whether to use strict threshold FDR < alpha/D
        verbose: Whether to show detailed debug info
    
    Returns:
        Total number of unique significant pathways
    """
    gsea_results_dir = Path(gsea_results_dir).resolve()
    report_dirs = []
    
    # Standard format: gsea_results/{label}.GseaPreranked.{timestamp}/
    if gsea_results_dir.exists():
        for item in gsea_results_dir.iterdir():
            if item.is_dir() and ".GseaPreranked." in item.name:
                report_dirs.append(item)
    
    # SCANB special format: TSV files in dim{sub_dim}/ subdirectories (e.g. dim0/dim0_pos.tsv)
    if not report_dirs:
        dim_subdirs = [d for d in gsea_results_dir.iterdir() 
                      if d.is_dir() and d.name.startswith("dim") and d.name[3:].isdigit()]
        if dim_subdirs:
            report_dirs = dim_subdirs
    
    # If not found, try looking for TSV files directly in the directory
    if not report_dirs:
        tsv_files = list(gsea_results_dir.glob("gsea_report_for_na_*.tsv"))
        if tsv_files:
            report_dirs = [gsea_results_dir]
    
    if not report_dirs:
        return 0
    
    # Calculate threshold: FDR < alpha/D
    D = dim
    thr = (float(fdr_alpha) / float(D)) if (divide_by_dims and D > 0) else float(fdr_alpha)
    
    if verbose:
        print(f"  [DEBUG] Dimension {dim}: Threshold FDR < {thr:.6f} (FDR < {fdr_alpha}/{D})")
        print(f"  [DEBUG] Found {len(report_dirs)} GSEA result directories for sub-dimensions")
    
    # Collect all significant pathways (set for auto de-duplication)
    sig_pathways = set()
    subdim_pathway_counts = {}
    
    for report_dir in report_dirs:
        # Find report files
        patterns = ["gsea_report_for_na_pos*.tsv", "gsea_report_for_na_neg*.tsv"]
        
        # SCANB C6 format: dim{sub_dim}_pos.tsv, dim{sub_dim}_neg.tsv
        if report_dir.name.startswith("dim") and report_dir.name[3:].isdigit():
            dim_num = report_dir.name[3:]
            patterns = [f"dim{dim_num}_pos.tsv", f"dim{dim_num}_neg.tsv"]
        
        for pattern in patterns:
            files = list(report_dir.glob(pattern))
            if not files:
                continue
            
            tsv_path = files[0]
            try:
                tsv_abs = tsv_path.resolve()
                df = pd.read_csv(_win_long_path(str(tsv_abs)), sep="\t", engine="python")
                df.columns = df.columns.str.strip().str.replace("<.*?>", "", regex=True)
                
                # Find FDR column
                fdr_col = None
                for cand in ["FDR q-val", "FDR q-val ", "FDR q-val\r", "FDR q-val\t", 
                            "fdr q-val", "FDR", "FDR q-value", "FDR q-val (FDR)"]:
                    if cand in df.columns:
                        fdr_col = cand
                        break
                if fdr_col is None:
                    continue
                
                # Find pathway name column
                name_col = None
                for cand in ["NAME", "Term", "PATHWAY", "Gene Set"]:
                    if cand in df.columns:
                        name_col = cand
                        break
                if name_col is None:
                    continue
                
                # Convert FDR column to numeric
                df[fdr_col] = pd.to_numeric(df[fdr_col], errors="coerce")
                
                # Filter significant pathways
                sig_df = df[df[fdr_col] < thr]
                subdim_sig_count = 0
                for v in sig_df[name_col].dropna().astype(str).tolist():
                    vv = v.strip()
                    if vv:
                        sig_pathways.add(vv)
                        subdim_sig_count += 1
                
                if verbose:
                    subdim_name = report_dir.name
                    subdim_pathway_counts[subdim_name] = subdim_pathway_counts.get(subdim_name, 0) + subdim_sig_count
                        
            except Exception as e:
                print(f"[WARN] Failed to read file {tsv_path}: {e}")
                continue
    
    if verbose:
        print(f"  [DEBUG] Significant pathway counts per sub-dimension: {subdim_pathway_counts}")
        print(f"  [DEBUG] Total unique significant pathways: {len(sig_pathways)}")
    
    return len(sig_pathways)


def count_significant_pathways_from_gsea_pca(
    gsea_results_dir: Path,
    dim: int,
    dataset: str,
    gene_set: str,
    fdr_alpha: float = 0.05,
    divide_by_dims: bool = True,
    verbose: bool = False,
) -> int:
    """
    Count significant pathways from PCA GSEA results directory (FDR q-val < fdr_alpha/D).
    PCA results contain 128 dimensions, selects top N dimensions based on requested dim value.
    
    Args:
        gsea_results_dir: GSEA results directory (containing results for all 128 dimensions)
        dim: Number of dimensions to use (select top dim from 128 PCs)
        dataset: Dataset name (METABRIC or SCANB)
        gene_set: Gene set name (KEGG, C6, GO)
        fdr_alpha: FDR threshold
        divide_by_dims: Whether to use strict threshold FDR < alpha/D
        verbose: Whether to show detailed debug info
    
    Returns:
        Total number of unique significant pathways
    """
    gsea_results_dir = Path(gsea_results_dir).resolve()
    
    if not gsea_results_dir.exists():
        return 0
    
    # Find all dimension folders: {label}_dim{N}.GseaPreranked.{timestamp}/
    all_report_dirs = []
    for item in gsea_results_dir.iterdir():
        if item.is_dir() and ".GseaPreranked." in item.name:
            # Extract dimension number
            match = re.search(r'_dim(\d+)\.', item.name)
            if match:
                dim_num = int(match.group(1))
                all_report_dirs.append((dim_num, item))
    
    # Based on requested dim value, select only top N dimensions (dim0 to dim{dim-1})
    selected_dirs = [d for dim_num, d in all_report_dirs if dim_num < dim]
    selected_dirs.sort(key=lambda x: int(re.search(r'_dim(\d+)\.', x.name).group(1)))
    
    if not selected_dirs:
        if verbose:
            print(f"  [DEBUG] PCA: No result directories found for dimension {dim}")
        return 0
    
    # Calculate threshold: FDR < alpha/D
    D = dim
    thr = (float(fdr_alpha) / float(D)) if (divide_by_dims and D > 0) else float(fdr_alpha)
    
    if verbose:
        print(f"  [DEBUG] PCA Dimension {dim}: Threshold FDR < {thr:.6f} (FDR < {fdr_alpha}/{D})")
        print(f"  [DEBUG] PCA: Selecting top {dim} dimensions from 128 PCs, found {len(selected_dirs)} result directories")
    
    # Collect all significant pathways (set for auto de-duplication)
    sig_pathways = set()
    subdim_pathway_counts = {}
    
    for report_dir in selected_dirs:
        # Find report files
        patterns = ["gsea_report_for_na_pos*.tsv", "gsea_report_for_na_neg*.tsv"]
        
        for pattern in patterns:
            files = list(report_dir.glob(pattern))
            if not files:
                continue
            
            tsv_path = files[0]
            try:
                tsv_abs = tsv_path.resolve()
                df = pd.read_csv(_win_long_path(str(tsv_abs)), sep="\t", engine="python")
                df.columns = df.columns.str.strip().str.replace("<.*?>", "", regex=True)
                
                # Find FDR column
                fdr_col = None
                for cand in ["FDR q-val", "FDR q-val ", "FDR q-val\r", "FDR q-val\t", 
                            "fdr q-val", "FDR", "FDR q-value", "FDR q-val (FDR)"]:
                    if cand in df.columns:
                        fdr_col = cand
                        break
                if fdr_col is None:
                    continue
                
                # Find pathway name column
                name_col = None
                for cand in ["NAME", "Term", "PATHWAY", "Gene Set"]:
                    if cand in df.columns:
                        name_col = cand
                        break
                if name_col is None:
                    continue
                
                # Convert FDR column to numeric
                df[fdr_col] = pd.to_numeric(df[fdr_col], errors="coerce")
                
                # Filter significant pathways
                sig_df = df[df[fdr_col] < thr]
                subdim_sig_count = 0
                for v in sig_df[name_col].dropna().astype(str).tolist():
                    vv = v.strip()
                    if vv:
                        sig_pathways.add(vv)
                        subdim_sig_count += 1
                
                if verbose:
                    dim_match = re.search(r'_dim(\d+)\.', report_dir.name)
                    subdim_name = f"dim{dim_match.group(1)}" if dim_match else report_dir.name
                    subdim_pathway_counts[subdim_name] = subdim_pathway_counts.get(subdim_name, 0) + subdim_sig_count
                        
            except Exception as e:
                if verbose:
                    print(f"[WARN] Failed to read file {tsv_path}: {e}")
                continue
    
    if verbose:
        print(f"  [DEBUG] PCA Significant pathway counts per sub-dimension: {subdim_pathway_counts}")
        print(f"  [DEBUG] PCA Total unique significant pathways: {len(sig_pathways)}")
    
    return len(sig_pathways)


def load_data_from_gsea_results(
    dataset: str,
    gene_set: str,
    dims: List[int],
    fdr_alpha: float = 0.05,
    divide_by_dims: bool = True,
    verbose: bool = False,
    method: str = "LaCoGSEA",
) -> pd.DataFrame:
    """
    Load data from GSEA results directory.
    
    Args:
        dataset: Dataset name (METABRIC or SCANB)
        gene_set: Gene set name (KEGG, C6, GO)
        dims: List of dimensions
        fdr_alpha: FDR threshold
        divide_by_dims: Whether to use strict threshold FDR < alpha/D
        verbose: Whether to show detailed debug info
        method: Method name ("LaCoGSEA" or "pca")
    
    Returns:
        DataFrame with columns: dimension, total_sig, gene_set, dataset, method
    """
    dataset_upper = dataset.upper()
    gene_set_upper = gene_set.upper()
    method_lower = method.lower()
    
    # Build base directory paths
    if dataset_upper == "METABRIC":
        result_dir = GSEA_OUTPUT_DIR / "Metabric_result"
        if method_lower == "pca":
            if gene_set_upper == "KEGG":
                base_dir = result_dir / "GSEA_PCA_kegg"
            elif gene_set_upper == "C6":
                base_dir = result_dir / "GSEA_PCA_C6"
            elif gene_set_upper == "GO":
                base_dir = result_dir / "GSEA_PCA_go"
            else:
                print(f"[ERROR] Unknown gene set: {gene_set_upper}")
                return pd.DataFrame()
        else:
            if gene_set_upper == "GO":
                base_dir_candidate1 = result_dir / "GSEA_auto_encoder_go"
                base_dir_candidate2 = result_dir / "GSEA_autoencoder_go"
                if base_dir_candidate1.exists():
                    base_dir = base_dir_candidate1
                elif base_dir_candidate2.exists():
                    base_dir = base_dir_candidate2
                else:
                    base_dir = base_dir_candidate1
            elif gene_set_upper == "KEGG":
                base_dir = result_dir / "GSEA_autoencoder_kegg"
            elif gene_set_upper == "C6":
                base_dir = result_dir / "GSEA_autoencoder_C6"
            else:
                print(f"[ERROR] Unknown gene set: {gene_set_upper}")
                return pd.DataFrame()
    elif dataset_upper == "SCANB":
        result_dir = GSEA_OUTPUT_DIR / "SCANB_result"
        if method_lower == "pca":
            if gene_set_upper == "KEGG":
                base_dir = result_dir / "GSEA_PCA_kegg"
            elif gene_set_upper == "C6":
                base_dir = result_dir / "GSEA_PCA_C6"
            elif gene_set_upper == "GO":
                base_dir = result_dir / "GSEA_PCA_go"
            else:
                print(f"[ERROR] Unknown gene set: {gene_set_upper}")
                return pd.DataFrame()
        else:
            if gene_set_upper == "GO":
                base_dir = result_dir / "GSEA_autoencoder_go"
            elif gene_set_upper == "KEGG":
                base_dir = result_dir / "GSEA_autoencoder_kegg"
            elif gene_set_upper == "C6":
                base_dir = result_dir / "GSEA_autoencoder_C6"
            else:
                print(f"[ERROR] Unknown gene set: {gene_set_upper}")
                return pd.DataFrame()
    else:
        print(f"[ERROR] Unknown dataset: {dataset_upper}")
        return pd.DataFrame()
    
    if not base_dir.exists():
        print(f"[WARN] Directory not found: {base_dir}")
        return pd.DataFrame()
    
    results = []
    
    for dim in dims:
        gsea_results_dir = None
        
        if method_lower == "pca":
            gsea_results_dir = base_dir / "gsea_results"
            if not gsea_results_dir.exists():
                if verbose:
                    print(f"[WARN] PCA GSEA result directory not found: {gsea_results_dir}")
                continue
            
            total_sig = count_significant_pathways_from_gsea_pca(
                gsea_results_dir=gsea_results_dir,
                dim=dim,
                dataset=dataset_upper,
                gene_set=gene_set_upper,
                fdr_alpha=fdr_alpha,
                divide_by_dims=divide_by_dims,
                verbose=verbose,
            )
        else:
            if dataset_upper == "METABRIC":
                dim_dir = base_dir / f"dim{dim}"
                gsea_results_dir_candidate = dim_dir / "gsea_results"
                if gsea_results_dir_candidate.exists():
                    gsea_results_dir = gsea_results_dir_candidate
                else:
                    if gene_set_upper == "GO":
                        pattern = f"full_pipeline_Metabric_dim{dim}_*"
                        matching_dirs = list(base_dir.glob(pattern))
                        if matching_dirs:
                            matching_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                            full_pipeline_dir = matching_dirs[0]
                            gsea_results_dir_candidate = full_pipeline_dir / "gsea_results"
                            if gsea_results_dir_candidate.exists():
                                gsea_results_dir = gsea_results_dir_candidate
                    
                    if gsea_results_dir is None and gene_set_upper == "C6":
                        c6_dir = base_dir / f"gsea_c6_Metabric_dim{dim}"
                        if c6_dir.exists():
                            c6_gsea_dir = c6_dir / "gsea_results"
                            if c6_gsea_dir.exists():
                                gsea_results_dir = c6_gsea_dir
                            else:
                                gsea_results_dir = c6_dir
            
            elif dataset_upper == "SCANB":
                if gene_set_upper == "KEGG":
                    dim_dir = base_dir / f"dim{dim}"
                    if dim_dir.exists():
                        gsea_results_dir = dim_dir
                    else:
                        gsea_dir = base_dir / f"gsea_kegg_scanb_dim{dim}"
                        if gsea_dir.exists():
                            gsea_results_dir = gsea_dir
                elif gene_set_upper == "C6":
                    gsea_dir = base_dir / f"gsea_c6_scanb_dim{dim}"
                    if gsea_dir.exists():
                        gsea_results_dir = gsea_dir
                elif gene_set_upper == "GO":
                    pattern = f"full_pipeline_scanb_GO_AUTOENCODER_dim{dim}_*"
                    matching_dirs = list(base_dir.glob(pattern))
                    if matching_dirs:
                        matching_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                        full_pipeline_dir = matching_dirs[0]
                        gsea_results_dir_candidate = full_pipeline_dir / "gsea_results"
                        if gsea_results_dir_candidate.exists():
                            gsea_results_dir = gsea_results_dir_candidate
            
            if gsea_results_dir is None or not gsea_results_dir.exists():
                if verbose:
                    print(f"[WARN] GSEA result directory not found for dimension {dim}")
                continue
            
            total_sig = count_significant_pathways_from_gsea(
                gsea_results_dir=gsea_results_dir,
                dim=dim,
                dataset=dataset_upper,
                gene_set=gene_set_upper,
                fdr_alpha=fdr_alpha,
                divide_by_dims=divide_by_dims,
                verbose=verbose,
            )
        
        results.append({
            'dimension': dim,
            'total_sig': total_sig,
            'gene_set': gene_set_upper,
            'dataset': dataset_upper,
            'method': method_lower.upper(),
        })
    
    if not results:
        return pd.DataFrame()
    
    return pd.DataFrame(results)


def plot_saturation(
    df: pd.DataFrame,
    gene_set: str,
    dataset: str,
    out_dir: Path,
    fdr_alpha: float = 0.05,
    divide_by_dims: bool = True,
):
    """
    Plot saturation analysis chart, containing LaCoGSEA and PCA lines.
    """
    if df.empty:
        print(f"[WARN] No data for {dataset} {gene_set}, skipping plot")
        return
    
    fig, ax = plt.subplots(figsize=(7, 5), layout='constrained')
    
    colors = {"LACOGSEA": "#2E86AB", "PCA": "#A23B72"}
    markers = {"LACOGSEA": "o", "PCA": "s"}
    
    methods = df["method"].unique() if "method" in df.columns else []
    if len(methods) == 0:
        print(f"[WARN] No method column found for {dataset} {gene_set}, skipping plot")
        return
    
    for method in sorted(methods):
        df_method = df[df["method"] == method].sort_values("dimension")
        if df_method.empty:
            continue
        
        ax.plot(
            df_method["dimension"],
            df_method["total_sig"],
            marker=markers.get(method, "o"),
            linewidth=2,
            markersize=10,
            color=colors.get(method, "#2E86AB"),
            label=method,
        )
        
        for _, row in df_method.iterrows():
            ax.annotate(
                str(int(row["total_sig"])),
                xy=(row["dimension"], row["total_sig"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=16,
                fontweight="heavy",
                color=colors.get(method, "#2E86AB"),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, 
                         edgecolor=colors.get(method, "#2E86AB"), linewidth=1.5),
            )
    
    fdr_label = f"FDR $<$ {fdr_alpha}/D" if divide_by_dims else f"FDR $<$ {fdr_alpha}"
    ax.set_xlabel("Latent Dimension", fontweight="bold")
    ax.set_ylabel(f"Significant Pathway Count\n({fdr_label})", fontweight="bold")
    
    display_dataset = dataset.replace("SCANB", "SCAN-B")
    ax.set_title(f"Saturation Analysis - {gene_set} ({display_dataset})", fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3)
    
    ax.set_xscale("log", base=2)
    all_dims = sorted(df["dimension"].unique())
    ax.set_xticks(all_dims)
    from matplotlib.ticker import ScalarFormatter
    ax.xaxis.set_major_formatter(ScalarFormatter())
    
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path_png = out_dir / f"Figure1A_Saturation_{dataset}_{gene_set}_FDR.png"
    fig_path_pdf = out_dir / f"Figure1A_Saturation_{dataset}_{gene_set}_FDR.pdf"
    plt.savefig(fig_path_png, dpi=600)
    plt.savefig(fig_path_pdf)
    plt.close()
    
    print(f"[OK] Saved image: {fig_path_png}")
    print(f"[OK] Saved image: {fig_path_pdf}")


def save_shared_legend(out_dir: Path):
    """Save legend separately"""
    fig, ax = plt.subplots(figsize=(4, 1))
    colors = {"LACOGSEA": "#2E86AB", "PCA": "#A23B72"}
    markers = {"LACOGSEA": "o", "PCA": "s"}
    
    ax.plot([], [], marker=markers["LACOGSEA"], color=colors["LACOGSEA"], linewidth=2, markersize=10, label="LaCoGSEA")
    ax.plot([], [], marker=markers["PCA"], color=colors["PCA"], linewidth=2, markersize=10, label="PCA")
    
    ax.axis('off')
    legend = ax.legend(loc="center", ncol=2, frameon=False, fontsize=20)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path_png = out_dir / "Figure1A_Legend.png"
    fig_path_pdf = out_dir / "Figure1A_Legend.pdf"
    
    def export_legend(legend, filename="legend.png"):
        fig = legend.figure
        fig.canvas.draw()
        bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(filename, dpi=600, bbox_inches=bbox)

    export_legend(legend, str(fig_path_png))
    export_legend(legend, str(fig_path_pdf))
    plt.close()
    print(f"[OK] Saved legend: {fig_path_png}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Figure 1A: Saturation analysis from GSEA results (FDR)")
    ap.add_argument(
        "--datasets",
        type=str,
        default="METABRIC,SCANB",
        help="Datasets to analyze, comma separated (default: METABRIC,SCANB)",
    )
    ap.add_argument(
        "--gene_sets",
        type=str,
        default="KEGG,C6,GO",
        help="Gene sets to analyze, comma separated (default: KEGG,C6,GO)",
    )
    ap.add_argument(
        "--dims",
        type=str,
        default="1,2,4,8,16,32,64,128",
        help="Dimension list to analyze, comma separated (default: 1,2,4,8,16,32,64,128)",
    )
    ap.add_argument(
        "--fdr_alpha",
        type=float,
        default=0.05,
        help="FDR threshold (default: 0.05)",
    )
    ap.add_argument(
        "--fdr_divide",
        action="store_true",
        default=True,
        help="Use strict threshold FDR < alpha/D (D=total dimensions, default: True)",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str(PROJECT_ROOT / "results" / "figure1" / "figure1a_outputs_saturation"),
        help="Output directory",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed debug info",
    )
    args = ap.parse_args()
    
    datasets = [ds.strip().upper() for ds in args.datasets.split(",")]
    gene_sets = [gs.strip().upper() for gs in args.gene_sets.split(",")]
    dims = sorted(set([int(d.strip()) for d in args.dims.split(",")]))
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Figure 1A: Saturation Analysis (using FDR, directly from GSEA results)")
    print("=" * 80)
    print(f"Datasets: {datasets}")
    print(f"Gene Sets: {gene_sets}")
    print(f"Dimensions: {dims}")
    print(f"FDR Threshold: {args.fdr_alpha}")
    print(f"Use FDR < alpha/D: {args.fdr_divide}")
    print()
    
    all_data = []
    
    for dataset in datasets:
        for gene_set in gene_sets:
            print("=" * 60)
            print(f"Loading {dataset} {gene_set} data (LaCoGSEA + PCA)...")
            print("=" * 60)
            
            df_LaCoGSEA = load_data_from_gsea_results(
                dataset=dataset,
                gene_set=gene_set,
                dims=dims,
                fdr_alpha=args.fdr_alpha,
                divide_by_dims=args.fdr_divide,
                verbose=args.verbose,
                method="LaCoGSEA",
            )
            
            df_pca = load_data_from_gsea_results(
                dataset=dataset,
                gene_set=gene_set,
                dims=dims,
                fdr_alpha=args.fdr_alpha,
                divide_by_dims=args.fdr_divide,
                verbose=args.verbose,
                method="pca",
            )
            
            dfs_to_merge = []
            if df_LaCoGSEA is not None and not df_LaCoGSEA.empty:
                dfs_to_merge.append(df_LaCoGSEA)
                print(f"\n{dataset} {gene_set} LaCoGSEA data summary:")
                print(df_LaCoGSEA[['dimension', 'total_sig']].to_string(index=False))
            else:
                print(f"[WARN] No LaCoGSEA data loaded for {dataset} {gene_set}")
            
            if df_pca is not None and not df_pca.empty:
                dfs_to_merge.append(df_pca)
                print(f"\n{dataset} {gene_set} PCA data summary:")
                print(df_pca[['dimension', 'total_sig']].to_string(index=False))
            else:
                print(f"[WARN] No PCA data loaded for {dataset} {gene_set}")
            
            if dfs_to_merge:
                df_combined = pd.concat(dfs_to_merge, ignore_index=True)
                all_data.append(df_combined)
            print()
    
    if not all_data:
        print("\n[ERROR] No data loaded!")
        return 1
    
    df_combined = pd.concat(all_data, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    summary_path = out_dir / "saturation_summary_fdr.csv"
    df_combined.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved summary data: {summary_path}")
    
    print("\n" + "=" * 60)
    print("Summary Results:")
    print("=" * 60)
    print(df_combined.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("Plotting line charts...")
    print("=" * 60)
    
    for dataset in datasets:
        for gene_set in gene_sets:
            df_subset = df_combined[
                (df_combined['dataset'] == dataset) & 
                (df_combined['gene_set'] == gene_set)
            ]
            plot_saturation(df_subset, gene_set, dataset, out_dir, args.fdr_alpha, args.fdr_divide)
    
    save_shared_legend(out_dir)
    return 0

if __name__ == "__main__":
    sys.exit(main())
