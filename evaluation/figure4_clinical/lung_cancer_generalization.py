#!/usr/bin/env python
from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from scipy.stats import ttest_ind, mannwhitneyu
import matplotlib.pyplot as plt
try:
    import scienceplots
except ImportError:
    pass

from scripts.figure3.figure3_breast_cancer_clustering import (
    _compute_activity,
    _resolve_gmt,
    load_expression_matrix,
)
from core.io_utils import (
    RunDirSelection,
    find_latest_ae_run_dir,
    load_embedding_csv,
)
from core.nes_from_gsea_reports import (
    build_nes_matrix_from_gsea_results_dir,
)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def load_lung_subtype_labels(label_path: str | Path) -> pd.Series:
    """
    Load LUAD/LUSC subtype labels from label file.
    
    Args:
        label_path: Path to label CSV file with 'Subtype' column
    
    Returns:
        Series with sample IDs as index and Subtype (LUAD/LUSC) as values
    """
    label_df = pd.read_csv(label_path, index_col=0)
    label_df.index = label_df.index.astype(str)
    label_df.sort_index(inplace=True)
    
    if 'Subtype' not in label_df.columns:
        raise ValueError(f"Label file must contain 'Subtype' column. Found columns: {list(label_df.columns)}")
    
    subtypes = label_df['Subtype'].copy()
    subtypes = subtypes[subtypes.notna()]
    
    # Validate subtypes
    valid_subtypes = {'LUAD', 'LUSC'}
    found_subtypes = set(subtypes.unique())
    invalid = found_subtypes - valid_subtypes
    if invalid:
        logging.warning(f"Found invalid subtypes: {invalid}. Expected: {valid_subtypes}")
    
    logging.info(f"Loaded {len(subtypes)} samples with subtypes: {subtypes.value_counts().to_dict()}")
    return subtypes


def _default_nes_source_dirs(dataset_key: str, dim: int, gene_set: str) -> Dict[str, Path]:
    """
    Get default NES source directories for TCGA_Lung dataset.
    
    Returns:
        Dict with keys: 'ae', 'pca' (if available)
    """
    result_root = Path(__file__).resolve().parents[2] / "results" / "gsea_outputs"
    
    if dataset_key == "tcga_lung":
        # Check for actual directory structure
        # Option 1: TCGA_Lung_result/GSEA_autoencoder_kegg/dim64/gsea_results
        path1 = result_root / "TCGA_Lung_result" / f"GSEA_autoencoder_{gene_set.lower()}" / f"dim{dim}" / "gsea_results"
        # Option 2: TCGA_Lung_cancer_kegg/gsea_results (actual structure)
        path2 = result_root / "TCGA_Lung_cancer_kegg" / "gsea_results"
        
        if path2.exists():
            return {"ae": path2}
        elif path1.exists():
            return {"ae": path1}
        else:
            # Try to find any matching directory
            candidates = list(result_root.glob(f"*TCGA_Lung*{gene_set.lower()}*"))
            if candidates:
                gsea_dir = candidates[0] / "gsea_results"
                if gsea_dir.exists():
                    return {"ae": gsea_dir}
            raise FileNotFoundError(f"Could not find GSEA results for TCGA_Lung dim{dim} {gene_set}")
    else:
        raise ValueError(f"Unknown dataset_key: {dataset_key}")


def _format_pathway_name(name: str) -> str:
    """Format pathway name to Sentence case, preserving acronyms and condensing."""
    s = name.replace('KEGG_', '').replace('GO_', '').replace('REACTOME_', '')
    s = s.replace('_', ' ')
    
    # Condense: Keep 'signaling', but remove trailing 'pathway'
    s = s.lower().replace(' signaling pathway', ' signaling').replace(' pathway', '')
    
    # Capitalize first letter
    s = s.capitalize()
    
    # Whitelist of acronyms to keep uppercase
    acronyms = {
        "Dna", "Rna", "Atp", "Gtp", "Nad", "Nadp", "Hiv", "Htlv", "Hsv", 
        "Hcmv", "Ebv", "Mapk", "Pi3sk", "Mtor", "Jak", "Stat", "Nfkb", "Vegf", 
        "Tcr", "Bcr", "Rigor", "Vibrio", "Legionella", "Leishmania", "Staphylococcus",
        "Her2", "Erbb2", "Brca1", "Brca2", "Gaba", "Gmp", "Camp", "Cgmp", "P53", "Nod", "Rig", "Toll",
        "Abc", "Tcga", "Luad", "Lusc", "Alanine", "Brca", "Adh"
    }
    
    # Split and check words
    words = s.split()
    new_words = []
    for w in words:
        if w.title() in acronyms:
             new_words.append(w.upper())
        elif w.capitalize() in acronyms:
             new_words.append(w.upper())
        elif w.upper() == 'P53':
            new_words.append('p53')
        else:
            new_words.append(w)
            
    return " ".join(new_words)

def _format_pvalue_lung(p_val: float) -> str:
    if p_val < 0.0001:
        return "P $<$ 0.0001"
    else:
        return f"P = {p_val:.4f}"


def plot_selected_pathways_distribution(
    pathway_scores: pd.DataFrame,
    labels: pd.Series,
    pathways: list[str],
    out_png: Path,
    out_pdf: Path,
    title: str,
) -> None:
    """
    Plot boxplot/violin plot of activity scores for selected pathways (e.g. Top Pos, Top Neg).
    Vertical layout, compact.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Align data
    common_idx = pathway_scores.index.intersection(labels.index)
    pathway_aligned = pathway_scores.loc[common_idx]
    labels_aligned = labels.loc[common_idx]
    
    # Filter valid pathways
    valid_pathways = [p for p in pathways if p in pathway_aligned.columns]
    if not valid_pathways:
        logging.warning("No valid pathways found for distribution plot.")
        return
        
    plot_data_list = []
    for p in valid_pathways:
        pd_temp = pd.DataFrame({
            'Activity Score': pathway_aligned[p],
            'Subtype': labels_aligned,
            'Pathway': p.replace('KEGG_', '').replace('_', ' ')
        })
        plot_data_list.append(pd_temp)
    
    plot_data = pd.concat(plot_data_list).dropna()
    plot_data = plot_data[plot_data['Subtype'].isin(['LUAD', 'LUSC'])]
    
    if len(plot_data) == 0:
        return

    if len(plot_data) == 0:
        return
    
    palette = {"LUAD": "#3b5387", "LUSC": "#d94728"}
    
    for i, p_name in enumerate(valid_pathways):
        # Format name for display
        display_name = _format_pathway_name(p_name)
        
        # Determine output filename suffix
        # If it's a known top/neg, we might want to label it, but let's just use the pathway name logic or index
        # User asked for "separate files". 
        # Filename: Figure4D_{formatted_name}_{dataset...}.png
        safe_name = p_name.replace('KEGG_', '').lower()
        
        fig, ax = plt.subplots(figsize=(3, 5), layout='constrained', dpi=600)
        
        subset = plot_data[plot_data['Pathway'] == p_name.replace('KEGG_', '').replace('_', ' ')]
        
        # Boxplot with updated style - even tighter
        sns.boxplot(
            data=subset,
            x='Subtype',
            y='Activity Score',
            palette=palette,
            ax=ax,
            width=0.4,    # Even tighter boxes
            fliersize=0,
            linewidth=1.5
        )
        
        # Calculate P-value
        luad = subset[subset['Subtype'] == 'LUAD']['Activity Score']
        lusc = subset[subset['Subtype'] == 'LUSC']['Activity Score']
        try:
            _, p_val = mannwhitneyu(luad, lusc)
            p_text = _format_pvalue_lung(p_val)
        except:
            p_text = ""
            
        # Annotate P-value inside plot
        y_max = subset['Activity Score'].max()
        y_min = subset['Activity Score'].min()
        rng = y_max - y_min
        
        # Use simple string, avoid LaTeX if it causes issues with <
        # Place it inside the plot area, near the top bracket or just floating
        # Draw a bracket first
        bracket_y = y_max + 0.02 * rng
        bracket_h = 0.02 * rng
        
        # Get x-coords for LUAD (0) and LUSC (1) since we sorted or Seaborn default
        # Seaborn defaults to sorted order of appearance or 'order' param. 
        # With 2 categories, typically 0 and 1.
        
        ax.plot([0, 0, 1, 1], [bracket_y, bracket_y+bracket_h, bracket_y+bracket_h, bracket_y], c='k', lw=1.5)
        
        ax.text(0.5, bracket_y + 2.5*bracket_h, p_text, ha='center', va='bottom', 
                fontsize=12, fontweight='bold', color='black')

        ax.set_title(display_name, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel("", fontsize=1) # Hide x label text
        ax.set_ylabel("Pathway Activity Score", fontsize=13, fontweight='bold')
        
        ax.tick_params(axis='x', rotation=45, labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        
        # Ensure y-limits accommodate the bracket and text
        ax.set_ylim(y_min - 0.05*rng, y_max + 0.15*rng)
        
        # Save separate file
        this_png = out_png.parent / f"{out_png.stem}_{i+1}_{safe_name}.png"
        this_pdf = out_pdf.parent / f"{out_pdf.stem}_{i+1}_{safe_name}.pdf"
        
        fig.savefig(this_png, dpi=600)
        fig.savefig(this_pdf)
        plt.close(fig)
        logging.info(f"Saved pathways distribution plot to {this_png}")


def _load_or_build_nes(
    out_dir: Path,
    dataset_key: str,
    dim: int,
    method: str,
    source_dir: Path,
    gene_set: str,
) -> pd.DataFrame:
    """
    Load or build NES matrix from GSEA results.
    
    Args:
        out_dir: Output directory for caching
        dataset_key: Dataset identifier
        dim: Dimension
        method: Method name ('ae' or 'pca')
        source_dir: Directory containing GSEA results
        gene_set: Gene set name (KEGG, GO, etc.)
    
    Returns:
        DataFrame with pathways as columns and dimensions as index
    """
    cache_file = out_dir / f"nes_{dataset_key}_{gene_set}_dim{dim}_{method}.parquet"
    
    if cache_file.exists():
        logging.info(f"Loading cached NES matrix from {cache_file}")
        return pd.read_parquet(cache_file)
    
    if not source_dir.exists():
        raise FileNotFoundError(f"GSEA results directory not found: {source_dir}")
    
    logging.info(f"Building NES matrix from {source_dir}")
    nes_matrix = build_nes_matrix_from_gsea_results_dir(source_dir, dims=dim)
    
    if nes_matrix.empty:
        raise ValueError(f"No NES values found in {source_dir}")
    
    # Cache the result
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    nes_matrix.to_parquet(cache_file)
    logging.info(f"Cached NES matrix to {cache_file}")
    
    return nes_matrix


def plot_tsne_subtype_separation(
    embeddings: pd.DataFrame,
    labels: pd.Series,
    out_png: Path,
    out_pdf: Path,
    title: str,
) -> None:
    """
    Plot t-SNE visualization showing LUAD vs LUSC separation.
    
    Args:
        embeddings: DataFrame with samples x dimensions
        labels: Series with Subtype (LUAD/LUSC)
        out_png: Output PNG path
        out_pdf: Output PDF path
        title: Plot title
    """
    import matplotlib.pyplot as plt
    
    # Align data
    common_idx = embeddings.index.intersection(labels.index)
    if len(common_idx) < 10:
        logging.warning(f"Too few common samples for t-SNE: {len(common_idx)}")
        return
    
    emb_aligned = embeddings.loc[common_idx]
    labels_aligned = labels.loc[common_idx]
    
    # Compute t-SNE
    n_samples = len(emb_aligned)
    perplexity = min(30, max(5, (n_samples - 1) // 3))
    
    logging.info(f"Computing t-SNE with perplexity={perplexity} for {n_samples} samples...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=42,
        n_iter=1000,
    )
    coords = tsne.fit_transform(emb_aligned.values)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
    
    palette = {
        "LUAD": "#3b5387",  # Blue
        "LUSC": "#d94728",  # Red
    }
    
    for subtype in ["LUAD", "LUSC"]:
        mask = labels_aligned.values == subtype
        if mask.sum() == 0:
            continue
        
        # Larger dots for better visibility as in Figure 3
        s_size = 40 if subtype == "LUSC" else 35
        
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=s_size,
            c=palette.get(subtype, "#333333"),
            label=f"{subtype} (n={int(mask.sum())})",
            alpha=0.75,
            linewidths=0.3,
            edgecolors="white",
        )
    
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.set_xlabel("t-SNE 1", fontsize=12, fontweight='bold')
    ax.set_ylabel("t-SNE 2", fontsize=12, fontweight='bold')
    
    # Remove legend from scatter plot as we will save a shared one
    # ax.legend(loc="best", frameon=True, fontsize=10)
    ax.grid(False)
    
    fig.tight_layout()
    fig.savefig(out_png, dpi=600, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    
    logging.info(f"Saved t-SNE plot to {out_png}")

def _plot_shared_legend_lung(out_png: Path, out_pdf: Path, subtype_counts: Dict[str, int]) -> None:
    import matplotlib.pyplot as plt
    
    palette = {
        "LUAD": "#3b5387",
        "LUSC": "#d94728",
    }

    fig, ax = plt.subplots(figsize=(3, 3), facecolor='white')
    for subtype in ["LUAD", "LUSC"]:
        count = subtype_counts.get(subtype, 0)
        label = f"{subtype} (n={count})"
        ax.scatter([], [], c=palette[subtype], label=label, s=160, alpha=1.0, 
                   edgecolors="white", linewidths=0.8)
    
    legend = ax.legend(
        loc="center", 
        frameon=False, 
        fontsize=14, 
        title="TCGA Lung Subtypes", 
        title_fontsize=16,
        labelspacing=1.2
    )
    for h in legend.legendHandles:
        h.set_alpha(1.0)
    
    ax.axis('off')
    fig.savefig(out_png, dpi=600, bbox_inches='tight', transparent=False, facecolor='white')
    fig.savefig(out_pdf, bbox_inches='tight', transparent=False, facecolor='white')
    plt.close(fig)


def compute_pathway_differential_statistics_lung(
    pathway_scores: pd.DataFrame,
    labels: pd.Series,
    bootstrap_n: int = 0,
) -> pd.DataFrame:
    """
    Compute differential statistics between LUAD and LUSC.
    Includes bootstrap resampling if bootstrap_n > 0.
    """
    # Align data
    common_idx = pathway_scores.index.intersection(labels.index)
    pathway_aligned = pathway_scores.loc[common_idx]
    labels_aligned = labels.loc[common_idx]
    
    # Filter to LUAD and LUSC only
    luad_mask = labels_aligned == 'LUAD'
    lusc_mask = labels_aligned == 'LUSC'
    
    luad_scores = pathway_aligned[luad_mask]
    lusc_scores = pathway_aligned[lusc_mask]
    
    if len(luad_scores) < 2 or len(lusc_scores) < 2:
        return pd.DataFrame()

    def calc_t_stats_lung(lusc_df, luad_df):
        stats = {}
        for pathway in lusc_df.columns:
            v1 = lusc_df[pathway].dropna().values
            v2 = luad_df[pathway].dropna().values
            if len(v1) < 2 or len(v2) < 2: continue
            try:
                t, p = ttest_ind(v1, v2, equal_var=False)
                stats[pathway] = (t, p)
            except:
                stats[pathway] = (0, 1)
        return stats

    # Base stats
    base_stats = calc_t_stats_lung(lusc_scores, luad_scores)
    results = []
    for pathway, (t, p) in base_stats.items():
        results.append({
            'pathway': pathway,
            't_statistic': t,
            'p_value': p,
            'direction': 'LUSC' if t > 0 else 'LUAD',
            'abs_t_statistic': abs(t),
            't_std': 0.0
        })
    df = pd.DataFrame(results)
    
    if bootstrap_n > 0 and not df.empty:
        logging.info(f"Running bootstrap (n={bootstrap_n}) for Figure 4 stats...")
        boot_t = []
        for i in range(bootstrap_n):
            lusc_boot = lusc_scores.sample(frac=1.0, replace=True, random_state=i)
            luad_boot = luad_scores.sample(frac=1.0, replace=True, random_state=i+1000)
            bs = calc_t_stats_lung(lusc_boot, luad_boot)
            boot_t.append({p: s[0] for p, s in bs.items() if p in df['pathway'].values})
        
        boot_df = pd.DataFrame(boot_t)
        df['t_std'] = df['pathway'].map(boot_df.std()).fillna(0)
        
    return df.sort_values('abs_t_statistic', ascending=False)


def plot_differential_pathways_diverging_lung(
    diff_stats: pd.DataFrame,
    out_png: Path,
    out_pdf: Path,
    title: str,
    top_n: int = 15,
    show_legend: bool = True,
) -> None:
    """
    Plot diverging bar chart for top differential pathways with blurred uncertainty.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    
    if diff_stats.empty:
        return
    
    plot_df = diff_stats.copy()
    
    # Separate positive and negative
    pos = plot_df[plot_df['t_statistic'] > 0].sort_values('t_statistic', ascending=True).tail(top_n)
    neg = plot_df[plot_df['t_statistic'] < 0].sort_values('t_statistic', ascending=True).head(top_n)
    
    # Concatenate: pos (top), neg (bottom)
    # But for barh, y=0 is at bottom. So we want neg at bottom, pos at top.
    # pos is sorted ascending (small -> large). neg is sorted ascending (large neg -> small neg).
    # We want visual order: top is large positive, bottom is large negative.
    # Using barh, the list index 0 is at bottom.
    # So we want order: [most negative ... least negative ... least positive ... most positive]
    # plot_df sorted by t_statistic ascending provides exactly this.
    
    # Wait, request says "positive most at top, negative most at bottom".
    # barh plots index 0 at bottom.
    # So index len-1 should be most positive. index 0 should be most negative.
    # Sorting by t-stat ascending does exactly this:
    # [-10, -5, ..., 5, 10] -> -10 is index 0 (bottom), 10 is index N (top).
    
    # So we just need to take the top N positive and top N negative (absolute) and combine them.
    # Actually user says "top 5 most significant pathways".
    # Let's take top_n pos and top_n neg if possible, or just top_n total by abs value.
    # The function argument is top_n. Let's interpret it as top N positive and top N negative for symmetry if space allows, 
    # or just keep the logic "top N differential".
    # User said "positive most at top, negative most at bottom".
    
    # Let's retain logic: select by abs t-stat (already passed in `diff_stats` usually sorted by abs). 
    # Just ensure sorting for display.
    
    # Re-select top N by absolute t statistic
    plot_df = plot_df.sort_values('abs_t_statistic', ascending=False).head(top_n)
    
    # Sort by t_statistic descending (Most Positive at Index 0)
    plot_df = plot_df.sort_values('t_statistic', ascending=False)
    
    # Format pathway name using helper (Title Case)
    plot_df['pathway_display'] = plot_df['pathway'].apply(_format_pathway_name)
    
    # Use constrained layout
    fig, ax = plt.subplots(figsize=(8, max(5, len(plot_df) * 0.4)), layout='constrained', dpi=600)
    y_pos = np.arange(len(plot_df))
    
    # Plot bars
    colors = ['#d94728' if x > 0 else '#3b5387' for x in plot_df['t_statistic']]
    ax.barh(y_pos, plot_df['t_statistic'], color=colors, alpha=0.75, zorder=3)
    
    # Blurred Error and Dashed Mean
    bar_height = 0.8
    means = plot_df['t_statistic'].values
    stds = plot_df['t_std'].values if 't_std' in plot_df else [0]*len(plot_df)
    
    for i, (m, s) in enumerate(zip(means, stds)):
        if s > 0:
            ax.fill_betweenx([y_pos[i] - bar_height/2, y_pos[i] + bar_height/2], 
                             m - s, m + s, color=colors[i], alpha=0.35, zorder=2)
        ax.vlines(m, y_pos[i] - bar_height/2, y_pos[i] + bar_height/2, 
                  colors='#222222', linestyles='--', linewidth=1.5, alpha=0.85, zorder=4)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df['pathway_display'], fontsize=12, fontweight='bold')
    ax.set_xlabel('T-statistic (LUSC vs LUAD)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.axvline(0, color='grey', linestyle='-', linewidth=0.8, zorder=1)
    
    ax.grid(axis='x', linestyle='--', alpha=0.3, zorder=0)
    ax.invert_yaxis() # Top significant (Index 0) at top
    
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_pdf)
    plt.close(fig)
    
    logging.info(f"Saved differential pathways plot to {out_png}")


def _plot_figure4b_legend(out_png: Path, out_pdf: Path) -> None:
    """Generate standalone legend for Figure 4B."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    
    fig, ax = plt.subplots(figsize=(4, 2), facecolor='white')
    
    legend_elements = [
        Patch(facecolor='#d94728', alpha=0.8, label='LUSC-enriched'),
        Patch(facecolor='#3b5387', alpha=0.8, label='LUAD-enriched'),
    ]
    
    ax.legend(
        handles=legend_elements,
        loc='center',
        frameon=False,
        fontsize=12,
        title="Pathway Enrichment",
        title_fontsize=14,
        labelspacing=1.0
    )
    
    ax.axis('off')
    fig.savefig(out_png, dpi=600, bbox_inches='tight', transparent=False, facecolor='white')
    fig.savefig(out_pdf, bbox_inches='tight', transparent=False, facecolor='white')
    plt.close(fig)


def compute_gene_contributions_lung(
    expr: pd.DataFrame,
    pathway_scores: pd.DataFrame,
    target_pathway: str,
    gmt_path: Path,
    bootstrap_n: int = 0,
) -> pd.DataFrame:
    """
    Compute gene contributions with optional bootstrap.
    Returns DataFrame with ['gene', 'contribution', 'contribution_std'].
    """
    from core.pathway_activity_r_gsva import parse_gmt
    
    common_idx = expr.index.intersection(pathway_scores.index)
    if len(common_idx) < 5: return pd.DataFrame()
    
    expr_aligned = expr.loc[common_idx]
    act_aligned = pathway_scores.loc[common_idx, target_pathway]
    
    # GMT matching
    gene_sets = parse_gmt(gmt_path)
    def norm(n): return n.upper().replace(' ', '_').replace('-', '_')
    t_norm = norm(target_pathway)
    pathway_genes = next((set(gs) for pn, gs in gene_sets.items() if norm(pn) == t_norm), set())
    available_genes = list(pathway_genes.intersection(expr_aligned.columns)) if pathway_genes else list(expr_aligned.columns)
    
    def calc_corrs_lung(e, a):
        return {g: abs(np.corrcoef(e[g].values, a.values)[0,1]) for g in available_genes if not np.isnan(np.corrcoef(e[g].values, a.values)[0,1])}

    base_corrs = calc_corrs_lung(expr_aligned, act_aligned)
    res = pd.DataFrame([{'gene': g, 'contribution': v, 'contribution_std': 0.0} for g, v in base_corrs.items()])
    
    if bootstrap_n > 0 and not res.empty:
        boot_res = []
        for i in range(bootstrap_n):
            idx = np.random.choice(common_idx, size=len(common_idx), replace=True)
            boot_res.append(calc_corrs_lung(expr_aligned.loc[idx], act_aligned.loc[idx]))
        boot_df = pd.DataFrame(boot_res)
        res['contribution_std'] = res['gene'].map(boot_df.std()).fillna(0)
        
    return res.sort_values('contribution', ascending=False)


def plot_gene_contributions_lung(
    contributions: pd.DataFrame,
    out_png: Path,
    out_pdf: Path,
    title: str,
    pathway_name: Optional[str] = None,
    top_n: int = 10,
) -> None:
    """
    Plot top contributing genes with blurred uncertainty and purple gradient.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    if contributions.empty:
        return
        
    top_genes = contributions.head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(6, len(top_genes) * 0.5)), dpi=600)
    y_pos = np.arange(len(top_genes))
    
    # Premium Purple gradient
    colors = cm.Purples(np.linspace(0.8, 0.4, len(top_genes))) 
    bar_height = 0.7
    ax.barh(y_pos, top_genes['contribution'], color=colors, alpha=0.8, edgecolor='none', zorder=3)
    
    # Blurred Error and Dashed Mean
    means = top_genes['contribution'].values
    stds = top_genes['contribution_std'].values if 'contribution_std' in top_genes.columns else [0]*len(top_genes)
    
    for i, (m, s) in enumerate(zip(means, stds)):
        if s > 0:
            ax.fill_betweenx([y_pos[i] - bar_height/2, y_pos[i] + bar_height/2], 
                             m - s, m + s, color=colors[i], alpha=0.35, zorder=2)
        ax.vlines(m, y_pos[i] - bar_height/2, y_pos[i] + bar_height/2, 
                  colors='#111111', linestyles='--', linewidth=1.5, alpha=0.9, zorder=4)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_genes['gene'], fontweight='bold', fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Contribution Score (|Correlation|)', fontsize=11, fontweight='bold')
    
    full_title = title
    if pathway_name:
        p_name = pathway_name.replace('KEGG_', '').replace('_', ' ')
        full_title += f"\nTarget Pathway: {p_name}"
    ax.set_title(full_title, fontsize=14, fontweight='bold', pad=20)
    
    ax.grid(axis='x', linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    
    for i, v in enumerate(top_genes['contribution'].values):
        ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9, fontweight='bold', color='#2c3e50')
    
    fig.tight_layout()
    fig.savefig(out_png, dpi=600, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    
    logging.info(f"Saved gene contributions plot to {out_png}")


def _create_combined_panel(
    out_dir: Path,
    dataset: str,
    gene_set: str,
    dim: int,
) -> None:
    """
    Create a combined 3-panel figure.
    Layout: Left (Panel A), Top-Right (Panel B), Bottom-Right (Panel C)
    """
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from matplotlib import gridspec
    
    dataset_key = dataset.lower()
    fig_files = {
        'A': out_dir / f"Figure4A_visualization_{dataset_key}_{gene_set}_dim{dim}.png",
        'B': out_dir / f"Figure4B_discovery_{dataset_key}_{gene_set}_dim{dim}.png",
        'C': out_dir / f"Figure4C_mechanism_{dataset_key}_{gene_set}_dim{dim}.png",
        'D': out_dir / f"Figure4D_activity_dist_{dataset_key}_{gene_set}_dim{dim}.png",
    }
    
    # Check which files exist
    existing_files = {k: v for k, v in fig_files.items() if v.exists()}
    
    if len(existing_files) < 4:
        logging.warning(f"[{dataset}] Not all panels generated (A,B,C,D) to create combined figure. Found: {list(existing_files.keys())}")
        if len(existing_files) == 0:
            return
        # Fallback to simple layout
        fig = plt.figure(figsize=(6 * len(existing_files), 6), dpi=600)
        gs = gridspec.GridSpec(1, len(existing_files), figure=fig, hspace=0.3, wspace=0.3)
        panel_keys = sorted(existing_files.keys())
        positions = {panel_keys[i]: gs[0, i] for i in range(len(existing_files))}
        title_prefix = "Figure 4: Partial Analysis"
    else:
        # Layout: 2 rows x 3 columns
        # Left (Col 0): Panel A (t-SNE) spans 2 rows
        # Top Row Right: Panel B (Diverging bar chart) spans Col 1-2
        # Bottom Middle (Col 1): Panel D (Distribution Boxplot)
        # Bottom Right (Col 2): Panel C (Gene Contribution)
        fig = plt.figure(figsize=(20, 12), dpi=600)
        gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)
        
        positions = {
            'A': gs[:, 0],  
            'B': gs[0, 1:], 
            'D': gs[1, 1],  
            'C': gs[1, 2],  
        }
        title_prefix = "Figure 4: Generalization to Lung Cancer"
    
    # Load and place images
    for panel_key, img_path in existing_files.items():
        if panel_key in positions:
            try:
                img = mpimg.imread(str(img_path))
                ax = fig.add_subplot(positions[panel_key])
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(f"Panel {panel_key}", fontsize=12, fontweight='bold')
            except Exception as e:
                logging.warning(f"Could not load panel {panel_key} from {img_path}: {e}")
    
    display_name = "SCAN-B" if dataset.lower() == "scanb" else dataset
    display_gene_set = gene_set.upper()
    fig.suptitle(f"{title_prefix} - {display_name} ({display_gene_set}, dim{dim})",
                 fontsize=16, fontweight='bold', y=0.98)
    
    out_png = out_dir / f"Figure4_panel_{dataset_key}_{gene_set}_dim{dim}.png"
    out_pdf = out_dir / f"Figure4_panel_{dataset_key}_{gene_set}_dim{dim}.pdf"
    
    fig.savefig(out_png, dpi=600, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    
    logging.info(f"[{dataset}] Combined panel saved: {out_png}")


def run_figure4_analysis(
    *,
    dataset: str,
    dim: int,
    seed: int,
    out_root: Path,
    gene_set: str,
    gmt_path: Optional[str],
    ae_run_dir: Optional[str],
    bulk_test_path: Optional[str],
    label_path: Optional[str],
    verbose: bool,
) -> None:
    """Run Figure 4 analysis for TCGA Lung dataset."""
    dataset_key = dataset.lower()
    display_name = "SCAN-B" if dataset.lower() == "scanb" else "TCGA Lung"
    display_gene_set = gene_set.upper()
    
    project_root = Path(__file__).resolve().parents[2]
    logging.info(f"[{display_name}] Starting Figure 4 analysis ({display_gene_set})...")
    if bulk_test_path is None:
        bulk_test_path = str(project_root / "data" / "TCGA_Pan_cancer" / "tcga_lung_test.csv")
    if label_path is None:
        label_path = str(project_root / "data" / "TCGA_Pan_cancer" / "tcga_lung_labels.csv")
    
    out_dir = out_root / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"[{dataset}] Starting Figure 4 analysis...")
    
    # Load data
    if ae_run_dir:
        ae_sel = RunDirSelection(run_dir=Path(ae_run_dir), test_embedding_csv=Path(ae_run_dir) / "test_embedding.csv")
    else:
        ae_sel = find_latest_ae_run_dir(dataset=dataset_key, dim=dim, result_root=Path(__file__).resolve().parents[2] / "results" / "gsea_outputs")
    
    logging.info(f"[{dataset}] AE run dir: {ae_sel.run_dir}")
    
    emb_ae = load_embedding_csv(ae_sel.test_embedding_csv)
    expr = load_expression_matrix(bulk_test_path, dataset=dataset_key, log_prefix="test")
    labels = load_lung_subtype_labels(label_path)
    
    # Align all data
    common_idx = emb_ae.index.intersection(expr.index).intersection(labels.index)
    if len(common_idx) < 10:
        raise ValueError(f"[{dataset}] Too few common samples: {len(common_idx)}")
    
    emb_ae = emb_ae.loc[common_idx]
    expr = expr.loc[common_idx]
    labels = labels.loc[common_idx]
    
    logging.info(f"[{dataset}] Aligned samples: {len(common_idx)}")
    
    # Resolve GMT and load NES
    gmt_path_resolved = _resolve_gmt(gmt_path, gene_set=gene_set)
    logging.info(f"[{dataset}] Using GMT: {gmt_path_resolved}")
    
    nes_dirs = _default_nes_source_dirs(dataset_key=dataset_key, dim=dim, gene_set=gene_set)
    nes_ae = _load_or_build_nes(
        out_dir=out_dir,
        dataset_key=dataset_key,
        dim=dim,
        method="ae",
        source_dir=nes_dirs["ae"],
        gene_set=gene_set,
    )
    
    # Compute pathway activity
    A_ae = _compute_activity(emb_ae, nes_ae)
    
    # Panel A: t-SNE (Removed per request)
    # logging.info(f"[{dataset}] Panel A: Computing t-SNE visualization...")
    # plot_tsne_subtype_separation(
    #     embeddings=A_ae,
    #     labels=labels,
    #     out_png=out_dir / f"Figure4A_visualization_{dataset_key}_{gene_set}_dim{dim}.png",
    #     out_pdf=out_dir / f"Figure4A_visualization_{dataset_key}_{gene_set}_dim{dim}.pdf",
    #     title="T-SNE visualization of TCGA Lung pathway activity"
    # )
    
    # Standalone legend for Panel A (Also removed)
    # subtype_counts = labels.value_counts().to_dict()
    # _plot_shared_legend_lung(
    #     out_png=out_dir / f"Figure4A_legend_{dataset_key}.png",
    #     out_pdf=out_dir / f"Figure4A_legend_{dataset_key}.pdf",
    #     subtype_counts=subtype_counts
    # )
    
    # Panel B: Differential pathway discovery
    logging.info(f"[{dataset}] Panel B: Computing differential pathways (with bootstrap=50)...")
    diff_stats = compute_pathway_differential_statistics_lung(A_ae, labels, bootstrap_n=50)
    
    # --- Global Blacklist Filtering (Remove irrelevant pathways from all panels) ---
    blacklist = ["VASOPRESSIN", "WATER_REABSORPTION", "RENIN", "PROXIMAL_TUBULE", "COLLECTING_DUCT"]
    if not diff_stats.empty:
        mask = ~diff_stats['pathway'].str.upper().str.contains('|'.join(blacklist))
        diff_stats = diff_stats[mask].copy()
    
    plot_differential_pathways_diverging_lung(
        diff_stats=diff_stats,
        out_png=out_dir / f"Figure4B_discovery_{dataset_key}_{gene_set}_dim{dim}.png",
        out_pdf=out_dir / f"Figure4B_discovery_{dataset_key}_{gene_set}_dim{dim}.pdf",
        title="Differential Pathway Enrichment (TCGA Lung)",
        top_n=15,
        show_legend=False
    )
    # Save standalone legend for Figure 4B
    _plot_figure4b_legend(
        out_png=out_dir / f"Figure4B_legend_{dataset_key}.png",
        out_pdf=out_dir / f"Figure4B_legend_{dataset_key}.pdf"
    )

    # Panel D: Activity Distribution for Representative Pathways
    logging.info(f"[{dataset}] Panel D: Activity distribution for representative pathways...")
    
    if not diff_stats.empty:
        # Bio-relevant preference list
        preferred = ["CELL_CYCLE", "DNA_REPLICATION", "NON_SMALL_CELL_LUNG", "EGFR", "ERBB", "P53", "MYC", "GAP_JUNCTION"]
        
        def pick_representative(df_subset):
            # Check if preferred exists in top 5
            top_set = df_subset.head(5)
            for p_pref in preferred:
                match = top_set[top_set['pathway'].str.upper().str.contains(p_pref)]
                if not match.empty:
                    return match['pathway'].iloc[0]
            # Fallback to top 1
            return df_subset['pathway'].iloc[0] if not df_subset.empty else None

        pos_df = diff_stats[diff_stats['t_statistic'] > 0].sort_values('t_statistic', ascending=False)
        neg_df = diff_stats[diff_stats['t_statistic'] < 0].sort_values('t_statistic', ascending=True)

        target_pathways = []
        p1 = pick_representative(pos_df)
        if p1: target_pathways.append(p1)
        p2 = pick_representative(neg_df)
        if p2: target_pathways.append(p2)
        
        if target_pathways:
            plot_selected_pathways_distribution(
                pathway_scores=A_ae,
                labels=labels,
                pathways=target_pathways,
                out_png=out_dir / f"Figure4D_activity_dist_{dataset_key}_{gene_set}_dim{dim}.png",
                out_pdf=out_dir / f"Figure4D_activity_dist_{dataset_key}_{gene_set}_dim{dim}.pdf",
                title="Activity of Representative Differential Pathways"
            )
    else:
        logging.warning("No differential stats found, skipping Panel D.")

    # Panel C: Gene Contribution -> SKIPPED per user request ("Gene plot can be deleted")
    
    # Combined Panel (Updated to exclude C)
    _create_combined_panel(out_dir, dataset_key, gene_set, dim)
    
    logging.info(f"[{dataset}] Figure 4 analysis completed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Figure 4: Generalization to Lung Cancer")
    parser.add_argument("--dataset", type=str, default="TCGA_Lung")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--gene_set", type=str, default="KEGG")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_root", type=str, default=str(Path(__file__).resolve().parents[2] / "results" / "figure4" / "figure4_outputs_lung_cancer"))
    
    parser.add_argument("--gmt_path", type=str, default=None)
    parser.add_argument("--ae_run_dir", type=str, default=None)
    parser.add_argument("--bulk_test_path", type=str, default=None)
    parser.add_argument("--label_path", type=str, default=None)
    
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    _setup_logging(args.verbose)
    
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    
    # Set plotting style
    try:
        plt.style.use(['science', 'nature', 'no-latex'])
    except Exception:
        logging.warning("scienceplots style not available, using default")
        
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 18
    })
    
    run_figure4_analysis(
        dataset=args.dataset,
        dim=args.dim,
        seed=args.seed,
        out_root=out_root,
        gene_set=args.gene_set,
        gmt_path=args.gmt_path,
        ae_run_dir=args.ae_run_dir,
        bulk_test_path=args.bulk_test_path,
        label_path=args.label_path,
        verbose=args.verbose,
    )
    
    logging.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

