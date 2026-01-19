#!/usr/bin/env python
from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
from core.model_structure import Encoder

import argparse
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
try:
    import scienceplots
except ImportError:
    pass

from scripts.figure3.figure3_breast_cancer_clustering import (
    _compute_activity,
    _dataset_defaults,
    _default_nes_source_dirs,
    _load_or_build_nes,
    _resolve_gmt,
    load_expression_matrix,
    load_labels_with_pam50,
)
from core.gene_mapping import convert_ensembl_to_symbol, is_ensembl_id
from core.io_utils import (
    PAM50_CANONICAL,
    RunDirSelection,
    find_latest_ae_run_dir,
    load_embedding_csv,
)
from core.nes_from_gsea_reports import (
    build_nes_matrix_from_gsea_results_dir,
    build_nes_matrix_from_scanb_dim_dirs,
)
from core.pathway_activity_r_gsva import parse_gmt, union_genes
from scipy.stats import ttest_ind, mannwhitneyu


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def load_survival_data(dataset: str, label_path: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Load survival data for the dataset.
    Returns DataFrame with columns: [time, event] indexed by sample IDs.
    Returns None if survival data is not available.
    """
    dataset_key = dataset.lower()
    
    # Define survival data paths
    if dataset_key == "metabric":
        survival_path = Path("../cancer_brca_metabric_bulk_data/survival.csv")
    elif dataset_key == "scanb":
        # ScanB survival data is in the label file, use OS_days and OS_event
        if label_path:
            label_df = pd.read_csv(label_path, index_col=0)
            label_df.index = label_df.index.astype(str)
            label_df.sort_index(inplace=True)
            
            # Look for OS_days and OS_event columns
            time_col = None
            event_col = None
            
            # Try exact match first
            if 'OS_days' in label_df.columns:
                time_col = 'OS_days'
            elif 'OS_days' in [str(c) for c in label_df.columns]:
                time_col = [c for c in label_df.columns if str(c) == 'OS_days'][0]
            
            if 'OS_event' in label_df.columns:
                event_col = 'OS_event'
            elif 'OS_event' in [str(c) for c in label_df.columns]:
                event_col = [c for c in label_df.columns if str(c) == 'OS_event'][0]
            
            # If not found, try case-insensitive search
            if time_col is None or event_col is None:
                cols_lower = {str(c).lower(): c for c in label_df.columns}
                if 'os_days' in cols_lower:
                    time_col = cols_lower['os_days']
                if 'os_event' in cols_lower:
                    event_col = cols_lower['os_event']
            
            if time_col and event_col:
                survival_df = pd.DataFrame({
                    'time': pd.to_numeric(label_df[time_col], errors='coerce'),
                    'event': pd.to_numeric(label_df[event_col], errors='coerce')
                })
                survival_df = survival_df.dropna()
                if len(survival_df) > 0:
                    logging.info(f"[{dataset}] Loaded survival data from label file: {len(survival_df)} samples")
                    return survival_df
                else:
                    logging.warning(f"[{dataset}] No valid survival data after processing OS_days/OS_event")
            else:
                logging.warning(f"[{dataset}] Could not find OS_days/OS_event columns. Available: {list(label_df.columns)[:10]}")
        return None
    else:
        return None
    
    if not survival_path.exists():
        logging.warning(f"[{dataset}] Survival data not found at {survival_path}")
        return None
    
    try:
        survival_df = pd.read_csv(survival_path, index_col=0)
        survival_df.index = survival_df.index.astype(str)
        survival_df.sort_index(inplace=True)
        
        # Try to detect time and event columns
        cols_lower = {str(c).lower(): c for c in survival_df.columns}
        time_col = None
        event_col = None
        
        # Common column names
        for time_key in ['time', 'os_time', 'dfs_time', 'survival_time', 'overall_survival', 'os']:
            if time_key in cols_lower:
                time_col = cols_lower[time_key]
                break
        
        for event_key in ['event', 'os_event', 'dfs_event', 'status', 'death', 'vital_status']:
            if event_key in cols_lower:
                event_col = cols_lower[event_key]
                break
        
        if time_col is None or event_col is None:
            logging.warning(f"[{dataset}] Could not detect time/event columns in survival data. Available columns: {list(survival_df.columns)}")
            return None
        
        result = pd.DataFrame({
            'time': pd.to_numeric(survival_df[time_col], errors='coerce'),
            'event': pd.to_numeric(survival_df[event_col], errors='coerce')
        })
        result = result.dropna()
        
        if len(result) == 0:
            logging.warning(f"[{dataset}] No valid survival data after processing")
            return None
        
        logging.info(f"[{dataset}] Loaded survival data: {len(result)} samples")
        return result
        
    except Exception as e:
        logging.warning(f"[{dataset}] Failed to load survival data: {e}")
        return None


def load_clinical_features(label_path: str | Path, dataset: str) -> pd.DataFrame:
    """
    Load clinical features from label file.
    Returns DataFrame with all available clinical columns.
    """
    label_path = Path(label_path)
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")
    
    df = pd.read_csv(label_path, index_col=0)
    df.index = df.index.astype(str)
    df.sort_index(inplace=True)
    
    # Remove PAM50 column (we'll use it separately)
    pam50_cols = [c for c in df.columns if 'pam50' in str(c).lower() or 'subtype' in str(c).lower()]
    clinical_df = df.drop(columns=pam50_cols, errors='ignore')
    
    # Convert numeric columns
    for col in clinical_df.columns:
        try:
            clinical_df[col] = pd.to_numeric(clinical_df[col], errors='ignore')
        except Exception:
            pass
    
    logging.info(f"[{dataset}] Loaded clinical features: {list(clinical_df.columns)}")
    return clinical_df


def plot_kaplan_meier(
    survival_df: pd.DataFrame,
    groups: pd.Series,
    out_png: Path,
    out_pdf: Path,
    title: str,
    xlabel: str = "Time (days)",
    ylabel: str = "Survival probability",
) -> Optional[float]:
    """
    Plot Kaplan-Meier survival curves.
    
    Args:
        survival_df: DataFrame with 'time' and 'event' columns
        groups: Series with group labels for each sample
        out_png, out_pdf: Output paths
        title: Plot title
    
    Returns:
        Log-rank test p-value, or None if test cannot be performed
    """
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test
    except ImportError:
        logging.error("lifelines package is required for survival analysis. Install with: pip install lifelines")
        return None
    
    import matplotlib.pyplot as plt
    
    # Align data
    common_idx = survival_df.index.intersection(groups.index)
    if len(common_idx) == 0:
        logging.warning("No common samples between survival data and groups")
        return None
    
    survival_aligned = survival_df.loc[common_idx]
    groups_aligned = groups.loc[common_idx]
    
    # Remove missing groups
    valid_mask = groups_aligned.notna()
    survival_aligned = survival_aligned[valid_mask]
    groups_aligned = groups_aligned[valid_mask]
    
    if len(survival_aligned) == 0:
        logging.warning("No valid samples after alignment")
        return None
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
    
    # Plot KM curves for each group
    unique_groups = sorted(groups_aligned.unique())
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_groups)))
    
    kmf = KaplanMeierFitter()
    for i, group in enumerate(unique_groups):
        mask = groups_aligned == group
        group_data = survival_aligned[mask]
        
        if len(group_data) == 0:
            continue
        
        kmf.fit(
            group_data['time'],
            group_data['event'],
            label=f"{group} (n={len(group_data)})"
        )
        kmf.plot_survival_function(ax=ax, color=colors[i])
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Perform log-rank test if we have at least 2 groups
    p_value = None
    if len(unique_groups) >= 2:
        try:
            # Prepare data for log-rank test
            # Group data by group label
            group_data_dict = {}
            for group in unique_groups:
                mask = groups_aligned == group
                group_data = survival_aligned[mask]
                if len(group_data) > 0:
                    group_data_dict[str(group)] = {
                        'durations': group_data['time'].values,
                        'events': group_data['event'].values
                    }
            
            if len(group_data_dict) >= 2:
                # For multiple groups, use multivariate_logrank_test
                # For 2 groups, use simple logrank_test
                if len(group_data_dict) == 2:
                    group_names = list(group_data_dict.keys())
                    durations_A = group_data_dict[group_names[0]]['durations']
                    events_A = group_data_dict[group_names[0]]['events']
                    durations_B = group_data_dict[group_names[1]]['durations']
                    events_B = group_data_dict[group_names[1]]['events']
                    
                    results = logrank_test(durations_A, durations_B, event_observed_A=events_A, event_observed_B=events_B)
                    p_value = results.p_value
                else:
                    # Multiple groups: use multivariate_logrank_test
                    from lifelines.statistics import multivariate_logrank_test
                    
                    # Prepare data as arrays
                    all_durations = []
                    all_events = []
                    all_groups = []
                    
                    for group_name, data in group_data_dict.items():
                        all_durations.extend(data['durations'].tolist())
                        all_events.extend(data['events'].tolist())
                        all_groups.extend([group_name] * len(data['durations']))
                    
                    results = multivariate_logrank_test(
                        np.array(all_durations),
                        np.array(all_groups),
                        np.array(all_events)
                    )
                    p_value = results.p_value
                
                # Add p-value to plot
                if p_value is not None:
                    p_text = f"Log-rank test p = {p_value:.4f}"
                    if p_value < 0.001:
                        p_text = f"Log-rank test p < 0.001"
                    elif p_value < 0.01:
                        p_text = f"Log-rank test p = {p_value:.3f}"
                    elif p_value < 0.05:
                        p_text = f"Log-rank test p = {p_value:.3f}"
                    else:
                        p_text = f"Log-rank test p = {p_value:.3f}"
                    
                    ax.text(0.02, 0.02, p_text, 
                           transform=ax.transAxes, fontsize=11, fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        except Exception as e:
            logging.warning(f"Could not perform log-rank test: {e}")
    
    fig.tight_layout()
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)
    
    return p_value


def _format_pathway_name(name: str) -> str:
    """Format pathway name to Sentence case, preserving acronyms."""
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

def _format_pvalue(p_val: float) -> str:
    """Format P-value."""
    if p_val < 0.0001:
        return "P $<$ 0.0001"
    else:
        return f"P = {p_val:.4f}"


def compute_pathway_differential_statistics(
    pathway_scores: pd.DataFrame,
    pam50: pd.Series,
    bootstrap_n: int = 0,
) -> pd.DataFrame:
    """
    Compute differential statistics for pathways between Basal and LumA.
    Includes bootstrap resampling if bootstrap_n > 0.
    
    Args:
        pathway_scores: DataFrame samples x pathways
        pam50: Series with PAM50 labels
        bootstrap_n: Number of bootstrap samples to perform for error estimation.
                     If 0, no bootstrapping is performed.
    
    Returns:
        DataFrame with columns [pathway, t_statistic, p_value, direction, abs_t_statistic, t_std]
        direction: 'Basal' (positive t_statistic) or 'LumA' (negative t_statistic)
        t_std: Standard deviation of t-statistic from bootstrapping (0 if no bootstrap)
    """
    # Align data
    common = pathway_scores.index.intersection(pam50.index)
    pathway_aligned = pathway_scores.loc[common]
    pam50_aligned = pam50.loc[common]
    
    # Filter for Basal and LumA
    basal_mask = pam50_aligned == 'Basal'
    luma_mask = pam50_aligned == 'LumA'
    
    basal_scores = pathway_aligned[basal_mask]
    luma_scores = pathway_aligned[luma_mask]
    
    if len(basal_scores) < 2 or len(luma_scores) < 2:
        logging.warning("Not enough Basal or LumA samples for differential analysis.")
        return pd.DataFrame()

    def calc_t_stats(b_df, l_df):
        from scipy.stats import ttest_ind
        stats = {}
        for pathway in b_df.columns:
            b_vals = b_df[pathway].dropna().values
            l_vals = l_df[pathway].dropna().values
            if len(b_vals) < 2 or len(l_vals) < 2:
                stats[pathway] = (0, 1) # Cannot compute t-test
                continue
            try:
                t, p = ttest_ind(b_vals, l_vals, equal_var=False)
                stats[pathway] = (t, p)
            except Exception:
                stats[pathway] = (0, 1) # Fallback if t-test fails
        return stats

    # Initial stats
    base_stats = calc_t_stats(basal_scores, luma_scores)
    
    results = []
    for pathway, (t, p) in base_stats.items():
        results.append({
            'pathway': pathway,
            't_statistic': t,
            'p_value': p,
            'direction': 'Basal' if t > 0 else 'LumA',
            'abs_t_statistic': abs(t),
            't_std': 0.0 # Initialize t_std
        })
    
    df = pd.DataFrame(results)
    
    # Bootstrap for error bars
    if bootstrap_n > 0 and not df.empty:
        logging.info(f"Running bootstrap (n={bootstrap_n}) for pathway stats...")
        all_boot_t = []
        # Use different random states for basal and luma to ensure independent sampling
        for i in range(bootstrap_n):
            b_boot = basal_scores.sample(frac=1.0, replace=True, random_state=i)
            l_boot = luma_scores.sample(frac=1.0, replace=True, random_state=i + 1000)
            boot_stats = calc_t_stats(b_boot, l_boot)
            all_boot_t.append({p: s[0] for p, s in boot_stats.items() if p in df['pathway'].values})
        
        if all_boot_t:
            boot_df = pd.DataFrame(all_boot_t)
            # Ensure all pathways from original df are in boot_df, fill missing with NaN
            boot_df = boot_df.reindex(columns=df['pathway'].values)
            
            t_std = boot_df.std()
            
            # Update df with t_std
            df['t_std'] = df['pathway'].map(t_std).fillna(0)
        else:
            logging.warning("No valid bootstrap samples generated.")
    
    # Calculate FDR (Benjamini-Hochberg)
    from statsmodels.stats.multitest import multipletests
    p_vals = df['p_value'].values
    p_vals = np.clip(p_vals, 0, 1) # Ensure p-values are within [0, 1]
    if len(p_vals) > 0:
        _, fdr, _, _ = multipletests(p_vals, method='fdr_bh')
        df['fdr'] = fdr
    else:
        df['fdr'] = [] # If no pathways, fdr column will be empty
        
    return df.sort_values('abs_t_statistic', ascending=False)


def select_top_differential_pathways(
    pathway_scores: pd.DataFrame,
    pam50: pd.Series,
    n_pathways: int = 2,
    preferred_pathways: Optional[list[str]] = None,
) -> list[str]:
    """
    Select top N differential pathways between Basal and LumA.
    Prioritizes breast cancer-relevant pathways if specified.
    
    Args:
        pathway_scores: DataFrame samples x pathways
        pam50: Series with PAM50 labels
        n_pathways: Number of pathways to select (default: 2)
        preferred_pathways: List of preferred pathway names (e.g., breast cancer-relevant pathways)
                          If provided, will try to select from these first, then fill with top differential
    
    Returns:
        List of pathway names (top N by absolute T-statistic, prioritizing preferred pathways)
    """
    diff_stats = compute_pathway_differential_statistics(pathway_scores, pam50)
    if len(diff_stats) == 0:
        logging.warning("No differential pathways found")
        return []
    
    selected = []
    
    # If preferred pathways are specified, try to find them in diff_stats
    if preferred_pathways:
        available_pathways = set(pathway_scores.columns)
        for pref_pathway in preferred_pathways:
            # Try exact match
            if pref_pathway in diff_stats['pathway'].values:
                selected.append(pref_pathway)
                if len(selected) >= n_pathways:
                    break
            else:
                # Try fuzzy match (case-insensitive, partial)
                pref_lower = pref_pathway.lower()
                for pathway in diff_stats['pathway'].values:
                    if pref_lower in pathway.lower() or pathway.lower() in pref_lower:
                        if pathway not in selected:
                            selected.append(pathway)
                            if len(selected) >= n_pathways:
                                break
                if len(selected) >= n_pathways:
                    break
    
    # Fill remaining slots with top differential pathways
    if len(selected) < n_pathways:
        for pathway in diff_stats['pathway'].values:
            if pathway not in selected:
                selected.append(pathway)
                if len(selected) >= n_pathways:
                    break
    
    return selected[:n_pathways]


def plot_subtype_pathway_activity(
    pathway_scores: pd.DataFrame,
    pam50: pd.Series,
    pathways: list[str],
    out_png: Path,
    out_pdf: Path,
    title: str,
) -> None:
    """
    Plot boxplots showing pathway activity across PAM50 subtypes.
    Specifically designed for 2 pathways with P-value annotation between Basal and LumA.
    
    Args:
        pathway_scores: DataFrame samples x pathways
        pam50: Series with PAM50 labels
        pathways: List of pathway names to plot (should be 2 pathways)
        out_png, out_pdf: Output paths
        title: Plot title
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Align data
    common_idx = pathway_scores.index.intersection(pam50.index)
    pathway_aligned = pathway_scores.loc[common_idx]
    pam50_aligned = pam50.loc[common_idx]
    
    # Remove missing PAM50
    valid_mask = pam50_aligned.notna()
    pathway_aligned = pathway_aligned[valid_mask]
    pam50_aligned = pam50_aligned[valid_mask]
    
    # Filter pathways that exist
    available_pathways = [p for p in pathways if p in pathway_aligned.columns]
    if not available_pathways:
        logging.warning(f"None of the requested pathways found in pathway scores. Available: {list(pathway_aligned.columns)[:10]}")
        return
    
    # Limit to 2 pathways
    if len(available_pathways) > 2:
        available_pathways = available_pathways[:2]
        logging.info(f"Limiting to first 2 pathways: {available_pathways}")
    elif len(available_pathways) < 2:
        logging.warning(f"Only {len(available_pathways)} pathway(s) available, expected 2")
        if len(available_pathways) == 0:
            return
    
    # Create figure with 2 subplots (vertical layout)
    if len(available_pathways) == 0:
        return

    # Loop through pathways and save separate plots
    for i, pathway in enumerate(available_pathways):
        # Prepare data
        plot_data = pd.DataFrame({
            'Pathway Activity': pathway_aligned[pathway],
            'PAM50 Subtype': pam50_aligned
        })
        plot_data = plot_data.dropna()
        
        subtype_order = [s for s in PAM50_CANONICAL if s in plot_data['PAM50 Subtype'].values]
        
        # Determine Safe filename suffix
        safe_name = pathway.replace('KEGG_', '').lower()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(3, 5), layout='constrained', dpi=600)
        
        # Boxplot
        palette = {
            "Basal": "#d94728",
            "LumA": "#3b5387",
            "LumB": "#039f89",
            "Her2": "#4eb9d3",
            "Normal": "#95a5a6",
        }
        sns.boxplot(
            data=plot_data,
            x='PAM50 Subtype',
            y='Pathway Activity',
            order=subtype_order,
            palette=palette,
            ax=ax,
            width=0.4,
            fliersize=0,
            linewidth=1.5
        )
        
        display_name = _format_pathway_name(pathway)
        ax.set_title(display_name, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel("", fontsize=1)
        ax.set_ylabel('Pathway Activity Score', fontsize=13, fontweight='bold')
        ax.tick_params(axis='x', rotation=45, labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        
        # Stats
        basal_values = plot_data[plot_data['PAM50 Subtype'] == 'Basal']['Pathway Activity'].dropna().values
        luma_values = plot_data[plot_data['PAM50 Subtype'] == 'LumA']['Pathway Activity'].dropna().values
        
        if len(basal_values) >= 3 and len(luma_values) >= 3:
            try:
                t_stat, p_val = ttest_ind(basal_values, luma_values, equal_var=False)
                p_text = _format_pvalue(p_val)
            except:
                p_text = ""
                
            basal_pos = subtype_order.index('Basal') if 'Basal' in subtype_order else -1
            luma_pos = subtype_order.index('LumA') if 'LumA' in subtype_order else -1
            
            if basal_pos != -1 and luma_pos != -1:
                y_min, y_max = ax.get_ylim()
                rng = y_max - y_min
                
                bracket_y = y_max + 0.02 * rng
                bracket_h = 0.02 * rng
                
                # Draw bracket
                ax.plot([basal_pos, basal_pos, luma_pos, luma_pos], 
                        [bracket_y, bracket_y+bracket_h, bracket_y+bracket_h, bracket_y], 
                        c='k', lw=1.5)
                
                ax.text((basal_pos + luma_pos)/2, bracket_y + 2.5*bracket_h, p_text,
                       ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')
                
                ax.set_ylim(y_min - 0.05*rng, y_max + 0.15*rng)

        # Save separate files
        this_png = out_png.parent / f"{out_png.stem}_{i+1}_{safe_name}.png"
        this_pdf = out_pdf.parent / f"{out_pdf.stem}_{i+1}_{safe_name}.pdf"
        
        fig.savefig(this_png, dpi=600)
        fig.savefig(this_pdf)
        plt.close(fig)
        logging.info(f"Saved boxplot for {pathway} to {this_png}")


def plot_differential_pathways_diverging(
    diff_stats: pd.DataFrame,
    out_png: Path,
    out_pdf: Path,
    title: str,
    top_n: int = 20,
) -> None:
    """
    Plot diverging horizontal bar chart showing top differential pathways.
    
    Args:
        diff_stats: DataFrame from compute_pathway_differential_statistics
        out_png, out_pdf: Output paths
        title: Plot title
        top_n: Number of top pathways to show (default: 15)
    """
    import matplotlib.pyplot as plt
    
    # Get top N pathways
    top_pathways = diff_stats.head(top_n).copy()
    
    if len(top_pathways) == 0:
        logging.warning("No pathways to plot")
        return
    
    # Format pathway names to Sentence case
    top_pathways['pathway_display'] = top_pathways['pathway'].apply(_format_pathway_name)
    
    # Separate positive (Basal) and negative (LumA) pathways
    basal_pathways = top_pathways[top_pathways['direction'] == 'Basal']
    luma_pathways = top_pathways[top_pathways['direction'] == 'LumA']
    
    # Create figure
    # Create figure
    # Ensure we use t-statistic for sorting
    plot_df = top_pathways.sort_values('t_statistic', ascending=False).copy()
    
    # Create figure - match Figure 4B size logic
    fig, ax = plt.subplots(figsize=(8, max(5, len(plot_df) * 0.4)), layout='constrained', dpi=600)
    
    y_pos = np.arange(len(plot_df))
    # Colors: Red for Basal (>0), Blue for LumA (<0)
    colors = ['#d94728' if t > 0 else '#3b5387' for t in plot_df['t_statistic']]
    
    # Main Bars
    ax.barh(y_pos, plot_df['t_statistic'], color=colors, alpha=0.75, zorder=3)
    
    # Blurred Error and Dashed Mean (Matching Figure 4 logic)
    bar_height = 0.8
    means = plot_df['t_statistic'].values
    stds = plot_df['t_std'].values if 't_std' in plot_df else [0]*len(plot_df)
    
    for i, (m, s) in enumerate(zip(means, stds)):
        if s > 0:
            ax.fill_betweenx([y_pos[i] - bar_height/2, y_pos[i] + bar_height/2], 
                             m - s, m + s, color=colors[i], alpha=0.35, zorder=2)
        ax.vlines(m, y_pos[i] - bar_height/2, y_pos[i] + bar_height/2, 
                  colors='#222222', linestyles='--', linewidth=1.5, alpha=0.9, zorder=4)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df['pathway_display'].values, fontsize=12, fontweight='bold')
    ax.set_xlabel('T-statistic (Basal vs LumA)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.axvline(0, color='grey', linestyle='-', linewidth=0.8, zorder=1)
    
    ax.grid(axis='x', linestyle='--', alpha=0.3, zorder=0)
    ax.invert_yaxis() # Top significant (Index 0) at top
    
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_pdf)
    plt.close(fig)
    
    # NEW: Generate standalone legend
    from matplotlib.patches import Patch
    legend_fig = plt.figure(figsize=(4, 2))
    # Create dummy handles for the legend
    legend_elements = [
        Patch(facecolor='#d94728', alpha=0.8, label='Basal-enriched'),
        Patch(facecolor='#3b5387', alpha=0.8, label='LumA-enriched')
    ]
    legend_fig.legend(handles=legend_elements, loc='center', fontsize=12, 
                      title="Pathway Enrichment", title_fontsize=14, labelspacing=1.0)
    plt.axis('off')
    
    legend_out_png = out_png.parent / out_png.name.replace('Figure5B_', 'Figure5B_legend_')
    legend_out_pdf = out_pdf.parent / out_pdf.name.replace('Figure5B_', 'Figure5B_legend_')
    
    legend_fig.savefig(legend_out_png, dpi=600, bbox_inches='tight')
    legend_fig.savefig(legend_out_pdf, bbox_inches='tight')
    plt.close(legend_fig)
    plt.close(fig)


def compute_gene_contributions(
    embedding: pd.DataFrame,
    nes: pd.DataFrame,
    pathway: str,
    expr: pd.DataFrame,
    gmt_path: Path,
    bootstrap_n: int = 0,
) -> pd.DataFrame:
    """
    Compute gene contributions with optional bootstrap resampling for error bars.
    Returns a DataFrame with ['gene', 'contribution', 'contribution_std'].
    """
    if pathway not in nes.columns:
        return pd.DataFrame()
    
    # Get pathway genes from GMT
    gene_sets = parse_gmt(gmt_path)
    
    def normalize_pathway_name(name: str) -> str:
        name = name.replace('KEGG_MEDICUS_REFERENCE_', 'KEGG_')
        name = name.replace('KEGG_MEDICUS_VARIANT_', 'KEGG_')
        return name
    
    # Matching logic
    if pathway in gene_sets:
        pathway_genes = gene_sets[pathway]
    else:
        normalized = normalize_pathway_name(pathway)
        if normalized in gene_sets:
            pathway_genes = gene_sets[normalized]
        else:
            pathway_lower = normalized.lower()
            matched_pathway = None
            for gmt_p in gene_sets.keys():
                if pathway_lower in gmt_p.lower() or gmt_p.lower() in pathway_lower:
                    matched_pathway = gmt_p
                    break
            if matched_pathway:
                pathway_genes = gene_sets[matched_pathway]
            else:
                logging.warning(f"Pathway {pathway} not found in GMT")
                return pd.DataFrame()

    # Align data
    common_dims = embedding.columns.intersection(nes.index)
    if common_dims.empty: return pd.DataFrame()
    
    Z = embedding[common_dims].values
    W = nes.loc[common_dims, pathway].values
    pathway_activity = pd.Series(Z @ W, index=embedding.index)
    
    common_samples = expr.index.intersection(embedding.index)
    if common_samples.empty: return pd.DataFrame()
    
    expr_aligned = expr.loc[common_samples]
    act_aligned = pathway_activity.loc[common_samples]
    
    available_genes = [g for g in pathway_genes if g in expr_aligned.columns]
    if not available_genes: return pd.DataFrame()

    def calc_corrs(e_df, a_ser):
        corrs = {}
        for gene in available_genes:
            c = e_df[gene].corr(a_ser)
            if not np.isnan(c):
                corrs[gene] = abs(c)
        return corrs

    # Base contributions
    base_corrs = calc_corrs(expr_aligned, act_aligned)
    res = pd.DataFrame([{'gene': g, 'contribution': v, 'contribution_std': 0.0} 
                       for g, v in base_corrs.items()])
    
    if bootstrap_n > 0 and not res.empty:
        logging.info(f"Running bootstrap (n={bootstrap_n}) for gene contributions...")
        boot_results = []
        for i in range(bootstrap_n):
            idx = np.random.choice(common_samples, size=len(common_samples), replace=True)
            e_boot = expr_aligned.loc[idx]
            a_boot = act_aligned.loc[idx]
            boot_corrs = calc_corrs(e_boot, a_boot)
            boot_results.append(boot_corrs)
        
        boot_df = pd.DataFrame(boot_results)
        # Ensure that the index of boot_df matches the 'gene' column of res for mapping
        boot_df_std = boot_df.std().reindex(res['gene']).fillna(0)
        res['contribution_std'] = boot_df_std.values
    
    return res.sort_values('contribution', ascending=False)


def plot_gene_contributions(
    contributions: pd.DataFrame,
    out_png: Path,
    out_pdf: Path,
    title: str,
    pathway_name: Optional[str] = None,
    top_n: int = 10,
) -> None:
    """
    Plot bar chart for gene contributions.
    Takes a DataFrame with ['gene', 'contribution', 'contribution_std'].
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    if contributions.empty:
        logging.warning("No contributions to plot")
        return
        
    top_genes = contributions.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(top_genes) * 0.5)), dpi=600)
    y_pos = np.arange(len(top_genes))
    
    # Premium color palette: Professional Purple/Violet gradient restored
    import matplotlib.cm as cm
    colors = cm.Purples(np.linspace(0.8, 0.4, len(top_genes))) 
    
    # Plot solid bars
    bar_height = 0.7
    bars = ax.barh(y_pos, top_genes['contribution'], color=colors, alpha=0.8,
                  edgecolor='none', linewidth=0.5, zorder=3)
    
    # Plot blurred error and dashed mean
    means = top_genes['contribution'].values
    stds = top_genes['contribution_std'].values if 'contribution_std' in top_genes.columns else [0]*len(top_genes)
    
    for i, (m, s) in enumerate(zip(means, stds)):
        if s > 0:
            # Darker error region (cloud/blur) to distinguish from the primary bar
            ax.fill_betweenx([y_pos[i] - bar_height/2, y_pos[i] + bar_height/2], 
                             m - s, m + s, color=colors[i], alpha=0.4, zorder=2)
        # Bold dashed line exactly at the mean value
        ax.vlines(m, y_pos[i] - bar_height/2, y_pos[i] + bar_height/2, 
                  colors='#111111', linestyles='--', linewidth=1.5, alpha=0.85, zorder=4)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_genes['gene'], fontsize=10, fontweight='bold')
    ax.invert_yaxis()
    
    ax.set_xlabel('Contribution Score (Pearson Correlation)', fontsize=12, fontweight='bold')
    
    full_title = title
    if pathway_name:
        pathway_display = pathway_name.replace('KEGG_', '').replace('_', ' ')
        full_title += f"\nTarget Pathway: {pathway_display}"
    ax.set_title(full_title, fontsize=14, fontweight='bold', pad=20)
    
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Add value labels with clean formatting
    for i, v in enumerate(top_genes['contribution'].values):
        ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9, fontweight='bold', color='#2c3e50')
    
    fig.tight_layout()
    fig.savefig(out_png, dpi=600, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)


def plot_clinical_correlation_heatmap(
    pathway_scores: pd.DataFrame,
    clinical_df: pd.DataFrame,
    pathways: list[str],
    out_png: Path,
    out_pdf: Path,
    title: str,
) -> None:
    """
    Plot heatmap of correlations between pathway activities and clinical features.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import spearmanr
    import numpy as np # Added import for np
    
    # Align data
    common_idx = pathway_scores.index.intersection(clinical_df.index)
    if len(common_idx) == 0:
        logging.warning("No common samples between pathway scores and clinical data")
        return
    
    pathway_aligned = pathway_scores.loc[common_idx]
    clinical_aligned = clinical_df.loc[common_idx]
    
    # Filter pathways
    available_pathways = [p for p in pathways if p in pathway_aligned.columns]
    if not available_pathways:
        logging.warning("No requested pathways found")
        return
    
    # Select numeric clinical features
    numeric_clinical = clinical_aligned.select_dtypes(include=[np.number])
    if len(numeric_clinical.columns) == 0:
        logging.warning("No numeric clinical features found")
        return
    
    # Compute correlations
    corr_matrix = np.zeros((len(available_pathways), len(numeric_clinical.columns)))
    
    for i, pathway in enumerate(available_pathways):
        for j, clinical_feat in enumerate(numeric_clinical.columns):
            pathway_vals = pathway_aligned[pathway].values
            clinical_vals = numeric_clinical[clinical_feat].values
            
            # Remove NaN
            valid_mask = ~(np.isnan(pathway_vals) | np.isnan(clinical_vals))
            if valid_mask.sum() < 3:  # Need at least 3 points
                corr_matrix[i, j] = np.nan
            else:
                corr, p_val = spearmanr(pathway_vals[valid_mask], clinical_vals[valid_mask])
                corr_matrix[i, j] = corr if not np.isnan(corr) else 0.0
    
    # Create DataFrame for plotting
    corr_df = pd.DataFrame(
        corr_matrix,
        index=available_pathways,
        columns=numeric_clinical.columns
    )
    
    # Plot
    fig, ax = plt.subplots(figsize=(max(8, len(numeric_clinical.columns) * 0.8), 
                                     max(6, len(available_pathways) * 0.4)), dpi=600)
    
    sns.heatmap(
        corr_df,
        annot=False,  # Don't show numbers in cells
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        cbar_kws={'label': 'Spearman Correlation'},
        linewidths=0.5,
        linecolor='gray'
    )
    
    ax.set_title(title)
    ax.set_xlabel('Clinical Features')
    ax.set_ylabel('Pathways')
    
    # Increase spacing between x-axis labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust x-axis tick spacing
    ax.set_xticks(range(len(numeric_clinical.columns)))
    ax.set_xticklabels(numeric_clinical.columns, rotation=45, ha='right')
    
    # Add more spacing between ticks
    ax.tick_params(axis='x', which='major', pad=10)
    
    fig.tight_layout()
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)


def run_one_dataset(
    *,
    dataset: str,
    dim: int,
    seed: int,
    out_root: Path,
    gene_set: str,
    gmt_path: Optional[str],
    pam50_col: Optional[str],
    ae_run_dir: Optional[str],
    bulk_test_path: Optional[str],
    bulk_train_path: Optional[str],
    label_path: Optional[str],
    panel_a_pathways: Optional[list[str]] = None,
    verbose: bool,
) -> None:
    """Run Figure 5 analysis for one dataset."""
    dataset_key = dataset.lower()
    defaults = _dataset_defaults(dataset_key)
    bulk_test_path = bulk_test_path or defaults["bulk_test_path"]
    label_path = label_path or defaults["label_path"]
    
    out_dir = out_root / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)
    
    display_name = "SCAN-B" if dataset.lower() == "scanb" else "TCGA-LUNG"
    display_gene_set = gene_set.upper()
    
    logging.info(f"[{display_name}] Starting Figure 5 analysis...")
    
    # Load data
    if ae_run_dir:
        ae_sel = RunDirSelection(run_dir=Path(ae_run_dir), test_embedding_csv=Path(ae_run_dir) / "test_embedding.csv")
    else:
        results_root = Path(__file__).resolve().parents[2] / "results" / "gsea_outputs"
        ae_sel = find_latest_ae_run_dir(dataset=dataset_key, dim=dim, result_root=results_root)
    
    logging.info(f"[{display_name}] AE run dir: {ae_sel.run_dir}")
    
    emb_ae = load_embedding_csv(ae_sel.test_embedding_csv)
    expr = load_expression_matrix(bulk_test_path, dataset=dataset_key, log_prefix="test", include_normal=True, seed=seed)
    
    # Load and combine labels for SCAN-B
    pam50 = load_labels_with_pam50(label_path=label_path, pam50_col=pam50_col)
    if dataset_key.lower() == "scanb":
        df_defaults = _dataset_defaults("scanb")
        if "train_label_path" in df_defaults:
            train_pam50 = load_labels_with_pam50(label_path=df_defaults["train_label_path"], pam50_col=pam50_col)
            pam50 = pd.concat([pam50, train_pam50])
            pam50 = pam50.loc[~pam50.index.duplicated()]
        
        # Explicitly label normal bulk samples as "Normal"
        norm_train_path = Path(df_defaults["bulk_normal_train_path"])
        norm_test_path = Path(df_defaults["bulk_normal_test_path"])
        if norm_train_path.exists() and norm_test_path.exists():
            norm_ids = pd.concat([pd.read_csv(norm_train_path, index_col=0), pd.read_csv(norm_test_path, index_col=0)]).index
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
                # Robust inference from state_dict
                sd = torch.load(encoder_path, map_location=device)
                import re
                weight_keys = sorted([k for k in sd.keys() if k.endswith(".weight") and "layers" in k and int(re.search(r'layers\.(\d+)', k).group(1)) % 3 == 0], 
                                     key=lambda x: int(re.search(r'layers\.(\d+)', x).group(1)))
                weights_shapes = [sd[k].shape for k in weight_keys]
                input_dim = weights_shapes[0][1]
                hidden_dims = [s[0] for s in weights_shapes[:-1]]
                out_dim = weights_shapes[-1][0]
                
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
                logging.warning(f"[{display_name}] Projecting missing samples failed: {e}")

    pam50 = pam50[pam50.notna()]
    clinical_df = load_clinical_features(label_path, dataset_key)
    
    # Align all data
    common_idx = emb_ae.index.intersection(expr.index).intersection(pam50.index)
    # Also intersect with clinical if possible, but keep numeric clinical separate if needed
    if len(common_idx) < 10:
        raise ValueError(f"[{display_name}] Too few common samples: {len(common_idx)}")
    
    emb_ae = emb_ae.loc[common_idx]
    expr = expr.loc[common_idx]
    pam50 = pam50.loc[common_idx]
    clinical_aligned = clinical_df.loc[clinical_df.index.intersection(common_idx)]
    
    logging.info(f"[{display_name}] Aligned samples: {len(common_idx)}")
    
    # Resolve GMT and load NES
    gmt_path_resolved = _resolve_gmt(gmt_path, gene_set=gene_set)
    logging.info(f"[{display_name}] Using GMT: {gmt_path_resolved}")
    
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
    
    # Panel A: Validation - Select top 2 differential pathways and plot boxplots
    # Initial stats for scaling/direction identification (using bootstrap now)
    diff_stats_all = compute_pathway_differential_statistics(A_ae, pam50, bootstrap_n=50)
    
    # Panel A: Boxplots
    # Prioritize canonical proliferative programs
    canonical_pathways = ['KEGG_CELL_CYCLE', 'KEGG_DNA_REPLICATION', 'KEGG_STEROID_BIOSYNTHESIS']
    
    # Panel A: Validation - logic moved to after Panel B usage to ensure consistency
    # (We will select the top pathways from the Discovery set)
    
    
    # Panel B: Discovery - Already computed in diff_stats_all (with bootstrap)
    logging.info(f"[{display_name}] Panel B: Preparing balanced canonical discovery panel...")
    
    # Define target canonical themes
    basal_keywords = [
        'CELL_CYCLE', 'DNA_REPLICATION', 'P53_SIGNALING',
        'MISMATCH_REPAIR', 'BASE_EXCISION_REPAIR', 'HOMOLOGOUS_RECOMBINATION',
        'FANCONI_ANEMIA', 'APOPTOSIS'
    ]
    luma_keywords = [
        'ESTROGEN_SIGNALING', 'PPAR_SIGNALING', 'FATTY_ACID_METABOLISM',
        'BUTANOATE_METABOLISM', 'PROPANOATE_METABOLISM', 'CITRATE_CYCLE',
        'ADIPOCYTOKINE_SIGNALING', 'PEROXISOME', 'GLYOXYLATE', 'PYRUVATE_METABOLISM',
        'VALINE_LEUCINE_ISOLEUCINE_DEGRADATION'
    ]
    all_keywords = basal_keywords + luma_keywords
    
    disease_keywords = [
        'CANCER', 'DISEASE', 'INFECTION', 'MALARIA', 'LEISHMANIA', 'TRYPANOSOMIASIS', 
        'SYPHILIS', 'HEPATITIS', 'MEASLES', 'INFLUENZA', 'TUBERCULOSIS', 'CARDIOMYOPATHY',
        'DIABETES', 'ALZHEIMER', 'PARKINSON', 'HUNTINGTON', 'AMYOTROPHIC', 'ASTHMA',
        'ARTHRITIS', 'AORTIC', 'VALVULAR', 'IMMUNODEFICIENCY', 'STAPHYLOCOCCUS', 'PNEUMONIA',
        'CHOLERA', 'AMEBIASIS', 'SHIGELLOSIS', 'LISTERIOSIS', 'TOXOPLASMOSIS', 'GLIOMA',
        'CARCINOMA', 'SARCOMA', 'MELANOMA', 'LEUKEMIA', 'LYMPHOMA'
    ]

    def is_valid_bio(name):
        name_u = name.upper().replace('-', '_').replace(' ', '_')
        # Filter out disease
        for dk in disease_keywords:
            if dk in name_u: return False
        return True

    def is_canonical(name):
        name_u = name.upper().replace('-', '_').replace(' ', '_')
        for kw in all_keywords:
            if kw in name_u: return True
        return False

    # Filter stats
    diff_stats_valid = diff_stats_all[diff_stats_all['pathway'].apply(is_valid_bio)].copy()
    
    # Select Top 6 Basal (T > 0) and Top 6 LumA (T < 0)
    # Prefer canonical pathways if they are significant
    basal_side = diff_stats_valid[diff_stats_valid['t_statistic'] > 0].sort_values('t_statistic', ascending=False)
    luma_side = diff_stats_valid[diff_stats_valid['t_statistic'] < 0].sort_values('t_statistic', ascending=True)
    
    def pick_balanced(df_side, n=6):
        # 1. Start with canonical pathways from this side
        canonical = df_side[df_side['pathway'].apply(is_canonical)]
        # 2. If those are enough, take them. Otherwise fill with non-canonical
        if len(canonical) >= n:
            return canonical.head(n)
        else:
            fillers = df_side[~df_side['pathway'].apply(is_canonical)]
            return pd.concat([canonical, fillers.head(n - len(canonical))])

    top_basal = pick_balanced(basal_side, n=6)
    top_luma = pick_balanced(luma_side, n=6)
    
    diff_stats_final = pd.concat([top_basal, top_luma]).sort_values('t_statistic', ascending=True)

    if len(diff_stats_final) > 0:
        plot_differential_pathways_diverging(
            diff_stats=diff_stats_final,
            out_png=out_dir / f"Figure5B_discovery_{dataset_key}_{gene_set}_dim{dim}.png",
            out_pdf=out_dir / f"Figure5B_discovery_{dataset_key}_{gene_set}_dim{dim}.pdf",
            title=f"Differential Pathway Enrichment ({display_name})",
            top_n=len(diff_stats_final)
        )
        logging.info(f"[{display_name}] Panel B: Plotted balanced {len(diff_stats_final)} pathways")
    else:
        logging.warning(f"[{display_name}] Panel B: No pathways found for plotting")
    
    # Update Panel A Selection to use the Top 1 Positive and Top 1 Negative from the Visualized set (diff_stats_final)
    # This ensures Figure 5A matches the top bars in Figure 5B
    if not diff_stats_final.empty:
        # diff_stats_final is sorted ascending.
        # Last element = Most Positive (Top of plot)
        # First element = Most Negative (Bottom of plot)
        top_pos_pathway = diff_stats_final.iloc[-1]['pathway']
        top_neg_pathway = diff_stats_final.iloc[0]['pathway']
        
        # Verify direction
        t_pos = diff_stats_final.iloc[-1]['t_statistic']
        t_neg = diff_stats_final.iloc[0]['t_statistic']
        
        selected_a = []
        if t_pos > 0:
            selected_a.append(top_pos_pathway)
        if t_neg < 0:
            selected_a.append(top_neg_pathway)
        
        # If we didn't get one of each (unlikely given selection logic), fallback to just top 2 absolute
        if len(selected_a) < 2:
             # Just take top 2 from final set
             selected_a = diff_stats_final.head(2)['pathway'].tolist()
             
        logging.info(f"[{dataset}] Panel A: Plotting boxplots for {selected_a} (Derived from Panel B)")
        
        plot_subtype_pathway_activity(
            pathway_scores=A_ae,
            pam50=pam50,
            pathways=selected_a,
            out_png=out_dir / f"Figure5A_validation_{dataset_key}_{gene_set}_dim{dim}.png",
            out_pdf=out_dir / f"Figure5A_validation_{dataset_key}_{gene_set}_dim{dim}.pdf",
            title=f"Top Pathways Analysis"
        )
    else:
        logging.warning(f"[{dataset}] Panel B empty, skipping Panel A update.")
    
    # Panel C: Mechanism - SKIPPED per user request ("Gene plot can be deleted")
    logging.info(f"[{dataset}] Panel C: Skipped per user request.")
    
    # Create combined panel figure (Skipped as we now have separate files for A)
    # _create_combined_panel(
    #     out_dir=out_dir,
    #     dataset=dataset,
    #     gene_set=gene_set,
    #     dim=dim,
    # )
    
    logging.info(f"[{dataset}] Figure 5 analysis complete. Outputs saved to {out_dir}")


def _create_combined_panel(
    out_dir: Path,
    dataset: str,
    gene_set: str,
    dim: int,
) -> None:
    """
    Create a combined 3-panel figure with new layout:
    - Left (1/3): Panel A (2 boxplots stacked vertically)
    - Top right (2/3): Panel B (diverging bar chart)
    - Bottom right (2/3): Panel C (gene contributions)
    """
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from matplotlib import gridspec
    
    dataset_key = dataset.lower()
    fig_files = {
        '5A': out_dir / f"Figure5A_validation_{dataset_key}_{gene_set}_dim{dim}.png",
        '5B': out_dir / f"Figure5B_discovery_{dataset_key}_{gene_set}_dim{dim}.png",
        '5C': out_dir / f"Figure5C_mechanism_{dataset_key}_{gene_set}_dim{dim}.png",
    }
    
    # Check which files exist
    # Check which files exist
    existing_files = {k: v for k, v in fig_files.items() if v.exists()}
    
    if len(existing_files) == 0:
        logging.warning("No panels found to create combined figure.")
        return
        
    # If we have A and B but not C (which is expected now), layout 1x2
    if '5A' in existing_files and '5B' in existing_files and '5C' not in existing_files:
        fig = plt.figure(figsize=(14, 8), dpi=600)
        gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1.2], wspace=0.3)
        
        # Panel A
        try:
            img_a = mpimg.imread(str(existing_files['5A']))
            ax_a = fig.add_subplot(gs[0])
            ax_a.imshow(img_a)
            ax_a.axis('off')
            ax_a.set_title("Panel A: Validation", fontsize=14, fontweight='bold', pad=10)
        except Exception as e:
            logging.warning(f"Could not load Panel 5A: {e}")
            
        # Panel B
        try:
            img_b = mpimg.imread(str(existing_files['5B']))
            ax_b = fig.add_subplot(gs[1])
            ax_b.imshow(img_b)
            ax_b.axis('off')
            ax_b.set_title("Panel B: Discovery (Differential)", fontsize=14, fontweight='bold', pad=10)
        except Exception as e:
            logging.warning(f"Could not load Panel 5B: {e}")
            
    elif len(existing_files) >= 3:
        # Original 2x2 layout if C exists
        fig = plt.figure(figsize=(18, 12), dpi=600)
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3, 
                              width_ratios=[1, 2], height_ratios=[1, 1])
        
        # Panel A: Left column, spans both rows
        if '5A' in existing_files:
            try:
                img_a = mpimg.imread(str(existing_files['5A']))
                ax_a = fig.add_subplot(gs[:, 0])
                ax_a.imshow(img_a)
                ax_a.axis('off')
                ax_a.set_title("Panel A: Validation", fontsize=14, fontweight='bold', pad=10)
            except Exception as e:
                logging.warning(f"Could not load Panel 5A: {e}")
        
        # Panel B: Top right
        if '5B' in existing_files:
            try:
                img_b = mpimg.imread(str(existing_files['5B']))
                ax_b = fig.add_subplot(gs[0, 1])
                ax_b.imshow(img_b)
                ax_b.axis('off')
                ax_b.set_title("Panel B: Discovery (Differential)", fontsize=14, fontweight='bold', pad=10)
            except Exception as e:
                logging.warning(f"Could not load Panel 5B: {e}")

        # Panel C: Bottom right
        if '5C' in existing_files:
            try:
                img_c = mpimg.imread(str(existing_files['5C']))
                ax_c = fig.add_subplot(gs[1, 1])
                ax_c.imshow(img_c)
                ax_c.axis('off')
                ax_c.set_title("Panel C: Mechanism (Top Contributing Genes)", fontsize=14, fontweight='bold', pad=10)
            except Exception as e:
                logging.warning(f"Could not load Panel 5C: {e}")
    else:
        logging.warning("Not enough panels for standard layouts. Skipping combined figure.")
        return

    # Panel C: Bottom right
    # Panel C: Skipped
    pass
    
    display_name = "SCAN-B" if dataset.lower() == "scanb" else "TCGA Lung"
    display_gene_set = gene_set.upper()
    fig.suptitle(f"Figure 5: Biological Mechanism & Interpretability - {display_name} ({display_gene_set}, dim{dim})", 
                 fontsize=16, fontweight='bold', y=0.98)
    
    out_png = out_dir / f"Figure5_panel_{dataset_key}_{gene_set}_dim{dim}.png"
    out_pdf = out_dir / f"Figure5_panel_{dataset_key}_{gene_set}_dim{dim}.pdf"
    
    fig.savefig(out_png, dpi=600, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    
    logging.info(f"[{dataset}] Combined panel saved: {out_png}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Figure 5: Biological Mechanism & Interpretability Analysis")
    parser.add_argument("--datasets", nargs="+", default=["scanb", "Metabric"])
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--gene_set", type=str, default="KEGG")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_root", type=str, default=str(Path(__file__).resolve().parents[2] / "results" / "figure5" / "figure5_outputs_clinical_relevance"))
    
    parser.add_argument("--gmt_path", type=str, default=None)
    parser.add_argument("--pam50_col", type=str, default=None)
    parser.add_argument("--ae_run_dir_scanb", type=str, default=None)
    parser.add_argument("--ae_run_dir_metabric", type=str, default=None)
    parser.add_argument("--scanb_test_path", type=str, default=None)
    parser.add_argument("--scanb_label_path", type=str, default=None)
    parser.add_argument("--metabric_test_path", type=str, default=None)
    parser.add_argument("--metabric_label_path", type=str, default=None)
    
    parser.add_argument(
        "--panel_a_pathways",
        type=str,
        nargs="+",
        default=None,
        help="Specify pathways for Panel A (e.g., KEGG_CELL_CYCLE KEGG_STEROID_BIOSYNTHESIS). "
             "If not specified, will auto-select from breast cancer-relevant pathways."
    )
    
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
    
    for ds in args.datasets:
        key = ds.lower()
        if key == "scanb":
            run_one_dataset(
                dataset=ds,
                dim=args.dim,
                seed=args.seed,
                out_root=out_root,
                gene_set=args.gene_set,
                gmt_path=args.gmt_path,
                pam50_col=args.pam50_col,
                ae_run_dir=args.ae_run_dir_scanb,
                bulk_test_path=args.scanb_test_path,
                label_path=args.scanb_label_path,
                bulk_train_path=None,
                panel_a_pathways=args.panel_a_pathways,
                verbose=args.verbose,
            )
        elif key == "metabric":
            run_one_dataset(
                dataset=ds,
                dim=args.dim,
                seed=args.seed,
                out_root=out_root,
                gene_set=args.gene_set,
                gmt_path=args.gmt_path,
                pam50_col=args.pam50_col,
                ae_run_dir=args.ae_run_dir_metabric,
                bulk_test_path=args.metabric_test_path,
                label_path=args.metabric_label_path,
                bulk_train_path=None,
                panel_a_pathways=args.panel_a_pathways,
                verbose=args.verbose,
            )
        else:
            raise ValueError(f"Unsupported dataset: {ds}")
    
    logging.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

