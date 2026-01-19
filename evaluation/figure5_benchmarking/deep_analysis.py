#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

# -*- coding: utf-8 -*-
"""
Deep Analysis for Figure 5: Quality vs Quantity (Original Figure 2)
Calculates:
1. Target Pathway Ranks (Are targets in Top 10 or Rank 100?)
2. Top 10 Specificity (What are the top hits? Housekeeping vs Disease?)
3. Redundancy (Jaccard Index of hits)
"""

import argparse
import logging
import pandas as pd
import numpy as np
from glob import glob

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def parse_gsea_report(report_path):
    """Parses a GSEA report file (tsv/xls)."""
    try:
        # GSEA reports are often tab-separated, but extension might vary
        df = pd.read_csv(report_path, sep='\t')
        return df
    except Exception as e:
        logging.error(f"Error reading {report_path}: {e}")
        return None

def load_gmt(gmt_path):
    """Loads GMT file into a dict: pathway -> set of genes"""
    pathway_genes = {}
    with open(gmt_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            name = parts[0]
            genes = set(parts[2:]) # Skip desc
            pathway_genes[name] = genes
    return pathway_genes

def calculate_jaccard(sets_list):
    """Calculates average pairwise Jaccard index."""
    if len(sets_list) < 2:
        return 0.0
    
    jaccards = []
    # limit to sample if too large to avoid O(N^2) explosion
    import itertools
    # Sample max 1000 pairs if needed? N=93 is fine (93*92/2 ~ 4000 pairs)
    for s1, s2 in itertools.combinations(sets_list, 2):
        u = len(s1.union(s2))
        i = len(s1.intersection(s2))
        if u > 0:
            jaccards.append(i / u)
    
    return np.mean(jaccards) if jaccards else 0.0

def find_gsea_directories(base_dir, pattern):
    """Finds all GSEA result directories matching a pattern."""
    return list(base_dir.glob(pattern))

def analyze_method(
    method_name, 
    method_base_dir, 
    target_pathways, 
    pathway_genes_map, 
    strict_fdr_threshold=0.05
):
    """
    Analyzes all GSEA runs for a specific method.
    Returns:
    - target_ranks: dict {pathway: best_rank}
    - top_pathways: list of top ranked unique pathways
    - redundancy_score: float
    - all_hits: list of hit pathways
    """
    
    # 1. Find all run directories (dimensions)
    # Use rglob to find all GseaPreranked directories recursively (handles nested gsea_runs)
    subdirs = [x for x in method_base_dir.rglob("*") if x.is_dir() and "GseaPreranked" in x.name]
    
    # Special Handle: If the base_dir itself contains reports (e.g. Standard DE in gsea_de folder)
    # Checks if we see report tsvs directly in base/
    if list(method_base_dir.glob("gsea_report_*.tsv")) or list(method_base_dir.glob("gsea_report_*.xls")):
        # If it's already in subdirs (unlikely if name doesn't match), don't duplicate
        if method_base_dir not in subdirs:
            subdirs.append(method_base_dir)
            
    logging.info(f"Analyzing {method_name}: Found {len(subdirs)} dimensions/runs.")
    
    all_results = []
    
    for subdir in subdirs:
        # Determine dimension from name if possible
        dim_str = subdir.name.split('_dim')[-1].split('.')[0]
        try:
            dim = int(dim_str)
        except:
            dim = -1
            
        # Find report file (pos and neg?) 
        # Usually checking BOTH pos and neg for hits
        # Filename pattern: gsea_report_for_na_pos_*.tsv
        
        for polarity in ['pos', 'neg']: # Check both up and down
            report_files = list(subdir.glob(f"gsea_report_for_na_{polarity}_*.tsv")) # GSEA 4.x uses tsv usually
            if not report_files:
                 report_files = list(subdir.glob(f"gsea_report_for_na_{polarity}_*.xls"))
            
            if not report_files:
                continue
                
            report_path = report_files[0]
            df = parse_gsea_report(report_path)
            
            if df is not None:
                # Columns: NAME, GS<br> follow link to MSigDB, GS DETAILS, SIZE, ES, NES, NOM p-val, FDR q-val, FWER p-val, RANK AT MAX, LEADING EDGE
                # Clean names
                df.columns = [c.upper() for c in df.columns]
                
                # Rank is roughly the Row index + 1
                # Standard reports are sorted. Row index is 'Rank in this list'.
                
                # Let's keep Rank as "Index in this file + 1"
                df['RANK'] = df.reset_index().index + 1
                df['DIMENSION'] = dim
                df['POLARITY'] = polarity
                
                all_results.append(df)
    
    if not all_results:
        logging.warning(f"No results found for {method_name}")
        return {}, [], 0.0, []
        
    full_df = pd.concat(all_results, ignore_index=True)
    
    # --- Metric 1: Target Ranks ---
    target_ranks = {}
    target_best_details = {} 
    
    for target in target_pathways:
        # Find this target in full_df
        t_rows = full_df[full_df['NAME'] == target]
        if not t_rows.empty:
            # Find best FDR
            best_row = t_rows.loc[t_rows['FDR Q-VAL'].idxmin()]
            
            # Use "Local Rank" in the *best performing dimension*.
            
            # Check significance
            if best_row['FDR Q-VAL'] > 0.05:
                # Found but not significant -> Penalize deeply
                rank = 500
                target_best_details[target] = f"ns (FDR={best_row['FDR Q-VAL']:.2f})"
            else:
                rank = best_row['RANK']
                target_best_details[target] = f"Dim {best_row['DIMENSION']} ({best_row['POLARITY']})"
            
            target_ranks[target] = rank
        else:
            target_ranks[target] = 9999 # Not found
            
    # --- Metric 2: Top 10 Specificity ---
    # Sort all results by FDR (asc), then NES (desc abs)
    sorted_df = full_df.sort_values(by=['FDR Q-VAL', 'NES'], ascending=[True, False])
    # Drop duplicates (keep best)
    unique_df = sorted_df.drop_duplicates(subset='NAME', keep='first')
    
    top_10 = unique_df.head(10)['NAME'].tolist()
    
    # --- Metric 3: Redundancy ---
    hits = unique_df[unique_df['FDR Q-VAL'] < strict_fdr_threshold]
    hits_names = hits['NAME'].tolist()
    
    jaccard_score = 0
    if len(hits_names) > 0:
        # Map names to gene sets
        sets = []
        for name in hits_names:
            if name in pathway_genes_map:
                sets.append(pathway_genes_map[name])
        
        jaccard_score = calculate_jaccard(sets)
        
    return target_ranks, top_10, jaccard_score, hits_names

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='GSE10846')
    parser.add_argument('--output_dir', default='../../results/figure2/figure2_deep_analytics')
    args = parser.parse_args()
    
    PROJECT_ROOT = Path(__file__).resolve().parents[2].absolute()
    base_dir = PROJECT_ROOT / "results" / "figure2" / "figure2_outputs" / args.dataset
    out_dir = PROJECT_ROOT / "results" / "figure2" / "figure2_deep_analytics"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Targets
    target_file = base_dir / f"{args.dataset}_target_pathways.csv"
    if not target_file.exists():
        logging.error("Target file not found")
        return
    
    target_df = pd.read_csv(target_file)
    target_pathways = target_df['pathway'].tolist()
    
    # 2. Find GMT (grab one from existing runs)
    gmt_files = list(base_dir.rglob("gene_sets.gmt"))
    if not gmt_files:
        logging.error("GMT file not found in results")
        return
    gmt_path = gmt_files[0]
    logging.info(f"Using GMT file: {gmt_path}")
    pathway_genes = load_gmt(gmt_path)
    
    # 3. Analyze Methods
    def get_path(new_name, old_name):
        new_path = base_dir / new_name
        if new_path.exists():
            return new_path
        old_path = base_dir / old_name
        if old_path.exists():
            return old_path
        return new_path

    methods = {
        'PCA_Corr': get_path('pca_corr', 'gsea_pca_correlation'),
        'PCA_Weights': get_path('pca_weight', 'gsea_pca_weights'),
        'AE_Correlation': get_path('ae_corr', 'gsea_ae_correlation_50'),
        'AE_DeepLIFT_Mean': base_dir / 'ae_deeplift_mean',
        'AE_SHAP_Mean': base_dir / 'ae_shap_mean',
        'AE_DeepLIFT_Zero': base_dir / 'ae_deeplift_zero',
        'AE_SHAP_Zero': base_dir / 'ae_shap_zero'
    }
    
    summary_data = []
    top_10_data = {}
    
    # Bonferroni for 64 dimensions
    strict_fdr = 0.05 / 64
    
    std_path = base_dir / 'standard_de'
    if std_path.exists():
         methods['Standard_DE'] = std_path
    
    for m_name, m_path in methods.items():
        if not m_path.exists():
            logging.warning(f"Path not found: {m_path}")
            continue
            
        threshold = 0.05 if m_name == 'Standard_DE' else strict_fdr
            
        ranks, top10, redundancy, hits = analyze_method(
            m_name, m_path, target_pathways, pathway_genes, threshold
        )
        
        top_10_data[m_name] = top10
        
        summary_data.append({
            'Method': m_name,
            'Num_Hits_Strict': len(hits),
            'Redundancy_Jaccard': redundancy,
            'Avg_Target_Rank': np.mean(list(ranks.values())) if ranks else 0
        })
        
        for t, r in ranks.items():
            summary_data.append({
                'Method': m_name,
                'Metric_Type': 'Target_Rank',
                'Pathway': t,
                'Value': r
            })
    
    df_summary = pd.DataFrame(summary_data)
    df_summary['Dataset'] = args.dataset
    df_summary.to_csv(out_dir / f'deep_metrics_{args.dataset}.csv', index=False)
    
    max_len = 10
    for k in top_10_data:
        if len(top_10_data[k]) < max_len:
             top_10_data[k].extend([None] * (max_len - len(top_10_data[k])))
             
    df_top10 = pd.DataFrame(top_10_data)
    df_top10.to_csv(out_dir / f'top_10_pathways_{args.dataset}.csv', index=False)
    
    logging.info(f"Deep analysis complete. Results saved to {out_dir / f'deep_metrics_{args.dataset}.csv'} and {out_dir / f'top_10_pathways_{args.dataset}.csv'}")

if __name__ == '__main__':
    main()
