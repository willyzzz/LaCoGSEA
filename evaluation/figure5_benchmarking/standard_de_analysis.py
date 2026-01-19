#!/usr/bin/env python
from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

# -*- coding: utf-8 -*-
"""
Figure 5: Standard Differential Expression Analysis (Limma/t-test + GSEA)
Benchmarking baseline: Filter Top 10 Target Pathways with FDR 0.05-0.25 (Original Figure 2)
"""

import argparse
import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from core.run_gsea_java import find_gsea_cli, find_gene_sets_dir, run_gsea_preranked

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def perform_differential_expression(
    expr_df: pd.DataFrame,
    labels: pd.Series,
) -> pd.DataFrame:
    """
    Perform differential expression analysis using t-test.
    
    Args:
        expr_df: Expression matrix (samples x genes)
        labels: Series with group labels (e.g., 'ABC' vs 'GCB', 'AD' vs 'Control')
    
    Returns:
        DataFrame with columns: [gene, logFC, t_statistic, p_value]
        Sorted by absolute t_statistic
    """
    logging.info("Performing differential expression analysis...")
    
    # Get unique groups
    unique_groups = labels.unique()
    if len(unique_groups) != 2:
        raise ValueError(f"Expected 2 groups, got {len(unique_groups)}: {unique_groups}")
    
    group1, group2 = unique_groups[0], unique_groups[1]
    logging.info(f"Comparing {group1} (n={sum(labels == group1)}) vs {group2} (n={sum(labels == group2)})")
    
    # Align data
    common_idx = expr_df.index.intersection(labels.index)
    expr_aligned = expr_df.loc[common_idx]
    labels_aligned = labels.loc[common_idx]
    
    group1_mask = labels_aligned == group1
    group2_mask = labels_aligned == group2
    
    group1_expr = expr_aligned[group1_mask]
    group2_expr = expr_aligned[group2_mask]
    
    # Perform t-test for each gene
    results = []
    for gene in expr_aligned.columns:
        group1_values = group1_expr[gene].dropna().values
        group2_values = group2_expr[gene].dropna().values
        
        if len(group1_values) < 3 or len(group2_values) < 3:
            continue
        
        try:
            t_stat, p_val = ttest_ind(group1_values, group2_values, equal_var=False)
            # Calculate logFC (mean difference)
            mean1 = np.mean(group1_values)
            mean2 = np.mean(group2_values)
            logFC = mean1 - mean2
            
            results.append({
                'gene': gene,
                'logFC': logFC,
                't_statistic': t_stat,
                'p_value': p_val,
            })
        except:
            continue
    
    de_df = pd.DataFrame(results)
    de_df = de_df.sort_values('t_statistic', key=abs, ascending=False)
    
    logging.info(f"DE analysis complete: {len(de_df)} genes analyzed")
    return de_df


def create_rnk_file(
    de_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Create RNK file for GSEA from DE results.
    
    Format: gene_name <tab> score
    Score can be t_statistic or logFC
    """
    # Use t_statistic as the ranking score
    rnk_df = de_df[['gene', 't_statistic']].copy()
    rnk_df = rnk_df.sort_values('t_statistic', ascending=False)
    
    rnk_df.to_csv(output_path, sep='\t', index=False, header=False)
    logging.info(f"Saved RNK file: {output_path}")


def run_gsea_analysis(
    rnk_file: Path,
    gene_set: str,
    output_dir: Path,
    dataset_name: str,
) -> Path:
    """
    Run GSEA analysis on the RNK file.
    
    Returns:
        Path to GSEA results directory
    """
    gsea_cli = find_gsea_cli()
    if not gsea_cli:
        raise FileNotFoundError("GSEA CLI not found")
    
    gene_sets_dir_str = find_gene_sets_dir()
    if not gene_sets_dir_str:
        raise FileNotFoundError("Gene sets directory not found")
    
    gene_sets_dir = Path(gene_sets_dir_str)
    
    # Find gene set file
    gene_set_lower = gene_set.lower()
    if gene_set_lower == 'kegg':
        gene_set_file = gene_sets_dir / "c2.cp.kegg_legacy.v2025.1.Hs.symbols.gmt"
        if not gene_set_file.exists():
            # Try other KEGG files
            possible_files = list(gene_sets_dir.glob("*kegg*.gmt"))
            if possible_files:
                gene_set_file = possible_files[0]
            else:
                raise FileNotFoundError(f"KEGG gene set file not found in {gene_sets_dir}")
    elif gene_set_lower == 'go':
        gene_set_file = gene_sets_dir / "c5.go.bp.v2025.1.Hs.symbols.gmt"
        if not gene_set_file.exists():
            possible_files = list(gene_sets_dir.glob("*go.bp*.gmt"))
            if possible_files:
                gene_set_file = possible_files[0]
            else:
                raise FileNotFoundError(f"GO gene set file not found in {gene_sets_dir}")
    else:
        raise ValueError(f"Unsupported gene set: {gene_set}")
    
    logging.info(f"Using gene set file: {gene_set_file}")
    
    # Convert paths to absolute paths (required by GSEA)
    rnk_file_abs = Path(rnk_file).resolve().absolute()
    gene_set_file_abs = Path(gene_set_file).resolve().absolute()
    output_dir_abs = Path(output_dir).resolve().absolute()
    
    # Ensure output directory exists
    output_dir_abs.mkdir(parents=True, exist_ok=True)
    
    # Run GSEA
    label = f"{dataset_name}_standard_DE"
    success = run_gsea_preranked(
        rnk_file=str(rnk_file_abs),
        gene_set_file=str(gene_set_file_abs),
        output_dir=str(output_dir_abs),
        label=label,
        memory="4g",
        permutations=1000,
    )
    
    if not success:
        raise RuntimeError(f"GSEA analysis failed for {dataset_name}")
    
    # Find the result directory (GSEA creates a directory with pattern: {label}.GseaPreranked.*)
    result_pattern = f"{label}.GseaPreranked.*"
    result_dirs = list(output_dir_abs.glob(result_pattern))
    
    if not result_dirs:
        # Also check for error directories
        error_pattern = f"error_{label}.GseaPreranked.*"
        error_dirs = list(output_dir_abs.glob(error_pattern))
        if error_dirs:
            # Check if error directory has results
            error_dir = error_dirs[0]
            tsv_files = list(error_dir.glob("gsea_report_for_na_*.tsv"))
            if tsv_files:
                logging.warning(f"Using error directory with results: {error_dir}")
                return error_dir
        
        raise FileNotFoundError(f"GSEA result directory not found in {output_dir_abs} with pattern {result_pattern}")
    
    return result_dirs[0]


def extract_gsea_fdr(
    gsea_result_dir: Path,
) -> pd.DataFrame:
    """
    Extract FDR values from GSEA results.
    
    Returns:
        DataFrame with columns: [pathway, NES, FDR, p_value]
    """
    from core.nes_from_gsea_reports import read_gsea_report_tsv, extract_pathway_nes, merge_pos_neg_nes
    
    # Find GSEA report files
    pos_files = list(gsea_result_dir.glob("*pos*.tsv"))
    neg_files = list(gsea_result_dir.glob("*neg*.tsv"))
    
    if not pos_files and not neg_files:
        raise FileNotFoundError(f"No GSEA report files found in {gsea_result_dir}")
    
    # Read reports
    all_pathways = {}
    
    for report_file in pos_files + neg_files:
        try:
            # Skip ranked_gene_list files
            if 'ranked_gene_list' in report_file.name:
                continue
            
            # Read directly with pandas to avoid column cleaning issues
            report_df = pd.read_csv(report_file, sep='\t', low_memory=False)
            
            # Extract pathway name, NES, FDR, p-value
            name_col = None
            nes_col = None
            fdr_col = None
            pval_col = None
            
            for col in report_df.columns:
                col_lower = str(col).lower()
                if col_lower == 'name' or (('name' in col_lower or 'term' in col_lower) and name_col is None):
                    name_col = col
                elif col_lower == 'nes':
                    nes_col = col
                elif 'fdr' in col_lower and 'q' in col_lower:
                    fdr_col = col
                elif ('nom' in col_lower or 'p-val' in col_lower) and 'fdr' not in col_lower and 'fwer' not in col_lower:
                    pval_col = col
            
            if name_col and fdr_col:
                for _, row in report_df.iterrows():
                    pathway = str(row[name_col]).strip()
                    if pathway == 'nan' or pathway == '':
                        continue
                    fdr = pd.to_numeric(row[fdr_col], errors='coerce')
                    nes = pd.to_numeric(row[nes_col], errors='coerce') if nes_col else np.nan
                    pval = pd.to_numeric(row[pval_col], errors='coerce') if pval_col else np.nan
                    
                    if not pd.isna(fdr):
                        # Keep pathway with lower FDR if duplicate
                        if pathway not in all_pathways or fdr < all_pathways[pathway]['FDR']:
                            all_pathways[pathway] = {
                                'NES': nes,
                                'FDR': fdr,
                                'p_value': pval,
                            }
        except Exception as e:
            logging.warning(f"Error reading {report_file}: {e}")
            import traceback
            traceback.print_exc()
    
    if len(all_pathways) == 0:
        raise ValueError("No pathways found in GSEA results")
    
    # Convert to DataFrame
    results = []
    for pathway, values in all_pathways.items():
        results.append({
            'pathway': pathway,
            'NES': values['NES'],
            'FDR': values['FDR'],
            'p_value': values['p_value'],
        })
    
    gsea_df = pd.DataFrame(results)
    gsea_df = gsea_df.sort_values('FDR')
    
    logging.info(f"Extracted {len(gsea_df)} pathways from GSEA results")
    return gsea_df


def select_target_pathways(
    gsea_df: pd.DataFrame,
    fdr_min: float = 0.05,
    fdr_max: float = 0.25,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Select target pathways with FDR between fdr_min and fdr_max.
    
    Args:
        gsea_df: DataFrame from extract_gsea_fdr
        fdr_min: Minimum FDR (default: 0.05)
        fdr_max: Maximum FDR (default: 0.25)
        top_n: Number of top pathways to select (default: 10)
    
    Returns:
        DataFrame with target pathways, sorted by FDR
    """
    # Filter by FDR range
    target_df = gsea_df[
        (gsea_df['FDR'] >= fdr_min) & 
        (gsea_df['FDR'] <= fdr_max)
    ].copy()
    
    if len(target_df) == 0:
        logging.warning(f"No pathways found with FDR between {fdr_min} and {fdr_max}")
        return pd.DataFrame(columns=['pathway', 'NES', 'FDR', 'p_value'])
    
    # Sort by FDR and select top N
    target_df = target_df.sort_values('FDR')
    target_df = target_df.head(top_n)
    
    logging.info(f"Selected {len(target_df)} target pathways (FDR {fdr_min}-{fdr_max})")
    return target_df


def run_standard_de_analysis(
    expr_path: Path,
    labels_path: Path,
    dataset_name: str,
    gene_set: str,
    output_dir: Path,
    fdr_min: float = 0.05,
    fdr_max: float = 0.25,
    top_n: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run complete standard DE analysis pipeline.
    
    Returns:
        (gsea_results_df, target_pathways_df)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logging.info(f"Loading data for {dataset_name}...")
    expr_df = pd.read_csv(expr_path, index_col=0)
    labels_df = pd.read_csv(labels_path, index_col=0)
    labels = labels_df['Subtype']
    
    logging.info(f"Expression matrix: {expr_df.shape}")
    logging.info(f"Labels: {labels.value_counts().to_dict()}")
    
    # Step 1: Differential expression
    de_df = perform_differential_expression(expr_df, labels)
    de_path = output_dir / f"{dataset_name}_de_results.csv"
    de_df.to_csv(de_path, index=False)
    logging.info(f"Saved DE results: {de_path}")
    
    # Step 2: Create RNK file
    rnk_path = output_dir / f"{dataset_name}_standard_DE.rnk"
    create_rnk_file(de_df, rnk_path)
    
    # Step 3: Run GSEA
    logging.info("Running GSEA analysis...")
    gsea_result_dir = run_gsea_analysis(
        rnk_file=rnk_path,
        gene_set=gene_set,
        output_dir=output_dir,
        dataset_name=dataset_name,
    )
    
    # Step 4: Extract FDR
    gsea_df = extract_gsea_fdr(gsea_result_dir)
    gsea_path = output_dir / f"{dataset_name}_standard_gsea_results.csv"
    gsea_df.to_csv(gsea_path, index=False)
    logging.info(f"Saved GSEA results: {gsea_path}")
    
    # Step 5: Select target pathways
    target_df = select_target_pathways(gsea_df, fdr_min=fdr_min, fdr_max=fdr_max, top_n=top_n)
    target_path = output_dir / f"{dataset_name}_target_pathways.csv"
    target_df.to_csv(target_path, index=False)
    logging.info(f"Saved target pathways: {target_path}")
    
    return gsea_df, target_df


def main():
    parser = argparse.ArgumentParser(description='Figure 5: Standard Differential Expression Analysis (Original Figure 2)')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (GSE10846, GSE48350, GSE11375)')
    parser.add_argument('--expr_path', type=str, default=None,
                       help='Path to expression matrix (default: figure2_prepared_data/{dataset}_expression.csv)')
    parser.add_argument('--labels_path', type=str, default=None,
                       help='Path to labels file (default: figure2_prepared_data/{dataset}_labels.csv)')
    parser.add_argument('--gene_set', type=str, default='KEGG',
                       choices=['KEGG', 'GO'],
                       help='Gene set for GSEA (default: KEGG)')
    parser.add_argument('--output_dir', type=str, default='../../results/figure2/figure2_outputs',
                       help='Output directory (default: figure2_outputs)')
    parser.add_argument('--fdr_min', type=float, default=0.05,
                       help='Minimum FDR for target pathways (default: 0.05)')
    parser.add_argument('--fdr_max', type=float, default=0.25,
                       help='Maximum FDR for target pathways (default: 0.25)')
    parser.add_argument('--top_n', type=int, default=10,
                       help='Number of top target pathways (default: 10)')
    
    args = parser.parse_args()
    
    # Set default paths
    if args.expr_path is None:
        args.expr_path = f"figure2_prepared_data/{args.dataset}_expression.csv"
    if args.labels_path is None:
        args.labels_path = f"figure2_prepared_data/{args.dataset}_labels.csv"
    
    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run analysis
    gsea_df, target_df = run_standard_de_analysis(
        expr_path=Path(args.expr_path),
        labels_path=Path(args.labels_path),
        dataset_name=args.dataset,
        gene_set=args.gene_set,
        output_dir=output_dir,
        fdr_min=args.fdr_min,
        fdr_max=args.fdr_max,
        top_n=args.top_n,
    )
    
    logging.info(f"Analysis complete. Results saved to {output_dir}")
    logging.info(f"Target pathways: {list(target_df['pathway'].values)}")


if __name__ == '__main__':
    main()
