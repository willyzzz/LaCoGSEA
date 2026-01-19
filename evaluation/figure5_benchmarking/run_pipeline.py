#!/usr/bin/env python
from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

# -*- coding: utf-8 -*-
"""
Orchestration script for Figure 2 Analysis (Standard Version).
Runs analysis on 5 datasets: GSE10846, GSE48350, GSE11375, GSE126848, GSE116250.
"""

import argparse
import logging
import pandas as pd
import numpy as np
from typing import List

# Import modules
from scripts.figure2.figure2_data_preparation import (
    load_gse10846_data, load_gse48350_data, load_gse11375_data, 
    load_gse126848_data, load_gse116250_data, save_prepared_data
)
from scripts.figure2.figure2_standard_de_analysis import run_standard_de_analysis, extract_gsea_fdr
from scripts.figure2.figure2_methods import (
    run_pca_correlation,
    run_pca_weights,
    train_ae,
    run_ae_correlation,
    run_ae_attribution
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def get_prepared_data(dataset_name: str, output_dir: Path):
    """Load or prepare data."""
    expr_path = output_dir / f"{dataset_name}_expression.csv"
    labels_path = output_dir / f"{dataset_name}_labels.csv"
    
    if expr_path.exists() and labels_path.exists():
        logging.info(f"Loading cached data for {dataset_name}...")
        expr_df = pd.read_csv(expr_path, index_col=0)
        labels_df = pd.read_csv(labels_path, index_col=0)
        labels = labels_df['Subtype']
        return expr_df, labels
    
    # Prepare fresh
    logging.info(f"Preparing data for {dataset_name}...")
    loaders = {
        'GSE10846': load_gse10846_data,
        'GSE48350': load_gse48350_data,
        'GSE11375': load_gse11375_data,
        'GSE126848': load_gse126848_data,
        'GSE116250': load_gse116250_data
    }
    
    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    expr_df, labels = loaders[dataset_name]()
    save_prepared_data(expr_df, labels, dataset_name, output_dir)
    return expr_df, labels

def process_single_gsea(rnk_file, gene_set, output_dir, dataset_name, method_name, target_pathways):
    """Worker function for parallel GSEA."""
    try:
        from core.run_gsea_java import run_gsea_preranked, find_gene_sets_dir
        
        dim_str = rnk_file.stem.split('_dim')[-1]
        try:
            dim = int(dim_str)
        except ValueError:
            dim = -1
            
        gsea_label = f"{dataset_name}_{method_name}_dim{dim}"
        gene_sets_dir = Path(find_gene_sets_dir())
        
        if gene_set.lower() == 'kegg':
            possible = list(gene_sets_dir.glob("c2.cp.kegg*.gmt"))
            if not possible: return []
            gmt_file = str(possible[0])
        else:
            possible = list(gene_sets_dir.glob("c5.go.bp*.gmt"))
            if not possible: return []
            gmt_file = str(possible[0])
            
        run_out_dir = output_dir / "gsea_runs"
        run_out_dir.mkdir(parents=True, exist_ok=True)
        
        success = run_gsea_preranked(
            rnk_file=str(rnk_file.resolve()),
            gene_set_file=gmt_file,
            output_dir=str(run_out_dir.resolve()),
            label=gsea_label,
            permutations=1000
        )
        
        if success:
            res_dirs = list(run_out_dir.glob(f"{gsea_label}.GseaPreranked.*"))
            if res_dirs:
                res_dir = sorted(res_dirs, key=lambda p: p.stat().st_mtime, reverse=True)[0]
                gsea_df = extract_gsea_fdr(res_dir)
                
                results = []
                if target_pathways:
                    gsea_df['pathway_norm'] = gsea_df['pathway'].str.upper()
                    target_norm = [t.upper() for t in target_pathways]
                    filtered = gsea_df[gsea_df['pathway_norm'].isin(target_norm)].copy()
                    
                    for _, row in filtered.iterrows():
                        results.append({
                            'method': method_name,
                            'dimension': dim,
                            'pathway': row['pathway'],
                            'FDR': row['FDR'],
                            'NES': row['NES']
                        })
                return results
    except Exception as e:
        logging.error(f"Failed GSEA for {rnk_file}: {e}")
    return []

def process_gsea_results(rnk_files, gene_set, output_dir, dataset_name, method_name, target_pathways=None):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import os
    results = []
    num_workers = min(os.cpu_count(), 4)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_single_gsea, rnk, gene_set, output_dir, dataset_name, method_name, target_pathways) for rnk in rnk_files]
        for future in as_completed(futures):
            results.extend(future.result())
    return pd.DataFrame(results)

def run_evaluation(expr_df, labels, ds, ds_out_dir, args, encoder=None, ae_emb=None):
    logging.info(f"--- Processing Dataset {ds} ---")
    ds_out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Standard DE Baseline
    std_de_dir = ds_out_dir / "standard_de"
    tmp_expr_path = ds_out_dir / f"{ds}_expr.csv"
    tmp_labels_path = ds_out_dir / f"{ds}_labels.csv"
    expr_df.to_csv(tmp_expr_path)
    labels.to_frame().to_csv(tmp_labels_path)

    gsea_df, target_df = run_standard_de_analysis(
        expr_path=tmp_expr_path,
        labels_path=tmp_labels_path,
        dataset_name=ds,
        gene_set="KEGG",
        output_dir=std_de_dir
    )
    
    # Clean up temp files
    if tmp_expr_path.exists(): tmp_expr_path.unlink()
    if tmp_labels_path.exists(): tmp_labels_path.unlink()

    target_pathways = target_df['pathway'].tolist()
    if not target_pathways: 
        logging.warning(f"No target pathways for {ds}")
        return []
    
    summary_results = []
    for _, row in target_df.iterrows():
        summary_results.append({
            'method': 'Standard_DE',
            'dimension': 'NA',
            'pathway': row['pathway'],
            'FDR': row['FDR'],
            'NES': row['NES']
        })
        
    methods = args.methods
    if 'all' in methods: methods = ['pca_corr', 'pca_weight', 'ae_corr', 'ae_deeplift']

    # PCA
    if 'pca_corr' in methods:
        rnk = run_pca_correlation(expr_df, 64, ds_out_dir / "pca_corr", ds)
        res = process_gsea_results(rnk, "KEGG", ds_out_dir / "pca_corr", ds, "PCA_Corr", target_pathways)
        if not res.empty:
            summary_results.extend(res.to_dict('records'))

    if 'pca_weight' in methods:
        rnk = run_pca_weights(expr_df, 64, ds_out_dir / "pca_weight", ds)
        res = process_gsea_results(rnk, "KEGG", ds_out_dir / "pca_weight", ds, "PCA_Weight", target_pathways)
        if not res.empty:
            summary_results.extend(res.to_dict('records'))

    # AE
    if encoder is not None:
        if 'ae_corr' in methods:
            rnk = run_ae_correlation(encoder, ae_emb, expr_df, ds_out_dir / "ae_corr", ds)
            res = process_gsea_results(rnk, "KEGG", ds_out_dir / "ae_corr", ds, "AE_Corr", target_pathways)
            if not res.empty:
                summary_results.extend(res.to_dict('records'))
        if 'ae_deeplift' in methods:
            rnk = run_ae_attribution(encoder, expr_df, ds_out_dir / "ae_deeplift", ds, 'deeplift')
            res = process_gsea_results(rnk, "KEGG", ds_out_dir / "ae_deeplift", ds, "AE_DeepLIFT", target_pathways)
            if not res.empty:
                summary_results.extend(res.to_dict('records'))
            
    return summary_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['GSE10846', 'GSE48350', 'GSE11375', 'GSE126848', 'GSE116250'])
    parser.add_argument('--methods', nargs='+', default=['all'])
    parser.add_argument('--output_dir', default='../../results/figure2/figure2_outputs')
    args = parser.parse_args()
    
    root_out = Path(args.output_dir).absolute()
    prep_dir = Path('../../results/figure2/figure2_prepared_data').absolute()
    
    for ds in args.datasets:
        logging.info(f"\n{'='*60}\nProcessing Dataset: {ds}\n{'='*60}")
        expr_df, labels = get_prepared_data(ds, prep_dir)
        ds_out = root_out / ds
        
        ae_config = {'hidden_dims': [1024, 512], 'encoder_output_dim': 64, 'batch_size': 128, 'num_epochs': 800}
        model_path = ds_out / "ae_model_800.pt"
        
        # Load or Train AE
        encoder, ae_emb = train_ae(expr_df, ae_config, model_path=model_path)
        
        all_results = run_evaluation(expr_df, labels, ds, ds_out, args, encoder, ae_emb)
        
        final_df = pd.DataFrame(all_results)
        final_df.to_csv(ds_out / f"{ds}_final_comparative_results.csv", index=False)
        logging.info(f"Saved results for {ds} to {ds_out / f'{ds}_final_comparative_results.csv'}")

if __name__ == '__main__':
    main()
