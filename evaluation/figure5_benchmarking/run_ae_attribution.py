#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

# -*- coding: utf-8 -*-
"""
AE-specific analysis script for Figure 2.
Focuses on DeepLIFT and SHAP methods.
"""

import argparse
import logging
import pandas as pd
import torch

# Import project modules
from scripts.figure2.figure2_data_preparation import load_gse10846_data, load_gse48350_data
from scripts.figure2.figure2_methods import train_ae, run_ae_attribution, save_rnk_file
from run_figure2_pipeline import process_gsea_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def get_existing_targets(dataset_name: str, output_dir: Path):
    target_path = output_dir / dataset_name / "GSE10846_target_pathways.csv"
    if not target_path.exists():
        # Try generic name format just in case
        target_path = output_dir / dataset_name / f"{dataset_name}_target_pathways.csv"
    
    if target_path.exists():
        logging.info(f"Loading target pathways from {target_path}")
        df = pd.read_csv(target_path)
        return df['pathway'].tolist()
    else:
        logging.warning(f"Target pathways file not found at {target_path}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='GSE10846')
    parser.add_argument('--methods', nargs='+', default=['deeplift', 'shap'], 
                        choices=['deeplift', 'shap'])
    parser.add_argument('--output_dir', default='../../results/figure2/figure2_outputs')
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    ds_out_dir = out_dir / args.dataset
    ds_out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    logging.info(f"Loading data for {args.dataset}...")
    data_prep_dir = Path('../../results/figure2/figure2_prepared_data')
    expr_path = data_prep_dir / f"{args.dataset}_expression.csv"
    
    if expr_path.exists():
        expr_df = pd.read_csv(expr_path, index_col=0)
    else:
        logging.error(f"Data file not found: {expr_path}")
        return

    # 2. Get Targets
    target_pathways = get_existing_targets(args.dataset, out_dir)
    if not target_pathways:
        logging.error("Cannot proceed without target pathways.")
        return
    logging.info(f"Found {len(target_pathways)} target pathways.")

    # 3. Train AE
    logging.info("Training Autoencoder...")
    ae_config = {
        'hidden_dims': [1024, 512],
        'encoder_output_dim': 64,
        'batch_size': 128,
        'num_epochs': 800
    }
    encoder, ae_emb = train_ae(expr_df, ae_config)
    
    summary_results = []
    
    # 4. Run Methods (Both baseline types: mean and zero)
    for baseline_type in ['mean', 'zero']:
        logging.info(f"=== Running analysis with {baseline_type} baseline ===")
        
        if 'deeplift' in args.methods:
            logging.info(f"Starting DeepLIFT ({baseline_type}) analysis...")
            dl_dir = ds_out_dir / f"ae_deeplift_{baseline_type}"
            rnk_files = run_ae_attribution(encoder, expr_df, dl_dir, args.dataset, 'deeplift', baseline_type)
            
            logging.info(f"Running GSEA for DeepLIFT ({baseline_type})...")
            # Use specific name for GSEA results based on baseline
            res = process_gsea_results(rnk_files, "KEGG", dl_dir, args.dataset, f"AE_DeepLIFT_{baseline_type}", target_pathways)
            res['dataset'] = args.dataset
            
            # Save intermediate results
            res.to_csv(dl_dir / f"deeplift_{baseline_type}_gsea_summary.csv", index=False)
            summary_results.extend(res.to_dict('records'))
            
        if 'shap' in args.methods:
            logging.info(f"Starting SHAP ({baseline_type}) analysis...")
            shap_dir = ds_out_dir / f"ae_shap_{baseline_type}"
            rnk_files = run_ae_attribution(encoder, expr_df, shap_dir, args.dataset, 'shap', baseline_type)
            
            logging.info(f"Running GSEA for SHAP ({baseline_type})...")
            res = process_gsea_results(rnk_files, "KEGG", shap_dir, args.dataset, f"AE_SHAP_{baseline_type}", target_pathways)
            res['dataset'] = args.dataset
            
            # Save intermediate results
            res.to_csv(shap_dir / f"shap_{baseline_type}_gsea_summary.csv", index=False)
            summary_results.extend(res.to_dict('records'))
        
    logging.info("Analysis complete.")

if __name__ == '__main__':
    main()
