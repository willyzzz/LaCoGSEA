#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

# -*- coding: utf-8 -*-
"""
Negative Control Experiment (Original Figure 1B)
Used to verify method specificity: On completely random data, the method should not produce significant pathways.

Workflow:
1. Generate random noise matrix (same size as real data).
2. Run AutoEncoder + GSEA.
3. Run PCA + GSEA.
4. Compare results: Ideally, random data should produce zero significant pathways.
"""

import os
import logging
import argparse
import numpy as np
import pandas as pd
import tempfile
import shutil
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Import core modules
from core.barlow_config import config
from scripts.run_full_pipeline import (
    train_auto_encoder_model, 
    find_gsea_cli,
    find_gene_sets_dir
)
from core.evaluation import calculate_pearson_correlation
from core.run_gsea_java import run_gsea_preranked
from core.dataset_input import create_dataloader_from_config, set_seed
from core.gene_mapping import convert_ensembl_to_symbol, is_ensembl_id

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2].absolute()


def generate_random_noise_matrix(n_samples, n_genes, seed=42, real_gene_names=None):
    """
    Generate random noise matrix (Normal distribution).
    
    Args:
        n_samples: Number of samples
        n_genes: Number of genes
        seed: Random seed
        real_gene_names: List of real gene names (if provided, use them; else use random names)
    
    Returns:
        DataFrame: Random matrix matching real data format (samples x genes)
    """
    np.random.seed(seed)
    
    # Generate random matrix with normal distribution
    random_matrix = np.random.normal(loc=0.0, scale=1.0, size=(n_samples, n_genes))
    
    # Create DataFrame
    sample_ids = [f"random_sample_{i}" for i in range(n_samples)]
    
    # Use real gene names if provided so GSEA can run against known gene sets
    if real_gene_names is not None and len(real_gene_names) >= n_genes:
        gene_names = real_gene_names[:n_genes]
    else:
        # Fallback to random names
        gene_names = [f"random_gene_{i}" for i in range(n_genes)]
    
    df = pd.DataFrame(random_matrix, index=sample_ids, columns=gene_names)
    
    return df


def load_real_data_shape(dataset_name="scanb"):
    """
    Load real data to get dimensions.
    
    Returns:
        tuple: (n_samples_train, n_samples_test, n_genes, train_df, test_df)
    """
    # Temporarily modify config to load data
    original_dataset = config['testing_dataset_name']
    config['testing_dataset_name'] = dataset_name
    
    try:
        # Load training set
        bulk_path = config.get('bulk_path', '')
        if not bulk_path or not os.path.exists(bulk_path):
            raise ValueError(f"Training data path not found: {bulk_path}")
        
        train_df = pd.read_csv(bulk_path, sep=',', index_col=0)
        if 'GEX.assay' in train_df.columns:
            train_df = train_df.drop(columns=['GEX.assay'])
        
        # Load test set
        test_bulk_path = config.get('bulk_test_path', '')
        if not test_bulk_path or not os.path.exists(test_bulk_path):
            raise ValueError(f"Test data path not found: {test_bulk_path}")
        
        test_df = pd.read_csv(test_bulk_path, sep=',', index_col=0)
        if 'GEX.assay' in test_df.columns:
            test_df = test_df.drop(columns=['GEX.assay'])
        
        n_samples_train = len(train_df)
        n_samples_test = len(test_df)
        n_genes = len(train_df.columns)
        
        logging.info(f"Real data dimensions: Train={n_samples_train} samples, Test={n_samples_test} samples, {n_genes} genes")
        
        return n_samples_train, n_samples_test, n_genes, train_df, test_df
        
    finally:
        # Restore original config
        config['testing_dataset_name'] = original_dataset


def run_pca_on_random_data(random_train_df, random_test_df, output_dir, n_components=128):
    """
    Run PCA on random data and generate RNK files.
    """
    logging.info("="*80)
    logging.info("Running PCA analysis (Random Data)")
    logging.info("="*80)
    
    # Combine train and test sets
    all_data = pd.concat([random_train_df, random_test_df], axis=0)
    
    # Standardize
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(all_data)
    
    # Run PCA
    n_components = min(n_components, min(scaled_data.shape) - 1)
    pca = PCA(n_components=n_components, random_state=42)
    pca_result = pca.fit_transform(scaled_data)
    
    logging.info(f"PCA complete: Explained variance ratio = {pca.explained_variance_ratio_[:5]}")
    logging.info(f"Cumulative variance (Top 5): {np.cumsum(pca.explained_variance_ratio_[:5])}")
    
    # Create output directory
    pca_rnk_dir = output_dir / "PCA_RNK_files"
    pca_rnk_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate RNK for each PC
    rnk_files = []
    for i in range(n_components):
        # Get loadings
        pc_loadings = pca.components_[i, :]
        
        # Sort by absolute value
        gene_names = all_data.columns
        pc_df = pd.DataFrame({
            'gene': gene_names,
            'score': pc_loadings
        })
        pc_df = pc_df.sort_values('score', key=abs, ascending=False)
        
        # Save as RNK
        rnk_file = pca_rnk_dir / f"Random_PCA_dim{n_components}_PC{i+1}.rnk"
        pc_df[['gene', 'score']].to_csv(rnk_file, sep='\t', index=False, header=False)
        rnk_files.append(str(rnk_file))
        
        if (i + 1) % 10 == 0:
            logging.info(f"Generated {i+1}/{n_components} PCA RNK files")
    
    logging.info(f"PCA RNK files saved to: {pca_rnk_dir}")
    return rnk_files


def run_gsea_on_rnk_files(rnk_files, gene_set_type="GO", output_dir=None, is_pca=False, n_perm=100):
    """
    Run GSEA on RNK files.
    """
    logging.info("="*80)
    logging.info(f"Running GSEA analysis (Gene Set: {gene_set_type})")
    logging.info("="*80)
    
    # Find GSEA and gene set files
    gsea_cli = find_gsea_cli()
    if not gsea_cli:
        logging.error("GSEA CLI not found")
        return None
    
    gene_sets_dir = find_gene_sets_dir()
    if not gene_sets_dir:
        logging.error("Gene sets directory not found")
        return None
    
    # Find gene set file
    gene_set_file = None
    if gene_set_type.upper() == 'GO':
        possible_names = [
            "c5.go.bp.v2025.1.Hs.symbols.gmt",
            "c5.go.bp.v2023.2.Hs.symbols.gmt",
            "c5.go.bp.v2022.1.Hs.symbols.gmt",
        ]
        for name in possible_names:
            test_path = os.path.join(gene_sets_dir, name)
            if os.path.exists(test_path):
                gene_set_file = test_path
                break
    elif gene_set_type.upper() == 'KEGG':
        possible_names = [
            "c2.cp.kegg.v2025.1.Hs.symbols.gmt",
            "c2.cp.kegg_medicus.v2025.1.Hs.symbols.gmt",
        ]
        for name in possible_names:
            test_path = os.path.join(gene_sets_dir, name)
            if os.path.exists(test_path):
                gene_set_file = test_path
                break
    
    if not gene_set_file:
        logging.error(f"Gene set file for {gene_set_type} not found")
        return None
    
    logging.info(f"Using gene set file: {gene_set_file}")
    
    # Run GSEA
    results = []
    significant_count = 0
    
    # Implementation of checkpoint/resumption
    completed_labels = set()
    if output_dir and os.path.exists(output_dir):
        for tsv_file in os.listdir(output_dir):
            if tsv_file.endswith('_pos.tsv') or tsv_file.endswith('_neg.tsv'):
                label = tsv_file.replace('_pos.tsv', '').replace('_neg.tsv', '')
                completed_labels.add(label)
        if completed_labels:
            logging.info(f"Found {len(completed_labels)} completed GSEA results, skipping...")
    
    for i, rnk_file in enumerate(rnk_files):
        if is_pca:
            pc_num = i + 1
            output_name = f"Random_PCA_PC{pc_num}"
        else:
            import re
            match = re.search(r'dimension_(\d+)_correlation', rnk_file)
            dim_num = int(match.group(1)) if match else i
            output_name = f"Random_AE_dim{dim_num}"
        
        # Check if already completed
        if output_name in completed_labels:
            logging.info(f"Skipping {output_name} (Already completed)")
            if output_dir:
                pos_file = os.path.join(output_dir, f"{output_name}_pos.tsv")
                neg_file = os.path.join(output_dir, f"{output_name}_neg.tsv")
                if os.path.exists(pos_file) or os.path.exists(neg_file):
                    try:
                        sig_count = 0
                        if os.path.exists(pos_file):
                            pos_df = pd.read_csv(pos_file, sep='\t')
                            pos_df.columns = pos_df.columns.str.strip().str.replace('<.*?>', '', regex=True)
                            if 'FDR q-val' in pos_df.columns:
                                sig_count += len(pos_df[pos_df['FDR q-val'] < 0.05])
                        if os.path.exists(neg_file):
                            neg_df = pd.read_csv(neg_file, sep='\t')
                            neg_df.columns = neg_df.columns.str.strip().str.replace('<.*?>', '', regex=True)
                            if 'FDR q-val' in neg_df.columns:
                                sig_count += len(neg_df[neg_df['FDR q-val'] < 0.05])
                        results.append({
                            'label': output_name,
                            'rnk_file': str(rnk_file),
                            'significant_pathways': sig_count,
                            'status': 'completed_skip'
                        })
                        significant_count += sig_count
                        continue
                    except Exception:
                        logging.warning(f"Failed to read existing results, re-running {output_name}")
        
        try:
            temp_output_dir = tempfile.mkdtemp(prefix=f"gsea_negative_control_")
            gsea_output_dir = os.path.join(temp_output_dir, "gsea_results")
            os.makedirs(gsea_output_dir, exist_ok=True)
            
            success = run_gsea_preranked(
                rnk_file=rnk_file,
                gene_set_file=gene_set_file,
                output_dir=gsea_output_dir,
                label=output_name,
                memory="4g",
                permutations=n_perm if n_perm else 100,
                min_size=15,
                max_size=500,
                seed=42,
                plot_top_x=0 
            )
            
            if success:
                result_pattern = f"{output_name}.GseaPreranked.*"
                result_dirs = list(Path(gsea_output_dir).glob(result_pattern))
                
                if result_dirs:
                    result_dir = result_dirs[0]
                    pos_files = list(result_dir.glob("gsea_report_for_na_pos_*.tsv"))
                    neg_files = list(result_dir.glob("gsea_report_for_na_neg_*.tsv"))
                    
                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)
                        if pos_files:
                            final_pos_file = os.path.join(output_dir, f"{output_name}_pos.tsv")
                            shutil.copy2(pos_files[0], final_pos_file)
                        if neg_files:
                            final_neg_file = os.path.join(output_dir, f"{output_name}_neg.tsv")
                            shutil.copy2(neg_files[0], final_neg_file)
                    
                    sig_count = 0
                    if pos_files:
                        pos_df = pd.read_csv(pos_files[0], sep='\t')
                        pos_df.columns = pos_df.columns.str.strip().str.replace('<.*?>', '', regex=True)
                        if 'FWER p-val' in pos_df.columns:
                            sig_count += len(pos_df[pos_df['FWER p-val'] < 0.05])
                    
                    if neg_files:
                        neg_df = pd.read_csv(neg_files[0], sep='\t')
                        neg_df.columns = neg_df.columns.str.strip().str.replace('<.*?>', '', regex=True)
                        if 'FWER p-val' in neg_df.columns:
                            sig_count += len(neg_df[neg_df['FWER p-val'] < 0.05])
                    
                    significant_count += sig_count
                    results.append({
                        'method': 'PCA' if is_pca else 'AutoEncoder',
                        'component': pc_num if is_pca else dim_num,
                        'significant_pathways': sig_count,
                        'output_name': output_name
                    })
                else:
                    logging.warning(f"GSEA succeeded but no result directory found: {result_pattern}")
            else:
                logging.warning(f"GSEA execution failed for: {rnk_file}")
            
            shutil.rmtree(temp_output_dir)
                
        except Exception as e:
            logging.error(f"Error processing {rnk_file}: {e}")
            continue
        
        if (i + 1) % 10 == 0:
            logging.info(f"Processed {i+1}/{len(rnk_files)} files")
    
    return {
        'total_components': len(rnk_files),
        'total_significant': significant_count,
        'results': results
    }


def main():
    parser = argparse.ArgumentParser(description='Negative Control Experiment: Verify specificity on random data')
    parser.add_argument('--dataset', type=str, default='scanb',
                        help='Reference dataset for dimensionality')
    parser.add_argument('--gene_set', type=str, default='GO',
                        choices=['GO', 'KEGG', 'REACTOME', 'C6'],
                        help='Gene set type')
    parser.add_argument('--dim', type=int, default=128,
                        help='AutoEncoder output dimensions')
    parser.add_argument('--n_components', type=int, default=128,
                        help='PCA components count')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--skip_save_random_data', action='store_true',
                        help='Skip saving random data files')
    parser.add_argument('--n_perm', type=int, default=100,
                        help='GSEA permutations count')
    parser.add_argument('--max_pca_components', type=int, default=None,
                        help='Max PCA components to process (for faster tests)')
    parser.add_argument('--skip_pca', action='store_true',
                        help='Skip PCA experiment')
    parser.add_argument('--skip_ae', action='store_true',
                        help='Skip AutoEncoder experiment')
    
    args = parser.parse_args()
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "results" / "figure1" / "figure1b_random_experiment" / f"negative_control_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / "negative_control.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info("="*80)
    logging.info("Negative Control Experiment")
    logging.info("="*80)
    logging.info(f"Reference Dataset: {args.dataset}")
    logging.info(f"Gene Set: {args.gene_set}")
    logging.info(f"AutoEncoder Dim: {args.dim}")
    logging.info(f"PCA Components: {args.n_components}")
    if args.max_pca_components:
        logging.info(f"Processing only Top {args.max_pca_components} PCA components")
    logging.info(f"GSEA Permutations: {args.n_perm}")
    logging.info(f"Seed: {args.seed}")
    logging.info(f"Output Directory: {output_dir}")
    logging.info("="*80)
    
    logging.info("\nStep 1: Fetching real data dimensions")
    try:
        n_samples_train, n_samples_test, n_genes, train_df, test_df = load_real_data_shape(args.dataset)
    except Exception as e:
        logging.error(f"Failed to load real data: {e}")
        return 1
    
    logging.info("\nStep 2: Generating random noise matrix")
    set_seed(args.seed)
    real_gene_names = list(train_df.columns)
    is_ensembl = any(is_ensembl_id(str(col)) for col in train_df.columns[:10])
    
    if is_ensembl:
        logging.info("Detected Ensembl ID format, converting to gene symbols...")
        temp_df = pd.DataFrame(index=train_df.index[:10], columns=train_df.columns)
        temp_df_converted = convert_ensembl_to_symbol(temp_df)
        real_gene_names = list(temp_df_converted.columns)
        logging.info(f"Conversion complete: {len(real_gene_names)} symbols from {len(train_df.columns)} Ensembl IDs")
    
    random_train_df = generate_random_noise_matrix(n_samples_train, n_genes, seed=args.seed, real_gene_names=real_gene_names)
    random_test_df = generate_random_noise_matrix(n_samples_test, n_genes, seed=args.seed + 1, real_gene_names=real_gene_names)
    
    common_genes = sorted(list(set(random_train_df.columns) & set(random_test_df.columns)))
    random_train_df = random_train_df[common_genes]
    random_test_df = random_test_df[common_genes]
    
    logging.info(f"Random Train Shape: {random_train_df.shape}")
    logging.info(f"Random Test Shape: {random_test_df.shape}")
    
    random_data_dir = output_dir / "random_data"
    random_data_dir.mkdir(exist_ok=True)
    
    if not args.skip_save_random_data:
        try:
            random_train_df.to_csv(random_data_dir / "random_train.csv", index=True)
            random_test_df.to_csv(random_data_dir / "random_test.csv", index=True)
            logging.info(f"Random data saved to: {random_data_dir}")
        except OSError as e:
            logging.warning(f"Could not save random data: {e}")
    
    pca_gsea_results = None
    if not args.skip_pca:
        logging.info("\n" + "="*80)
        logging.info("Experiment A: PCA + GSEA (Random Data)")
        logging.info("="*80)
        
        actual_n_components = args.max_pca_components if args.max_pca_components else args.n_components
        pca_rnk_files = run_pca_on_random_data(random_train_df, random_test_df, output_dir, n_components=args.n_components)
        
        if args.max_pca_components:
            pca_rnk_files = pca_rnk_files[:actual_n_components]
        
        pca_gsea_results = run_gsea_on_rnk_files(
            pca_rnk_files, 
            gene_set_type=args.gene_set,
            output_dir=output_dir / "gsea_pca_results",
            is_pca=True,
            n_perm=args.n_perm
        )
    
    ae_gsea_results = None
    if not args.skip_ae:
        logging.info("\n" + "="*80)
        logging.info("Experiment B: AutoEncoder + GSEA (Random Data)")
        logging.info("="*80)
        
        import copy
        temp_config = copy.deepcopy(config)
        temp_config['testing_dataset_name'] = 'Metabric'
        
        temp_train_file = random_data_dir / "random_train.csv"
        temp_test_file = random_data_dir / "random_test.csv"
        
        if not temp_train_file.exists():
            random_train_df.to_csv(temp_train_file, index=True)
        if not temp_test_file.exists():
            random_test_df.to_csv(temp_test_file, index=True)
        
        temp_config['bulk_path'] = str(temp_train_file)
        temp_config['bulk_test_path'] = str(temp_test_file)
        temp_config['encoder_output_dim'] = [args.dim]
        temp_config['num_epochs'] = 100 
        temp_config['batch_size'] = [256] if isinstance(config['batch_size'], list) else 256
        temp_config['bulk_id_path'] = None  
        
        random_train_df_final = random_train_df.sort_index(axis=1)
        random_test_df_final = random_test_df.sort_index(axis=1)
        random_train_df_final.to_csv(temp_train_file, index=True)
        random_test_df_final.to_csv(temp_test_file, index=True)
        
        try:
            ae_result_dir = output_dir / "auto_encoder_results"
            ae_result_dir.mkdir(exist_ok=True)
            
            original_dataset_name = config['testing_dataset_name']
            original_bulk_path = config.get('bulk_path', '')
            original_bulk_test_path = config.get('bulk_test_path', '')
            original_bulk_id_path = config.get('bulk_id_path', '')
            
            try:
                config['testing_dataset_name'] = temp_config['testing_dataset_name']
                config['bulk_path'] = temp_config['bulk_path']
                config['bulk_test_path'] = temp_config['bulk_test_path']
                config['bulk_id_path'] = temp_config['bulk_id_path']
                config['encoder_output_dim'] = temp_config['encoder_output_dim']
                config['num_epochs'] = temp_config['num_epochs']
                config['batch_size'] = temp_config['batch_size']
                
                logging.info("Training AutoEncoder model...")
                encoder, test_embedding_df, test_bulk_original = train_auto_encoder_model(temp_config, ae_result_dir)
            finally:
                config['testing_dataset_name'] = original_dataset_name
                if original_bulk_path: config['bulk_path'] = original_bulk_path
                if original_bulk_test_path: config['bulk_test_path'] = original_bulk_test_path
                if original_bulk_id_path: config['bulk_id_path'] = original_bulk_id_path
            
            logging.info("Calculating correlations and generating RNK files...")
            correlations_dir = ae_result_dir / "correlations"
            correlations_dir.mkdir(exist_ok=True)
            
            correlation_lists = calculate_pearson_correlation(test_embedding_df, test_bulk_original)
            
            rnk_files = []
            for dim, corr_df in enumerate(correlation_lists):
                output_file = correlations_dir / f"Random_dimension_{dim}_correlation.rnk"
                corr_df_sorted = corr_df.sort_values('correlation', ascending=False)
                if 'gene' in corr_df_sorted.columns and 'correlation' in corr_df_sorted.columns:
                    corr_df_sorted[['gene', 'correlation']].to_csv(output_file, sep='\t', index=False, header=False)
                    rnk_files.append(str(output_file))
            
            ae_gsea_results = run_gsea_on_rnk_files(
                rnk_files,
                gene_set_type=args.gene_set,
                output_dir=ae_result_dir / "gsea_results",
                is_pca=False,
                n_perm=args.n_perm
            )
            
        except Exception as e:
            logging.error(f"AutoEncoder experiment failed: {e}")
            ae_gsea_results = None
    
    logging.info("\n" + "="*80)
    logging.info("Summary of Results")
    logging.info("="*80)
    
    summary = {
        'experiment_type': 'Negative Control',
        'dataset': args.dataset,
        'gene_set': args.gene_set,
        'random_seed': args.seed,
        'data_shape': {'train': n_samples_train, 'test': n_samples_test, 'genes': n_genes}
    }
    
    if pca_gsea_results:
        summary['pca'] = {'hits': pca_gsea_results['total_significant']}
        logging.info(f"PCA Significant Pathways: {pca_gsea_results['total_significant']}")
    
    if ae_gsea_results:
        summary['ae'] = {'hits': ae_gsea_results['total_significant']}
        logging.info(f"AutoEncoder Significant Pathways: {ae_gsea_results['total_significant']}")
    
    import json
    with open(output_dir / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    logging.info("\nConclusions:")
    if pca_gsea_results and ae_gsea_results:
        pca_sig = pca_gsea_results['total_significant']
        ae_sig = ae_gsea_results['total_significant']
        if pca_sig > 0 and ae_sig == 0:
            logging.info("Result: AutoEncoder maintained specificity (0 hits), whereas PCA produced false positives.")
        elif pca_sig == 0 and ae_sig == 0:
            logging.info("Result: Both methods maintained specificity.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
