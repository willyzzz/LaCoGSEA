#!/usr/bin/env python
from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

# -*- coding: utf-8 -*-
"""
Figure 5 Data Preparation: Load benchmarking datasets and extract group labels (Original Figure 2)
"""

import argparse
import pandas as pd
import numpy as np
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def load_gse10846_data(
    expr_path: str = "results/figure2/figure2_Dataset/DLBCL_data/data/GSE10846/GSE10846_clean_log2.csv",
    clin_path: str = "results/figure2/figure2_Dataset/DLBCL_data/data/GSE10846/GSE10846_clinical.csv",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load GSE10846 (DLBCL) data and extract ABC vs GCB labels.
    
    Returns:
        expr_df: Expression matrix (samples x genes)
        labels: Series with 'ABC' or 'GCB' labels
    """
    logging.info("Loading GSE10846 data...")
    
    # Load expression matrix
    expr_df = pd.read_csv(expr_path, index_col=0)
    logging.info(f"Expression matrix shape: {expr_df.shape}")
    
    # Load clinical data
    clin_df = pd.read_csv(clin_path, index_col=0)
    logging.info(f"Clinical data shape: {clin_df.shape}")
    
    # Extract labels from "Final microarray diagnosis" column
    diag_col = "characteristics_ch1.6.Clinical info"
    if diag_col not in clin_df.columns:
        # Try to find the column
        possible_cols = [c for c in clin_df.columns if 'diagnosis' in c.lower() or 'final' in c.lower()]
        if possible_cols:
            diag_col = possible_cols[0]
            logging.info(f"Using column: {diag_col}")
        else:
            raise ValueError(f"Could not find diagnosis column. Available columns: {list(clin_df.columns)[:10]}")
    
    # Extract ABC/GCB labels
    labels = []
    for sample_id in expr_df.index:
        if sample_id in clin_df.index:
            diag = str(clin_df.loc[sample_id, diag_col]).upper()
            if 'ABC' in diag and 'DLBCL' in diag:
                labels.append('ABC')
            elif 'GCB' in diag and 'DLBCL' in diag:
                labels.append('GCB')
            else:
                # Skip Unclassified or NA
                labels.append(np.nan)
        else:
            labels.append(np.nan)
    
    labels_series = pd.Series(labels, index=expr_df.index, name='Subtype')
    labels_series = labels_series[labels_series.notna()]
    
    # Filter expression to labeled samples
    expr_df = expr_df.loc[labels_series.index]
    
    logging.info(f"GSE10846: {len(expr_df)} samples, {labels_series.value_counts().to_dict()}")
    
    return expr_df, labels_series


def load_gse48350_data(
    expr_path: str = "results/figure2/figure2_Dataset/GSE48350_data/data/GSE48350/GSE48350_clean_log2.csv",
    soft_path: str = "results/figure2/figure2_Dataset/GSE48350_data/data/GSE48350/GSE48350_family.soft.gz",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load GSE48350 (AD) data and extract AD vs Control labels.
    
    Returns:
        expr_df: Expression matrix (samples x genes)
        labels: Series with 'AD' or 'Control' labels
    """
    logging.info("Loading GSE48350 data...")
    
    expr_df = pd.read_csv(expr_path, index_col=0)
    logging.info(f"Expression matrix shape: {expr_df.shape}")
    
    # Extract labels from GEO metadata
    try:
        import GEOparse
        gse = GEOparse.get_GEO(filepath=soft_path, silent=True)
        
        labels = []
        for sample_id in expr_df.index:
            if sample_id in gse.gsms:
                gsm = gse.gsms[sample_id]
                disease_state = None
                if 'characteristics_ch1' in gsm.metadata:
                    chars = gsm.metadata['characteristics_ch1']
                    if isinstance(chars, list):
                        for char in chars:
                            char_str = str(char).upper()
                            if ', C' in char_str or 'CONTROL' in char_str:
                                disease_state = 'Control'
                                break
                            elif ', AD' in char_str or 'ALZHEIMER' in char_str or 'AD' in char_str:
                                disease_state = 'AD'
                                break
                
                if disease_state is None and 'source_name_ch1' in gsm.metadata:
                    source = gsm.metadata['source_name_ch1']
                    if isinstance(source, list) and len(source) > 0:
                        source_str = str(source[0]).upper()
                        if 'ALZHEIMER' in source_str or 'AD' in source_str:
                            disease_state = 'AD'
                        elif 'CONTROL' in source_str or 'NORMAL' in source_str:
                            disease_state = 'Control'
                
                if disease_state == 'AD':
                    labels.append('AD')
                elif disease_state == 'Control':
                    labels.append('Control')
                else:
                    labels.append(np.nan)
            else:
                labels.append(np.nan)
        
        labels_series = pd.Series(labels, index=expr_df.index, name='Subtype')
        labels_series = labels_series[labels_series.notna()]
        expr_df = expr_df.loc[labels_series.index]
        
        logging.info(f"GSE48350: {len(expr_df)} samples, {labels_series.value_counts().to_dict()}")
    except Exception as e:
        logging.warning(f"Could not extract labels from GEO metadata: {e}")
        import traceback
        traceback.print_exc()
        logging.warning("Using placeholder labels - will need manual correction")
        labels_series = pd.Series(['AD'] * len(expr_df), index=expr_df.index, name='Subtype')
    
    return expr_df, labels_series


def load_gse11375_data(
    expr_path: str = "results/figure2/figure2_Dataset/GSE11375_data/data/GSE11375/GSE11375_clean_log2.csv",
    series_matrix_path: str = "results/figure2/figure2_Dataset/GSE11375_data/data/GSE11375/GSE11375_series_matrix.txt.gz",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load GSE11375 (Trauma) data and extract Trauma vs Healthy labels.
    
    Returns:
        expr_df: Expression matrix (samples x genes)
        labels: Series with 'Trauma' or 'Healthy' labels
    """
    logging.info("Loading GSE11375 data...")
    
    expr_df = pd.read_csv(expr_path, index_col=0)
    logging.info(f"Expression matrix shape: {expr_df.shape}")
    
    labels = []
    sample_ids = list(expr_df.index)
    
    try:
        import gzip
        
        geo_accession_line = None
        disposition_line = None
        title_line = None
        
        with gzip.open(series_matrix_path, 'rt', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.startswith('!Sample_geo_accession'):
                    geo_accession_line = line.strip()
                elif 'final disposition' in line.lower():
                    disposition_line = line.strip()
                elif line.startswith('!Sample_title'):
                    title_line = line.strip()
                if geo_accession_line and disposition_line:
                    break
        
        if not geo_accession_line or not disposition_line:
            raise ValueError("Could not find required metadata lines in series_matrix file")
        
        geo_accessions = geo_accession_line.split('\t')[1:]  
        geo_accessions = [acc.strip('"') for acc in geo_accessions]
        
        dispositions = disposition_line.split('\t')[1:]  
        dispositions = [disp.strip('"').lower() for disp in dispositions]
        
        gsm_to_disposition = {}
        for gsm, disp in zip(geo_accessions, dispositions):
            if gsm.startswith('GSM'):
                if 'control' in disp or 'healthy' in disp or 'volunteer' in disp:
                    gsm_to_disposition[gsm] = 'Healthy'
                elif 'trauma' in disp or 'patient' in disp or 'injury' in disp:
                    gsm_to_disposition[gsm] = 'Trauma'
        
        if title_line:
            titles = title_line.split('\t')[1:]
            titles = [t.strip('"').lower() for t in titles]
            for gsm, title in zip(geo_accessions, titles):
                if gsm.startswith('GSM'):
                    if gsm not in gsm_to_disposition:
                        if 'ctl' in title or 'control' in title:
                            gsm_to_disposition[gsm] = 'Healthy'
                        elif 'pt' in title or 'trauma' in title or 'patient' in title:
                            gsm_to_disposition[gsm] = 'Trauma'
                    elif gsm_to_disposition.get(gsm) is None:
                        if 'ctl' in title:
                            gsm_to_disposition[gsm] = 'Healthy'
                        elif 'pt' in title:
                            gsm_to_disposition[gsm] = 'Trauma'
        
        logging.info(f"Parsed {len(gsm_to_disposition)} sample labels from metadata")
        
        for sample_id in sample_ids:
            if sample_id in gsm_to_disposition:
                labels.append(gsm_to_disposition[sample_id])
            else:
                labels.append(np.nan)
        
    except Exception as e:
        logging.warning(f"Error parsing series_matrix file: {e}")
        import traceback
        traceback.print_exc()
        labels = [np.nan] * len(sample_ids)
    
    labels_series = pd.Series(labels, index=expr_df.index, name='Subtype')
    
    if labels_series.notna().sum() > 0:
        labels_series = labels_series[labels_series.notna()]
        expr_df = expr_df.loc[labels_series.index]
        logging.info(f"GSE11375: {len(expr_df)} samples, {labels_series.value_counts().to_dict()}")
    else:
        logging.warning("GSE11375: Could not extract labels automatically.")
        labels_series = pd.Series([], dtype=str, name='Subtype')
    
    return expr_df, labels_series


def load_gse126848_data(
    expr_path: str = "results/figure2/figure2_Dataset/GSE126848_data/GSE126848.csv",
    meta_path: str = "results/figure2/figure2_Dataset/GSE126848_data/metadata.csv",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load GSE126848 (Liver Disease) data and extract NASH vs Healthy labels.
    
    Returns:
        expr_df: Expression matrix (samples x genes)
        labels: Series with 'NASH' or 'Healthy' labels
    """
    logging.info("Loading GSE126848 data...")
    
    expr_df = pd.read_csv(expr_path, index_col=0)
    if expr_df.shape[1] > 25000:
         logging.info("Assuming genes are columns in GSE126848, no transpose needed.")
    
    logging.info(f"Expression matrix shape: {expr_df.shape}")
    
    meta_df = pd.read_csv(meta_path, index_col=0)
    logging.info(f"Metadata shape: {meta_df.shape}")
    
    disease_col = "characteristics_ch1.2.disease"
    desc_col = "description"
    
    desc_to_disease = {}
    for idx, row in meta_df.iterrows():
        disease = str(row[disease_col]).lower()
        desc = str(row[desc_col])
        if 'nash' in disease:
            desc_to_disease[desc] = 'NASH'
        elif 'healthy' in disease:
            desc_to_disease[desc] = 'Healthy'
            
    labels = []
    sample_ids = []
    
    for sample_id in expr_df.index:
        sid_str = str(sample_id).lstrip('0')
        if sid_str in desc_to_disease:
            labels.append(desc_to_disease[sid_str])
            sample_ids.append(sample_id)
        elif str(sample_id) in desc_to_disease:
            labels.append(desc_to_disease[str(sample_id)])
            sample_ids.append(sample_id)
            
    labels_series = pd.Series(labels, index=sample_ids, name='Subtype')
    expr_df = expr_df.loc[labels_series.index]
    logging.info(f"GSE126848: {len(expr_df)} samples, {labels_series.value_counts().to_dict()}")
    
    return expr_df, labels_series


def load_gse116250_data(
    expr_path: str = "results/figure2/figure2_Dataset/GSE116250_data/GSE116250_Data/GSE116250_X_input_ae.csv",
    labels_path: str = "results/figure2/figure2_Dataset/GSE116250_data/GSE116250_Data/GSE116250_y_labels.csv",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load GSE116250 (Heart Failure) data and extract DCM vs Non-Failing labels.
    """
    logging.info("Loading GSE116250 data...")
    
    expr_df = pd.read_csv(expr_path, index_col=0)
    labels_df = pd.read_csv(labels_path, index_col=0)
    
    labels_series = labels_df['Group']
    labels_series.name = 'Subtype'
    
    common_samples = expr_df.index.intersection(labels_series.index)
    expr_df = expr_df.loc[common_samples]
    labels_series = labels_series.loc[common_samples]
    
    logging.info(f"GSE116250: {len(expr_df)} samples, {labels_series.value_counts().to_dict()}")
    
    return expr_df, labels_series


def save_prepared_data(
    expr_df: pd.DataFrame,
    labels: pd.Series,
    dataset_name: str,
    output_dir: Path,
) -> None:
    """Save prepared expression matrix and labels."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    expr_path = output_dir / f"{dataset_name}_expression.csv"
    labels_path = output_dir / f"{dataset_name}_labels.csv"
    
    expr_df.to_csv(expr_path)
    labels.name = 'Subtype'
    labels.to_frame().to_csv(labels_path)
    
    logging.info(f"Saved {dataset_name} data:")
    logging.info(f"  Expression: {expr_path}")
    logging.info(f"  Labels: {labels_path}")


def main():
    parser = argparse.ArgumentParser(description='Prepare Figure 5 (Benchmarking) datasets')
    parser.add_argument('--datasets', nargs='+', default=['GSE10846', 'GSE48350', 'GSE11375', 'GSE126848', 'GSE116250'],
                       help='Datasets to prepare')
    parser.add_argument('--output_dir', type=str, default='figure2_prepared_data',
                       help='Output directory for prepared data')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_loaders = {
        'GSE10846': load_gse10846_data,
        'GSE48350': load_gse48350_data,
        'GSE11375': load_gse11375_data,
        'GSE126848': load_gse126848_data,
        'GSE116250': load_gse116250_data,
    }
    
    for dataset_name in args.datasets:
        if dataset_name not in dataset_loaders:
            logging.warning(f"Unknown dataset: {dataset_name}, skipping")
            continue
        
        try:
            loader = dataset_loaders[dataset_name]
            expr_df, labels = loader()
            save_prepared_data(expr_df, labels, dataset_name, output_dir)
        except Exception as e:
            logging.error(f"Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
