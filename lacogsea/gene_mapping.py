#!/usr/bin/env python
# coding: utf-8
"""
Ensembl ID to Gene Symbol conversion module
Uses locally cached mapping files to avoid repeated downloads
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import urllib.request
import gzip
import logging

# Set UTF-8 encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def is_ensembl_id(gene_id):
    """
    Check if a string is in Ensembl ID format.
    Format: Starts with ENS, followed by species code (e.g., ENSG, ENSMUS) and numbers.
    Example: ENSG00000139618.12 or ENSG00000139618
    """
    if not isinstance(gene_id, str):
        return False
    gene_id = gene_id.strip()
    # Ensembl IDs usually start with ENS, followed by species code (e.g., ENSG, ENSMUS) and numbers.
    # Some IDs have version numbers (with a dot), some don't.
    if not gene_id.startswith('ENS'):
        return False
    # Check if it's standard Ensembl ID format: ENS + 1 letter (species code) + 11 digits + optional version
    import re
    # Match: ENS + species code (1 letter, e.g., G=human, M=mouse) + 11 digits + optional .version
    pattern = r'^ENS[A-Z]\d{11}(\.\d+)?$'
    return bool(re.match(pattern, gene_id))


def is_gene_symbol(gene_id):
    """
    Check if a string is in Gene Symbol format.
    Gene Symbols are usually alphanumeric and don't start with ENS.
    """
    if not isinstance(gene_id, str):
        return False
    gene_id = gene_id.strip()
    return not gene_id.startswith('ENS') and len(gene_id) > 0


def download_ensembl_mapping(force_download=False):
    """
    Download GTF file from Ensembl FTP and parse it to generate Ensembl ID to Gene Symbol mapping.
    Save to local cache file.
    """
    cache_dir = Path(".gene_mapping_cache")
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / "ensembl_to_symbol.csv"
    
    # Return if cache exists and not forcing download
    if cache_file.exists() and not force_download:
        logging.info(f"Using existing mapping cache: {cache_file}")
        return str(cache_file)
    
    logging.info("Downloading Ensembl GTF file...")
    
    # Use Human genome GTF file (GRCh38)
    gtf_url = "https://ftp.ensembl.org/pub/release-111/gtf/homo_sapiens/Homo_sapiens.GRCh38.111.gtf.gz"
    temp_gtf = cache_dir / "Homo_sapiens.GRCh38.111.gtf.gz"
    
    try:
        # Download GTF file
        logging.info(f"Downloading from {gtf_url}...")
        urllib.request.urlretrieve(gtf_url, temp_gtf)
        logging.info("Download completed. Parsing GTF file...")
        
        # Parse GTF file
        mapping_dict = {}
        with gzip.open(temp_gtf, 'rt', encoding='utf-8') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                fields = line.strip().split('\t')
                if len(fields) < 9:
                    continue
                attributes = fields[8]
                
                # Parse attribute fields to extract gene_id and gene_name
                gene_id = None
                gene_name = None
                
                for attr in attributes.split(';'):
                    attr = attr.strip()
                    if attr.startswith('gene_id'):
                        gene_id = attr.split('"')[1] if '"' in attr else None
                    elif attr.startswith('gene_name'):
                        gene_name = attr.split('"')[1] if '"' in attr else None
                
                if gene_id and gene_name:
                    # Keep only the first mapping (avoid duplicates)
                    if gene_id not in mapping_dict:
                        mapping_dict[gene_id] = gene_name
        
        # Save to CSV
        mapping_df = pd.DataFrame([
            {'ensembl_id': k, 'gene_symbol': v}
            for k, v in mapping_dict.items()
        ])
        mapping_df.to_csv(cache_file, index=False)
        logging.info(f"Saved {len(mapping_df)} mappings to {cache_file}")
        
        # Delete temporary GTF file
        if temp_gtf.exists():
            temp_gtf.unlink()
        
        return str(cache_file)
    
    except Exception as e:
        logging.error(f"Error downloading or parsing Ensembl GTF: {str(e)}")
        raise


def get_ensembl_to_symbol_mapping(force_download=False):
    """
    Get Ensembl ID to Gene Symbol mapping.
    Prioritize local cache.
    """
    cache_file = Path(".gene_mapping_cache") / "ensembl_to_symbol.csv"
    
    # Download if cache doesn't exist
    if not cache_file.exists():
        download_ensembl_mapping(force_download=force_download)
    
    # Read mapping
    mapping_df = pd.read_csv(cache_file)
    mapping_dict = dict(zip(mapping_df['ensembl_id'], mapping_df['gene_symbol']))
    
    return mapping_dict


def convert_ensembl_to_symbol(df):
    """
    Convert DataFrame column names from Ensembl ID to Gene Symbol.
    
    Args:
        df: DataFrame with Ensembl IDs as column names.
    
    Returns:
        DataFrame with Gene Symbols as column names.
    """
    logging.info("Loading Ensembl to Gene Symbol mapping from local file...")
    
    # Get mapping
    cache_file = Path(".gene_mapping_cache") / "ensembl_to_symbol.csv"
    if not cache_file.exists():
        logging.info("Mapping cache not found, downloading...")
        download_ensembl_mapping()
    
    logging.info(f"Loading Ensembl mapping from local cache: {cache_file.absolute()}")
    mapping_df = pd.read_csv(cache_file)
    mapping_dict = dict(zip(mapping_df['ensembl_id'], mapping_df['gene_symbol']))
    logging.info(f"Loaded {len(mapping_dict)} mappings from cache")
    
    # Convert column names
    old_columns = df.columns.tolist()
    new_columns = []
    unmapped = []
    
    for col in old_columns:
        # Handle versioned Ensembl IDs (e.g., ENSG00000139618.12 -> ENSG00000139618)
        col_clean = str(col).split('.')[0]
        
        if col_clean in mapping_dict:
            new_columns.append(mapping_dict[col_clean])
        else:
            # Keep original column name if no mapping found
            new_columns.append(str(col))
            unmapped.append(str(col))
    
    # Create new DataFrame
    df_new = df.copy()
    df_new.columns = new_columns
    
    # If there are duplicate gene symbols, aggregate by mean
    if df_new.columns.duplicated().any():
        logging.warning(f"Found duplicate gene symbols after mapping, aggregating...")
        df_new = df_new.groupby(level=0, axis=1).mean()
    
    if unmapped:
        logging.warning(f"Unmapped {len(unmapped)} genes: {unmapped[:10]}...")
    
    mapped_count = len(old_columns) - len(unmapped)
    logging.info(f"Conversion completed: {mapped_count} mapped, {len(unmapped)} unmapped")
    logging.info(f"Final gene count: {len(df_new.columns)}")
    
    return df_new
