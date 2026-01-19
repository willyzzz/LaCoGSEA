#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

# -*- coding: utf-8 -*-
"""
Prepare TCGA Lung Cancer Data: Split train/test sets and save label files.
"""

import argparse
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Prepare TCGA Lung Cancer Data: Split train/test sets')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Train set ratio (default: 0.8, deprecated: using all data now)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42, deprecated)')
    parser.add_argument('--expr_path', type=str, default='TCGA_Pan_cancer/tcga_lung_expression_clean.csv',
                       help='Expression matrix path')
    parser.add_argument('--clin_path', type=str, default='TCGA_Pan_cancer/tcga_lung_clinical_clean.csv',
                       help='Clinical data path')
    
    args = parser.parse_args()
    
    # Read data
    print("Loading expression matrix...")
    expr_df = pd.read_csv(args.expr_path, index_col=0)
    print(f"Expression matrix shape: {expr_df.shape}")
    
    print("Loading clinical data...")
    clin_df = pd.read_csv(args.clin_path, index_col=0)
    print(f"Clinical data shape: {clin_df.shape}")
    
    # Check Subtype column
    if 'Subtype' not in clin_df.columns:
        raise ValueError("Clinical data must contain 'Subtype' column (LUAD/LUSC)")
    
    # Align samples (intersection)
    common_samples = expr_df.index.intersection(clin_df.index)
    print(f"Common samples: {len(common_samples)}")
    
    expr_aligned = expr_df.loc[common_samples]
    clin_aligned = clin_df.loc[common_samples]
    
    # Check Subtype distribution
    subtype_counts = clin_aligned['Subtype'].value_counts()
    print(f"Subtype distribution:\n{subtype_counts}")
    
    # No split, use all data for both training and testing
    print(f"\nUsing all {len(common_samples)} samples for both training and testing")
    print(f"\nSubtype distribution:")
    print(clin_aligned['Subtype'].value_counts())
    
    # Save data (train and test use the same data)
    output_dir = Path('../../data/TCGA_Pan_cancer')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / 'tcga_lung_train.csv'
    test_path = output_dir / 'tcga_lung_test.csv'
    
    print(f"\nSaving data to: {train_path} (train) and {test_path} (test, same as train)")
    expr_aligned.to_csv(train_path)
    expr_aligned.to_csv(test_path)
    
    # Save label file (including all sample Subtypes)
    labels_df = pd.DataFrame({
        'Subtype': clin_aligned['Subtype']
    })
    labels_path = output_dir / 'tcga_lung_labels.csv'
    print(f"Saving labels to: {labels_path}")
    labels_df.to_csv(labels_path)
    
    print("\nDone! Data preparation complete.")
    print(f"Train set: {train_path}")
    print(f"Test set: {test_path}")
    print(f"Labels: {labels_path}")

if __name__ == '__main__':
    main()
