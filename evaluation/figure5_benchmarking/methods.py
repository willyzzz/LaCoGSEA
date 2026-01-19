#!/usr/bin/env python
from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

# -*- coding: utf-8 -*-
"""
Figure 2 Methods: PCA and AE based analysis.
Implements:
1. PCA + Correlation
2. PCA + Weights
3. AE + Correlation
4. AE + Attribution (DeepLIFT/SHAP)
"""


import logging
import os
from typing import Optional, List, Dict, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Import project modules
from core.evaluation import calculate_pearson_correlation
from core.dataset_input import set_seed
from core.model_structure import Encoder, Decoder
from core.barlow_config import config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def save_rnk_file(
    gene_names: List[str],
    scores: np.ndarray,
    output_path: Path,
) -> None:
    """Save gene scores to RNK file."""
    df = pd.DataFrame({'gene': gene_names, 'score': scores})
    df = df.sort_values('score', ascending=False)
    df.to_csv(output_path, sep='\t', index=False, header=False)


def run_pca_correlation(
    expr_df: pd.DataFrame,
    n_components: int,
    output_dir: Path,
    dataset_name: str,
) -> List[Path]:
    """
    Run PCA and calculate correlation between PCs and original genes.
    
    Args:
        expr_df: Expression matrix (samples x genes)
        n_components: Number of PCA components
        output_dir: Output directory
        dataset_name: Dataset name
        
    Returns:
        List of generated RNK file paths
    """
    logging.info(f"Running PCA + Correlation for {dataset_name}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Standardize
    scaler = StandardScaler()
    expr_scaled = scaler.fit_transform(expr_df)
    
    # PCA
    n_components = min(n_components, min(expr_df.shape) - 1)
    pca = PCA(n_components=n_components, random_state=42)
    pcs = pca.fit_transform(expr_scaled)
    
    # Create DataFrame for PCs
    pc_df = pd.DataFrame(
        pcs, 
        index=expr_df.index, 
        columns=[f'PC_{i}' for i in range(n_components)]
    )
    
    # Calculate correlations
    # Uses existing function from evaluation.py
    # Returns list of DataFrames, one per dimension, sorted by correlation
    # Each DF has columns: ['gene', 'correlation']
    correlation_lists = calculate_pearson_correlation(pc_df, expr_df)
    
    rnk_files = []
    for dim, corr_df in enumerate(correlation_lists):
        rnk_path = output_dir / f"{dataset_name}_PCA_Corr_dim{dim}.rnk"
        
        # Save RNK
        corr_df[['gene', 'correlation']].to_csv(
            rnk_path, sep='\t', index=False, header=False
        )
        rnk_files.append(rnk_path)
        
    logging.info(f"Generated {len(rnk_files)} RNK files for PCA + Correlation")
    return rnk_files


def run_pca_weights(
    expr_df: pd.DataFrame,
    n_components: int,
    output_dir: Path,
    dataset_name: str,
) -> List[Path]:
    """
    Run PCA and use component loadings (weights) directly.
    """
    logging.info(f"Running PCA + Weights for {dataset_name}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Standardize
    scaler = StandardScaler()
    expr_scaled = scaler.fit_transform(expr_df)
    
    # PCA
    n_components = min(n_components, min(expr_df.shape) - 1)
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(expr_scaled)
    
    # Components matrix: (n_components, n_features)
    components = pca.components_
    gene_names = expr_df.columns.tolist()
    
    rnk_files = []
    for dim in range(n_components):
        loadings = components[dim]
        rnk_path = output_dir / f"{dataset_name}_PCA_Weight_dim{dim}.rnk"
        
        save_rnk_file(gene_names, loadings, rnk_path)
        rnk_files.append(rnk_path)
        
    logging.info(f"Generated {len(rnk_files)} RNK files for PCA + Weights")
    return rnk_files


def train_ae(
    expr_df: pd.DataFrame, 
    params: Dict, 
    model_path: Optional[Path] = None,
    eval_expr_df: Optional[pd.DataFrame] = None
):
    """
    Train or load Autoencoder.
    params: dict with keys 'hidden_dims', 'encoder_output_dim', 'batch_size', 'num_epochs', 'learning_rate'
    model_path: Path to save/load model state_dict. If exists, loads instead of training.
    eval_expr_df: Optional dataframe for evaluation/embedding generation if different from training data.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    # Extract params
    input_dim = expr_df.shape[1]
    hidden_dims = params.get('hidden_dims', [1024, 512]) 
    output_dim = params.get('encoder_output_dim', 64)
    batch_size = params.get('batch_size', 256)
    epochs = params.get('num_epochs', 50)
    lr = params.get('learning_rate', 1e-3)
    
    # Model Structure
    encoder = Encoder(input_dim, hidden_dims, output_dim).to(device)
    decoder = Decoder(output_dim, hidden_dims, input_dim).to(device)
    
    # Check for existing model
    model_loaded = False
    if model_path is not None and model_path.exists():
        try:
            logging.info(f"Loading pre-trained AE model from {model_path}")
            checkpoint = torch.load(model_path, map_location=device)
            encoder.load_state_dict(checkpoint['encoder'])
            decoder.load_state_dict(checkpoint['decoder'])
            model_loaded = True
        except Exception as e:
            logging.warning(f"Failed to load model from {model_path}, retraining. Error: {e}")
            model_loaded = False
            
    if not model_loaded:
        logging.info(f"Training AE from scratch for {epochs} epochs...")
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()), 
            lr=lr
        )
        criterion = torch.nn.MSELoss()
        
        # Data
        tensor_data = torch.FloatTensor(expr_df.values)
        dataset = torch.utils.data.TensorDataset(tensor_data, tensor_data) 
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Train
        encoder.train()
        decoder.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                x, _ = batch
                x = x.to(device)
                
                optimizer.zero_grad()
                encoded = encoder(x)
                decoded = decoder(encoded)
                loss = criterion(decoded, x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        # Save model
        if model_path is not None:
            logging.info(f"Saving trained model to {model_path}")
            torch.save({
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict()
            }, model_path)
            
    # Get embeddings (Evaluation Mode)
    encoder.eval()
    
    # Determine which data to embed
    # If eval_expr_df is provided (e.g. bootstrap sample), use it.
    # Otherwise use the training data (expr_df)
    target_df = eval_expr_df if eval_expr_df is not None else expr_df
    
    tensor_data = torch.FloatTensor(target_df.values)
    dataset = torch.utils.data.TensorDataset(tensor_data, tensor_data)
    
    with torch.no_grad():
        all_embeddings = []
        # Process in batches to avoid OOM
        eval_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for batch in eval_loader:
             x, _ = batch
             x = x.to(device)
             emb = encoder(x)
             all_embeddings.append(emb.cpu().numpy())
             
        embeddings = np.vstack(all_embeddings)
        
    embedding_df = pd.DataFrame(
        embeddings,
        index=target_df.index,
        columns=[f'dim_{i}' for i in range(output_dim)]
    )
    
    return encoder, embedding_df


def run_ae_correlation(
    encoder: Encoder,
    embedding_df: pd.DataFrame,
    expr_df: pd.DataFrame,
    output_dir: Path,
    dataset_name: str,
) -> List[Path]:
    """
    Calculate correlation between AE latent dimensions and input genes.
    """
    logging.info(f"Running AE + Correlation for {dataset_name}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    correlation_lists = calculate_pearson_correlation(embedding_df, expr_df)
    
    rnk_files = []
    for dim, corr_df in enumerate(correlation_lists):
        rnk_path = output_dir / f"{dataset_name}_AE_Corr_dim{dim}.rnk"
        
        corr_df[['gene', 'correlation']].to_csv(
            rnk_path, sep='\t', index=False, header=False
        )
        rnk_files.append(rnk_path)
        
    logging.info(f"Generated {len(rnk_files)} RNK files for AE + Correlation")
    return rnk_files


def run_ae_attribution(
    encoder: Encoder,
    expr_df: pd.DataFrame,
    output_dir: Path,
    dataset_name: str,
    method: str = 'deeplift',  # 'deeplift' or 'shap'
    baseline_type: str = 'mean', # 'mean' or 'zero'
) -> List[Path]:
    """
    Run DeepLIFT or SHAP to get feature importance.
    """
    logging.info(f"Running AE + {method} ({baseline_type} baseline) for {dataset_name}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = next(encoder.parameters()).device
    encoder.eval()
    
    # Input data tensor
    input_tensor = torch.FloatTensor(expr_df.values).to(device)
    
    # Background for DeepLIFT/SHAP
    if baseline_type == 'zero':
        baseline = torch.zeros_like(input_tensor).to(device)
    else:
        # Default to mean
        baseline = torch.mean(input_tensor, dim=0, keepdim=True).expand_as(input_tensor).to(device)
    
    n_dims = encoder.layers[-1].out_features if hasattr(encoder.layers[-1], 'out_features') else 64
    gene_names = expr_df.columns.tolist()
    
    rnk_files = []
    
    # Setup Attribution Method
    if method.lower() == 'deeplift':
        from captum.attr import DeepLift
        dl = DeepLift(encoder)
    elif method.lower() == 'shap':
        from captum.attr import GradientShap
        # GradientShap needs a baseline distribution, usually we take a random subset of data
        # Here we use the calculated baseline (zeros) for simplicity or random samples
        gs = GradientShap(encoder)
    else:
        raise ValueError(f"Unknown method: {method}")

    logging.info(f"Calculating attributions across {n_dims} dimensions...")
    
    # Compute attribution for each output dimension
    for dim in tqdm(range(n_dims), desc=f"{method} Attribution"):
        # Target is the output index (dimension)
        if method.lower() == 'deeplift':
            # attribution shape: (n_samples, n_features)
            attributions = dl.attribute(input_tensor, baselines=baseline, target=dim)
        elif method.lower() == 'shap':
             # Use a subset for background to speed up if needed, but here simple zero baseline
             attributions = gs.attribute(input_tensor, baselines=baseline, target=dim)
        
        # Aggregate attributions across samples (e.g., mean absolute value)
        # We want global importance of a gene for this dimension
        # Option 1: Mean attribution (keeps direction)
        # Option 2: Mean Absolute attribution (magnitude only)
        # For RNK file which needs direction (up/down regulation association), Mean is better.
        # Positive attribution => increasing gene increases dim value.
        avg_attr = torch.mean(attributions, dim=0).detach().cpu().numpy()
        
        rnk_path = output_dir / f"{dataset_name}_AE_{method}_dim{dim}.rnk"
        save_rnk_file(gene_names, avg_attr, rnk_path)
        rnk_files.append(rnk_path)
        
    logging.info(f"Generated {len(rnk_files)} RNK files for AE + {method}")
    return rnk_files
