from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
import warnings
import shutil
import concurrent.futures
import multiprocessing

from .model_structure import Encoder, Decoder
from .training import train_auto_encoder
from .run_gsea_java import run_gsea_preranked, find_gsea_cli
from .summarize import (
    build_nes_matrix_from_gsea_results_dir,
    build_nes_plus_minus_from_gsea_results_dir,
    check_gsea_result_exists,
)
from .activity import calculate_sample_pathway_activity


def load_data(path: Union[str, Path]) -> pd.DataFrame:
    """Robustly load expression data from CSV, TXT, or TSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    
    # For .txt or .tsv, try to auto-detect the delimiter (usually tab)
    if path.suffix.lower() in [".txt", ".tsv"]:
        try:
            # engine='python' allows for auto-detection of delimiters
            return pd.read_csv(path, sep=None, engine='python', index_col=0)
        except Exception:
            # Fallback for complex txt files
            return pd.read_csv(path, sep="\t", index_col=0)
            
    # Default to standard CSV
    return pd.read_csv(path, index_col=0)


LOGGER = logging.getLogger(__name__)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _auto_log2_transform(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check max value and apply log2(x+1) only if data seems non-logged (max > 50)."""
    max_val = train_df.values.max()
    if max_val > 50:
        LOGGER.info(f"   📊 [Data] High scale detected (max={max_val:.1f}). Applying Log2(x+1).")
        train_df = pd.DataFrame(np.log2(train_df.values.astype(np.float32) + 1.0), index=train_df.index, columns=train_df.columns)
        test_df = pd.DataFrame(np.log2(test_df.values.astype(np.float32) + 1.0), index=test_df.index, columns=test_df.columns)
    else:
        LOGGER.info(f"   📊 [Data] Low scale detected (max={max_val:.1f}). Data assumed to be Log-transformed.")
    return train_df, test_df


def _align_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    common = train_df.columns.intersection(test_df.columns)
    if common.empty:
        raise ValueError("Train and test files do not share any gene columns.")
    train_df = train_df[common]
    test_df = test_df[common]
    train_df = train_df.apply(lambda r: r.fillna(r.mean()), axis=1)
    test_df = test_df.apply(lambda r: r.fillna(r.mean()), axis=1)
    return train_df, test_df


@dataclass
class TrainResult:
    encoder_path: Path
    decoder_path: Path
    embedding_path: Path
    embedding: pd.DataFrame


def train_autoencoder(
    train_csv: Union[str, Path],
    test_csv: Union[str, Path],
    output_dir: Union[str, Path],
    dim: int = 32,
    batch_size: int = 128,
    epochs: int = 200,
    seed: int = 42,
    hidden_dims: Optional[Iterable[int]] = None,
) -> TrainResult:
    output_dir = Path(output_dir)
    _ensure_dir(output_dir)

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_df = load_data(train_csv)
    test_df = load_data(test_csv)
    train_df, test_df = _align_train_test(train_df, test_df)

    train_df, test_df = _auto_log2_transform(train_df, test_df)

    input_dim = train_df.shape[1]
    if hidden_dims is None:
        hidden_dims = [max(dim * 4, min(input_dim // 2, 512))]

    train_tensor = torch.tensor(train_df.values, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(input_dim, list(hidden_dims), dim).to(device)
    decoder = Decoder(dim, list(hidden_dims), input_dim).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=5e-3)
    
    # Suppress warnings for GradScaler if CUDA is not available
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        scaler = GradScaler(enabled=torch.cuda.is_available())

    # AE Training progress bar
    bar_length_ae = 20
    for epoch in range(epochs):
        loss = train_auto_encoder(
            encoder, decoder, train_loader, criterion, optimizer, scaler, device
        )
        
        # Show progress every epoch in a compact way
        progress_ae = (epoch + 1) / epochs
        filled_ae = int(bar_length_ae * progress_ae)
        bar_ae = '█' * filled_ae + '░' * (bar_length_ae - filled_ae)
        percent_ae = int(progress_ae * 100)
        sys.stdout.write(f"\r      ⚡ AE Training: [{bar_ae}] {percent_ae}% | Epoch {epoch+1}/{epochs} | Loss: {loss:.4f}")
        sys.stdout.flush()

    sys.stdout.write("\n")

    encoder_path = output_dir / "encoder.pt"
    decoder_path = output_dir / "decoder.pt"
    torch.save(encoder.state_dict(), encoder_path)
    torch.save(decoder.state_dict(), decoder_path)

    encoder.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(test_df.values, dtype=torch.float32, device=device)
        emb = encoder(test_tensor).cpu().numpy()
    emb_df = pd.DataFrame(emb, index=test_df.index, columns=[f"dim_{i}" for i in range(dim)])
    embedding_path = output_dir / "test_embedding.csv"
    emb_df.to_csv(embedding_path)

    return TrainResult(
        encoder_path=encoder_path,
        decoder_path=decoder_path,
        embedding_path=embedding_path,
        embedding=emb_df,
    )


def run_gsea(
    rnk_file: Union[str, Path],
    gene_set: Union[str, Path],
    output_dir: Union[str, Path],
    label: str,
    permutations: int = 1000,
    min_size: int = 15,
    max_size: int = 500,
    memory: str = "4g",
    scoring_scheme: str = "weighted",
    make_sets: bool = True,
    quiet: bool = True,
) -> tuple[bool, str]:
    gsea_cli = find_gsea_cli()
    if gsea_cli:
        return run_gsea_preranked(
            rnk_file=str(rnk_file),
            gene_set_file=str(gene_set),
            output_dir=str(output_dir),
            label=label,
            memory=memory,
            permutations=permutations,
            min_size=min_size,
            max_size=max_size,
            seed=42,
            plot_top_x=0,
            scoring_scheme=scoring_scheme,
            make_sets="true" if make_sets else "false",
            quiet=quiet,
        )
    else:
        return False, "GSEA Java CLI not found. Please install Java and ensure GSEA is configured."


def summarize_gsea(
    gsea_results_dir: Union[str, Path],
    dims: int,
    output_path: Union[str, Path],
    plus_minus: bool = False,
) -> Path:
    gsea_results_dir = Path(gsea_results_dir)
    output_path = Path(output_path)
    _ensure_dir(output_path.parent)

    if plus_minus:
        plus_df, minus_df = build_nes_plus_minus_from_gsea_results_dir(gsea_results_dir, dims)
        plus_df.to_csv(output_path, sep="\t")
        minus_df.to_csv(output_path.with_name(output_path.stem + "_minus.tsv"), sep="\t")
    else:
        nes_df = build_nes_matrix_from_gsea_results_dir(gsea_results_dir, dims)
        nes_df.to_csv(output_path, sep="\t")
    return output_path


def compute_activity(
    embedding_csv: Union[str, Path],
    nes_tsv: Union[str, Path],
    output_path: Union[str, Path],
) -> Path:
    emb_df = pd.read_csv(embedding_csv, index_col=0)
    nes_df = pd.read_csv(nes_tsv, sep="\t", index_col=0)
    calculate_sample_pathway_activity(emb_df, nes_df, output_path)
    return Path(output_path)


def run_full_pipeline(
    train_csv: Union[str, Path],
    test_csv: Union[str, Path],
    gene_set: Union[str, Path],
    output_dir: Union[str, Path],
    dim: int = 4,
    epochs: int = 200,
    batch_size: int = 128,
    label: str = "lacogsea",
    permutations: int = 1000,
    min_size: int = 15,
    max_size: int = 500,
    scoring_scheme: str = "weighted",
    make_sets: bool = True,
    workers: int | None = None,
) -> Path:
    from .evaluation import calculate_pearson_correlation, save_correlation_lists
    output_dir = Path(output_dir)
    _ensure_dir(output_dir)

    # 1. Train
    LOGGER.info(f"\n[1/6] 🤖 Training Autoencoder (dim={dim}, epochs={epochs})...")
    train_res = train_autoencoder(train_csv, test_csv, output_dir, dim, batch_size, epochs)

    # 2. RNKs
    LOGGER.info("\n[2/6] 📉 Calculating Pearson Correlations...")
    expr_df = load_data(test_csv)
    corrs = calculate_pearson_correlation(train_res.embedding, expr_df)
    save_correlation_lists(corrs, output_dir)
    correlations_dir = output_dir / "correlations"

    # 3. GSEA
    LOGGER.info(f"\n[3/6] 🧬 Running GSEA Preranked for {dim} dimensions...")
    
    # Safety Check: If output_dir exists but was for a different dimension, clean it
    gsea_base_dir = output_dir / "gsea"
    if gsea_base_dir.exists():
        existing_dirs = list(gsea_base_dir.glob(f"{label}_dim*"))
        if existing_dirs:
            # Check if any folder index exceeds or is far from current dim
            # Simple heuristic: if any dimN exists where N >= dim, it's definitely incompatible
            try:
                # Extract dimension numbers from existing GSEA result directories
                existing_dim_nums = []
                for d_path in existing_dirs:
                    try:
                        # Expected format: {label}_dim{N}.GseaPreranked.{timestamp}
                        # We need to parse the {N} part
                        parts = d_path.name.split('_dim')
                        if len(parts) > 1:
                            dim_str = parts[1].split('.')[0]
                            existing_dim_nums.append(int(dim_str))
                    except ValueError:
                        continue # Skip directories that don't match the expected pattern
                
                if existing_dim_nums:
                    max_existing_dim = max(existing_dim_nums)
                    # If the maximum existing dimension index is greater than or equal to the current 'dim',
                    # it suggests an incompatible previous run.
                    # Or if the number of existing dimensions is not equal to 'dim'
                    if max_existing_dim >= dim or len(existing_dim_nums) != dim:
                        LOGGER.warning(f"Existing GSEA results in {gsea_base_dir} are incompatible (Dim mismatch or incomplete). Performing fresh run.")
                        shutil.rmtree(gsea_base_dir)
            except Exception as e:
                LOGGER.warning(f"Error checking existing GSEA results for compatibility: {e}. Proceeding with fresh run if needed.")
                # If an error occurs during parsing, it's safer to assume incompatibility
                shutil.rmtree(gsea_base_dir, ignore_errors=True)

    gmt_name = Path(gene_set).name
    LOGGER.info(f"[3/6] Running GSEA (GeneSet: {gmt_name}, Size: {min_size}-{max_size}, Permutations: {permutations}) for {dim} dimensions...")
    gsea_output_dir = output_dir / "gsea"
    _ensure_dir(gsea_output_dir)

    # Multi-threaded GSEA execution
    if workers is not None:
        max_workers = max(1, workers)
        LOGGER.info(f"   [Performance] Using {max_workers} user-specified workers for GSEA.")
    else:
        max_workers = max(1, multiprocessing.cpu_count() // 2)
        # Cap workers to avoid memory issues (GSEA is Java-heavy)
        # 4 workers * 2GB = 8GB headroom, which is safer for most systems
        max_workers = min(max_workers, 4)
        LOGGER.info(f"   [Performance] Using {max_workers} auto-calculated workers for GSEA.")
    
    tasks = []
    for i in range(dim):
        dim_label = f"{label}_dim{i}"
        rnk_file = correlations_dir / f"dimension_{i}.rnk"
        tasks.append((rnk_file, gene_set, gsea_output_dir, dim_label))

    completed = 0
    bar_length = 30 # Define bar_length for progress bar
    # Use ThreadPoolExecutor for I/O bound subprocesses
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_dim = {
            executor.submit(
                run_gsea, 
                t[0], t[1], t[2], t[3], 
                permutations, min_size, max_size, 
                memory="2g", # Lower memory per worker to allow parallelism
                scoring_scheme=scoring_scheme, 
                make_sets=make_sets, 
                quiet=True
            ): i for i, t in enumerate(tasks)
        }

        for future in concurrent.futures.as_completed(future_to_dim):
            i = future_to_dim[future]
            success, err_msg = future.result()
            completed += 1
            
            if not success:
                LOGGER.error(f"   ❌ [Error] Dimension {i} failed: {err_msg}")
                # We don't exit immediately here to let other threads finish/error properly
            
            # Progress Update
            progress = completed / dim
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            percent = int(progress * 100)
            sys.stdout.write(f"\r      🏁 GSEA Progress: [{bar}] {percent}% ({completed}/{dim} dims)")
            sys.stdout.flush()
            
    sys.stdout.write("\n")

    # 4. Summarize
    LOGGER.info("\n[4/6] 📋 Summarizing results into NES matrix...")
    nes_path = output_dir / "nes.tsv"
    summarize_gsea(gsea_output_dir, dim, nes_path)

    # 5. Activity
    LOGGER.info("\n[5/6] 🧬 Calculating Pathway Activity Matrix...")
    activity_path = output_dir / "pathway_activity.tsv"
    compute_activity(train_res.embedding_path, nes_path, activity_path)

    # 6. Visualization
    LOGGER.info("\n[6/6] 🎨 Generating Top Pathways Heatmap...")
    from .summarize import get_top_pathways_for_dims
    from .plotting import plot_top_pathways_heatmap
    
    # Use max 32 dims for visualization
    viz_dims = min(dim, 32)
    top_df = get_top_pathways_for_dims(gsea_output_dir, viz_dims, top_n=5)
    plot_path = output_dir / "top_pathways_heatmap.png"
    plot_top_pathways_heatmap(top_df, plot_path)

    LOGGER.info(f"\n[DONE] Final output at: {output_dir}")
    return nes_path
