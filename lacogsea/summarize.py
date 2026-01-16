from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union
import numpy as np
import pandas as pd

def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Normalize all whitespace (including non-breaking spaces) to a single space
    df.columns = (
        df.columns.astype(str)
        .str.replace(r"<.*?>", "", regex=True) # Remove HTML tags
        .str.replace(r"\s+", " ", regex=True)  # Normalize whitespace
        .str.strip()
    )
    return df

def _find_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        c = cols.get(cand.lower())
        if c:
            return c
    return None

def read_gsea_report_tsv(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path, sep="\t", low_memory=False)
    return _clean_columns(df)

def extract_pathway_stats(report_df: pd.DataFrame) -> pd.DataFrame:
    name_col = _find_col(report_df, ["NAME", "Term", "pathway", "Pathway"])
    nes_col = _find_col(report_df, ["NES", "nes", "Normalized Enrichment Score"])
    fdr_col = _find_col(report_df, ["FDR q-val", "FDR q-value", "fdr_qval", "FDR", "fdr", "fdr_q-val", "q-value"])
    
    if not name_col or not nes_col or not fdr_col:
        return pd.DataFrame()

    def clean_numeric(val):
        if pd.isna(val) or str(val).strip() == "---":
            return np.nan
        # Handle cases like "< 0.001" or "> 1.0"
        s = str(val).replace("<", "").replace(">", "").strip()
        try:
            return float(s)
        except ValueError:
            return np.nan

    df = pd.DataFrame({
        "Pathway": report_df[name_col].astype(str),
        "NES": report_df[nes_col].apply(clean_numeric),
        "FDR": report_df[fdr_col].apply(clean_numeric)
    })
    return df.dropna().drop_duplicates("Pathway").set_index("Pathway")

def merge_pos_neg_nes(pos_nes: pd.Series, neg_nes: pd.Series) -> pd.Series:
    all_idx = pos_nes.index.union(neg_nes.index)
    merged = {}
    for name in all_idx:
        pv = pos_nes.get(name, np.nan)
        nv = neg_nes.get(name, np.nan)
        if pd.isna(pv):
            merged[name] = float(nv)
        elif pd.isna(nv):
            merged[name] = float(pv)
        else:
            merged[name] = float(pv) if abs(float(pv)) >= abs(float(nv)) else float(nv)
    return pd.Series(merged, name="NES")

def _extract_dim_from_name(name: str) -> Optional[int]:
    m = re.search(r"_dim(\d+)\.gseapreranked", name.lower())
    if m:
        return int(m.group(1))
    m = re.search(r"_dim(\d+)$", name.lower())
    if m:
        return int(m.group(1))
    return None

def build_nes_matrix_from_gsea_results_dir(gsea_results_dir: Union[str, Path], dims: int) -> pd.DataFrame:
    gsea_results_dir = Path(gsea_results_dir)
    dim_to_folder: Dict[int, Path] = {}
    for folder in gsea_results_dir.iterdir():
        if not folder.is_dir():
            continue
        if ".gseapreranked." not in folder.name.lower():
            continue
        d = _extract_dim_from_name(folder.name)
        if d is None or d >= dims:
            continue
        prev = dim_to_folder.get(d)
        if prev is None or folder.stat().st_mtime > prev.stat().st_mtime:
            dim_to_folder[d] = folder

    per_dim: Dict[int, pd.Series] = {}
    pathways = set()

    for d in range(dims):
        folder = dim_to_folder.get(d)
        if folder is None:
            per_dim[d] = pd.Series(dtype=float)
            continue
        pos_files = list(folder.glob("gsea_report_for_na_pos_*.tsv"))
        neg_files = list(folder.glob("gsea_report_for_na_neg_*.tsv"))
        if not pos_files or not neg_files:
            per_dim[d] = pd.Series(dtype=float)
            continue
        pos_nes = extract_pathway_stats(read_gsea_report_tsv(pos_files[0]))["NES"]
        neg_nes = extract_pathway_stats(read_gsea_report_tsv(neg_files[0]))["NES"]
        merged = merge_pos_neg_nes(pos_nes, neg_nes)
        per_dim[d] = merged
        pathways |= set(merged.index)

    pathways = sorted(pathways)
    mat = np.zeros((dims, len(pathways)), dtype=float)
    for d in range(dims):
        s = per_dim[d]
        if s.empty:
            continue
        mat[d, :] = s.reindex(pathways).fillna(0.0).astype(float).values

    return pd.DataFrame(mat, index=[f"dim_{d}" for d in range(dims)], columns=pathways)

def build_nes_plus_minus_from_gsea_results_dir(gsea_results_dir: Union[str, Path], dims: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gsea_results_dir = Path(gsea_results_dir)
    dim_to_folder: Dict[int, Path] = {}
    for folder in gsea_results_dir.iterdir():
        if not folder.is_dir():
            continue
        if ".gseapreranked." not in folder.name.lower():
            continue
        d = _extract_dim_from_name(folder.name)
        if d is None or d >= dims:
            continue
        prev = dim_to_folder.get(d)
        if prev is None or folder.stat().st_mtime > prev.stat().st_mtime:
            dim_to_folder[d] = folder

    per_dim_plus: Dict[int, pd.Series] = {}
    per_dim_minus: Dict[int, pd.Series] = {}
    pathways = set()

    for d in range(dims):
        folder = dim_to_folder.get(d)
        if folder is None:
            per_dim_plus[d] = pd.Series(dtype=float)
            per_dim_minus[d] = pd.Series(dtype=float)
            continue
        pos_files = list(folder.glob("gsea_report_for_na_pos_*.tsv"))
        neg_files = list(folder.glob("gsea_report_for_na_neg_*.tsv"))
        if not pos_files or not neg_files:
            per_dim_plus[d] = pd.Series(dtype=float)
            per_dim_minus[d] = pd.Series(dtype=float)
            continue
        pos_stats = extract_pathway_stats(read_gsea_report_tsv(pos_files[0]))
        neg_stats = extract_pathway_stats(read_gsea_report_tsv(neg_files[0]))
        pos_nes = pos_stats["NES"]
        neg_nes = neg_stats["NES"]
        neg_mag = neg_nes.abs()
        per_dim_plus[d] = pos_nes
        per_dim_minus[d] = neg_mag
        pathways |= set(pos_nes.index) | set(neg_mag.index)

    pathways = sorted(pathways)
    mat_plus = np.zeros((dims, len(pathways)), dtype=float)
    mat_minus = np.zeros((dims, len(pathways)), dtype=float)
    for d in range(dims):
        s_plus = per_dim_plus[d]
        s_minus = per_dim_minus[d]
        if not s_plus.empty:
            mat_plus[d, :] = s_plus.reindex(pathways).fillna(0.0).astype(float).values
        if not s_minus.empty:
            mat_minus[d, :] = s_minus.reindex(pathways).fillna(0.0).astype(float).values

    df_plus = pd.DataFrame(mat_plus, index=[f"dim_{d}" for d in range(dims)], columns=pathways)
    df_minus = pd.DataFrame(mat_minus, index=[f"dim_{d}" for d in range(dims)], columns=pathways)
    return df_plus, df_minus
def check_gsea_result_exists(gsea_results_dir: Union[str, Path], label: str) -> bool:
    """Check if valid GSEA results already exist for a given label."""
    gsea_results_dir = Path(gsea_results_dir)
    if not gsea_results_dir.exists():
        return False
        
    pattern = f"{label}.GseaPreranked.*"
    for folder in gsea_results_dir.glob(pattern):
        if folder.is_dir():
            # Check for report files and index.html
            pos = list(folder.glob("gsea_report_for_na_pos_*.tsv"))
            neg = list(folder.glob("gsea_report_for_na_neg_*.tsv"))
            if pos and neg and (folder / "index.html").exists():
                return True
    return False
def get_top_pathways_for_dims(gsea_results_dir: Union[str, Path], dims: int, top_n: int = 5) -> pd.DataFrame:
    """Collect top N pathways per dimension based on |NES| where FDR < 0.05."""
    gsea_results_dir = Path(gsea_results_dir)
    dim_to_folder: Dict[int, Path] = {}
    for folder in gsea_results_dir.iterdir():
        if not folder.is_dir() or ".gseapreranked." not in folder.name.lower():
            continue
        d = _extract_dim_from_name(folder.name)
        if d is not None and d < dims:
            prev = dim_to_folder.get(d)
            if prev is None or folder.stat().st_mtime > prev.stat().st_mtime:
                dim_to_folder[d] = folder

    all_stats = []
    for d in range(dims):
        folder = dim_to_folder.get(d)
        if folder is None: continue
        
        pos_files = list(folder.glob("gsea_report_for_na_pos_*.tsv"))
        neg_files = list(folder.glob("gsea_report_for_na_neg_*.tsv"))
        if not pos_files or not neg_files: continue
        
        pos_df = extract_pathway_stats(read_gsea_report_tsv(pos_files[0]))
        neg_df = extract_pathway_stats(read_gsea_report_tsv(neg_files[0]))
        combined = pd.concat([pos_df, neg_df])
        
        # Filter significance
        sig = combined[combined["FDR"] < 0.05].copy()
        sig["abs_NES"] = sig["NES"].abs()
        top = sig.sort_values("abs_NES", ascending=False).head(top_n)
        
        if not top.empty:
            top_series = top["NES"]
            top_series.name = f"dim_{d}"
            all_stats.append(top_series)

    if not all_stats:
        return pd.DataFrame()

    # Create a full matrix of all unique top pathways
    all_pathways = pd.concat(all_stats, axis=1)
    return all_pathways
