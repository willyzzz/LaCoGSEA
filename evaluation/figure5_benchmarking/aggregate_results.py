#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def aggregate_metrics():
    PROJECT_ROOT = Path(__file__).resolve().parents[2].absolute()
    out_dir = PROJECT_ROOT / "results" / "figure2" / "figure2_deep_analytics"
    metric_files = list(out_dir.glob('deep_metrics_GSE*.csv'))
    
    if not metric_files:
        logging.error("No dataset-specific metric files found.")
        return

    all_dfs = []
    for f in metric_files:
        logging.info(f"Loading {f.name}")
        df = pd.read_csv(f)
        all_dfs.append(df)
    
    summary_df = pd.concat(all_dfs, ignore_index=True)
    
    # Save combined
    summary_df.to_csv(out_dir / 'deep_metrics_all_datasets.csv', index=False)
    
    # Create a nice summary table: Method vs Dataset for Avg_Target_Rank
    # Filter for the summary rows (Metric_Type is NaN)
    summary_rows = summary_df[summary_df['Metric_Type'].isna()]
    
    pivot_rank = summary_rows.pivot(index='Method', columns='Dataset', values='Avg_Target_Rank')
    pivot_rank['Mean_Rank'] = pivot_rank.mean(axis=1)
    pivot_rank = pivot_rank.sort_values('Mean_Rank')
    
    pivot_rank.to_csv(out_dir / 'avg_target_rank_summary.csv')
    
    logging.info("Aggregation complete.")
    logging.info(f"Full summary saved to {out_dir / 'deep_metrics_all_datasets.csv'}")
    logging.info(f"Pivot summary saved to {out_dir / 'avg_target_rank_summary.csv'}")
    print("\nSummary Table (Avg Target Rank):")
    print(pivot_rank)

if __name__ == '__main__':
    aggregate_metrics()
