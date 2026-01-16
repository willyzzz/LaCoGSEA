from __future__ import annotations

import argparse
import logging
import subprocess
import os
import sys
from pathlib import Path

from .pipeline import train_autoencoder, run_gsea, summarize_gsea, run_full_pipeline, compute_activity
from .evaluation import calculate_pearson_correlation, save_correlation_lists
from .run_gsea_java import find_gsea_cli, find_java_cmd, find_gsea_dir
from .utils import install_internal_java

LOGO = r"""
    __          ______      ______  _____ ______   ___ 
   / /   ____ _/ ____/___  / ____/ / ___// ____/  /   |
  / /   / __ `// /   / __ \/ / __   \__ \/ __/    / /| |
 / /___/ /_/ // /___/ /_/ / /_/ /  ___/ / /___   / ___ |
/_____/\__,_/ \____/\____/\____/  /____/_____/  /_/  |_|
"""

BUILTIN_GENE_SETS = {
    "kegg": "kegg.gmt",
    "go_bp": "go_bp.gmt",
    "reactome": "reactome.gmt",
    "c6": "c6.gmt",
}

def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(message)s" if not verbose else "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=level, format=fmt, stream=sys.stdout)
    for logger_name in ["torch", "matplotlib", "PIL", "urllib3"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_default_data_path(filename: str) -> str:
    """Get absolute path to bundled data file."""
    package_root = Path(__file__).parent
    data_path = package_root / "data" / filename
    return str(data_path)


def resolve_gene_set(path_or_key: str) -> str:
    """Resolve a gene set string to an absolute path."""
    if path_or_key.lower() in BUILTIN_GENE_SETS:
        return get_default_data_path(BUILTIN_GENE_SETS[path_or_key.lower()])
    
    # If it looks like a file path and exists, use it
    if os.path.exists(path_or_key):
        return os.path.abspath(path_or_key)
        
    return path_or_key


def cmd_train(args: argparse.Namespace) -> None:
    logging.info(f"--- Training Autoencoder (dim={args.dim}, epochs={args.epochs}) ---")
    result = train_autoencoder(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        output_dir=args.output,
        dim=args.dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
    )
    logging.info("[OK] Training complete.")
    logging.info(f"     Encoder:   {result.encoder_path}")
    logging.info(f"     Embedding: {result.embedding_path}")


def cmd_gsea(args: argparse.Namespace) -> None:
    gene_set_path = resolve_gene_set(args.gene_set)
    logging.info(f"--- Running GSEA Preranked for {args.label} ---")
    logging.info(f"    Gene Set: {os.path.basename(gene_set_path)}")
    
    success, err_msg = run_gsea(
        rnk_file=args.rnk,
        gene_set=gene_set_path,
        output_dir=args.output,
        label=args.label,
        permutations=args.permutations,
        min_size=args.min_size,
        max_size=args.max_size,
        memory=args.memory,
        scoring_scheme=args.scoring_scheme,
        make_sets=args.make_sets,
    )
    if not success:
        raise SystemExit(f"[ERROR] GSEA failed: {err_msg}")
    logging.info("[OK] GSEA complete.")


def cmd_summarize(args: argparse.Namespace) -> None:
    logging.info(f"--- Summarizing GSEA results from {args.gsea_dir} ---")
    summarize_gsea(
        gsea_results_dir=args.gsea_dir,
        dims=args.dims,
        output_path=args.output,
        plus_minus=args.plus_minus,
    )
    logging.info(f"[OK] Summary saved to: {args.output}")


def cmd_rnk(args: argparse.Namespace) -> None:
    import pandas as pd
    logging.info("--- Generating RNK files ---")
    emb = pd.read_csv(args.embedding, index_col=0)
    expr = pd.read_csv(args.expression, index_col=0)
    corrs = calculate_pearson_correlation(emb, expr)
    save_correlation_lists(corrs, args.output)
    logging.info(f"[OK] Saved RNK files to: {Path(args.output) / 'correlations'}")


def cmd_activity(args: argparse.Namespace) -> None:
    logging.info("--- Computing Sample x Pathway activity matrix ---")
    compute_activity(
        embedding_csv=args.embedding,
        nes_tsv=args.nes,
        output_path=args.output
    )
    logging.info(f"[OK] Activity matrix saved to: {args.output}")


def cmd_run(args: argparse.Namespace) -> None:
    logging.info(LOGO)
    logging.info("Starting Full LaCoGSEA Pipeline")
    logging.info("Notice: LaCoGSEA uses the Broad Institute GSEA software (https://www.gsea-msigdb.org/)")
    logging.info("-" * 40)
    
    import datetime
    gene_set_path = resolve_gene_set(args.gene_set)
    gmt_label = Path(gene_set_path).stem
    
    # Logic for output directory:
    # 1. If args.output is a path that already has gsea/ results, we use it directly (Resume mode)
    # 2. Otherwise, we create a subfolder with GMT and Timestamp
    candidate_dir = Path(args.output)
    if (candidate_dir / "gsea").exists():
        run_dir = candidate_dir
        logging.info(f"Existing results detected in {run_dir}. Entering Resume mode.")
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = candidate_dir / f"{gmt_label}_{timestamp}"
    
    if gene_set_path == args.gene_set and not os.path.exists(gene_set_path):
         logging.warning(f"Note: Gene set '{args.gene_set}' not found as file or builtin. Passing raw string to GSEA.")

    run_full_pipeline(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        gene_set=gene_set_path,
        output_dir=run_dir,
        dim=args.dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        label=args.label,
        permutations=args.permutations,
        min_size=args.min_size,
        max_size=args.max_size,
        scoring_scheme=args.scoring_scheme,
        make_sets=args.make_sets,
    )


def cmd_setup(args: argparse.Namespace) -> None:
    print("\n=== LaCoGSEA Setup Check ===\n")
    gsea_dir = find_gsea_dir()
    java_found = False
    if gsea_dir:
        java_cmd = find_java_cmd(gsea_dir)
        try:
            subprocess.run([java_cmd, "-version"], capture_output=True, check=True)
            print(f"[OK] Java found: {java_cmd}")
            java_found = True
        except Exception:
            print(f"[!!] Java NOT found using: {java_cmd}")
    else:
        try:
            subprocess.run(["java", "-version"], capture_output=True, check=True)
            print("[OK] System Java found.")
            java_found = True
        except Exception:
            print("[!!] System Java NOT found.")

    gsea_cli = find_gsea_cli()
    if gsea_cli:
        origin = "Bundled" if "vendor" in gsea_cli else "External"
        print(f"[OK] GSEA CLI found ({origin}): {gsea_cli}")
    else:
        print("[!!] GSEA CLI NOT found.")
        
    if java_found and gsea_cli:
        print("\n[SUCCESS] Environment ready for GSEA Java engine!")
    else:
        if not java_found and gsea_cli:
            print("\n[NOTICE] Java is missing, but GSEA CLI is found.")
            if getattr(args, 'yes', False):
                choice = 'y'
            else:
                try:
                    choice = input("Would you like to automatically download and install a portable JRE? (y/n): ")
                except EOFError:
                    choice = 'n'

            if choice.lower() == 'y':
                if install_internal_java(find_gsea_dir()):
                    print("[SUCCESS] Java installed! Please run 'lacogsea setup' again to verify.")
                else:
                    print("[ERROR] Java installation failed.")
        else:
            print("\n[NOTICE] GSEA Java engine not configured. Please install Java to use LaCoGSEA.")

def cmd_install_java(args: argparse.Namespace) -> None:
    """Internal command to trigger java installation."""
    gsea_dir = find_gsea_dir()
    if not gsea_dir:
        print("[ERROR] GSEA vendor directory not found. Cannot install Java here.")
        return
    print("Starting automated Java installation...")
    if install_internal_java(gsea_dir):
        print("[SUCCESS] Portable Java installation complete.")
    else:
        print("[ERROR] Installation failed.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lacogsea", description="LaCoGSEA core CLI")
    p.add_argument("--verbose", action="store_true", help="Enable verbose debug logging")
    sub = p.add_subparsers(dest="command", required=True)

    default_csv = get_default_data_path("gse126848.csv")
    
    gsea_help_msg = "Gene set: use built-in names (kegg, go_bp, reactome, c6) or a path to your own .gmt file."

    t = sub.add_parser("train", help="Train autoencoder and export embeddings")
    t.add_argument("--train-csv", default=default_csv)
    t.add_argument("--test-csv", default=default_csv)
    t.add_argument("--output", default="result/lacogsea_train")
    t.add_argument("--dim", type=int, default=32)
    t.add_argument("--batch-size", type=int, default=128)
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--seed", type=int, default=42)
    t.set_defaults(func=cmd_train)

    g = sub.add_parser("gsea", help="Run GSEA Preranked")
    g.add_argument("--rnk", required=True)
    g.add_argument("--gene-set", default="kegg", help=gsea_help_msg)
    g.add_argument("--output", required=True)
    g.add_argument("--label", default="lacogsea")
    g.add_argument("--permutations", type=int, default=1000)
    g.add_argument("--min-size", type=int, default=15)
    g.add_argument("--max-size", type=int, default=500)
    g.add_argument("--memory", default="4g")
    g.add_argument("--scoring-scheme", default="weighted", choices=["weighted", "classic", "weighted_p2", "weighted_p1.5"])
    g.add_argument("--no-make-sets", action="store_false", dest="make_sets", help="Disable generating detailed gene set reports to save space.")
    g.set_defaults(make_sets=True)
    g.set_defaults(func=cmd_gsea)

    s = sub.add_parser("summarize", help="Summarize GSEA outputs")
    s.add_argument("--gsea-dir", required=True)
    s.add_argument("--dims", type=int, required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--plus-minus", action="store_true")
    s.set_defaults(func=cmd_summarize)

    r = sub.add_parser("rnks", help="Compute RNK files")
    r.add_argument("--embedding", required=True)
    r.add_argument("--expression", required=True)
    r.add_argument("--output", required=True)
    r.set_defaults(func=cmd_rnk)

    act = sub.add_parser("activity", help="Compute activity matrix")
    act.add_argument("--embedding", required=True)
    act.add_argument("--nes", required=True)
    act.add_argument("--output", required=True)
    act.set_defaults(func=cmd_activity)

    setup = sub.add_parser("setup", help="Check environment")
    setup.add_argument("--yes", action="store_true", help="Auto-confirm Java installation")
    setup.set_defaults(func=cmd_setup)

    ij = sub.add_parser("install-java", help="Automatically install a portable JRE")
    ij.set_defaults(func=cmd_install_java)

    run = sub.add_parser("run", help="Run full pipeline")
    run.add_argument("--train-csv", default=default_csv)
    run.add_argument("--test-csv", default=default_csv)
    run.add_argument("--gene-set", default="kegg", help=gsea_help_msg)
    run.add_argument("--output", default="result")
    run.add_argument("--dim", type=int, default=4)
    run.add_argument("--epochs", type=int, default=100)
    run.add_argument("--batch-size", type=int, default=128)
    run.add_argument("--label", default="lacogsea")
    run.add_argument("--permutations", type=int, default=1000)
    run.add_argument("--min-size", type=int, default=15)
    run.add_argument("--max-size", type=int, default=500)
    run.add_argument("--scoring-scheme", default="weighted", choices=["weighted", "classic", "weighted_p2", "weighted_p1.5"])
    run.add_argument("--no-make-sets", action="store_false", dest="make_sets", help="Disable generating detailed gene set reports to save space.")
    run.set_defaults(make_sets=True)
    run.set_defaults(func=cmd_run)

    return p


def main(argv: list[str] | None = None) -> None:
    # Fix Windows encoding issues for emojis in terminal
    if sys.platform == "win32":
        try:
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8')
            if hasattr(sys.stderr, 'reconfigure'):
                sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass

    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    try:
        args.func(args)
    except KeyboardInterrupt:
        logging.info("\nAborted by user.")
        sys.exit(1)
    except Exception as e:
        if args.verbose:
            logging.exception("Fatal error occurred:")
        else:
            logging.error(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
