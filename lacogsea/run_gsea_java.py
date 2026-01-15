#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GSEA Java tool invocation module
"""

import os
import sys
import subprocess
import logging
import time
from pathlib import Path

def find_gsea_cli():
    """Find GSEA CLI tool. Supports override via GSEA_CLI environment variable."""
    env_path = os.getenv("GSEA_CLI")
    cli_filename = "gsea-cli.bat" if os.name == "nt" else "gsea-cli.sh"
    
    package_dir = os.path.dirname(os.path.abspath(__file__))
    bundled_path = os.path.join(package_dir, "vendor", "GSEA_4.4.0", cli_filename)
    
    candidates = [
        env_path,
        bundled_path,
        os.path.join(os.getcwd(), cli_filename),
    ]
    
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None

def find_gsea_dir():
    gsea_cli = find_gsea_cli()
    return os.path.dirname(gsea_cli) if gsea_cli else None

def find_java_cmd(gsea_dir):
    if not gsea_dir: return "java"
    bundled_jdk = os.path.join(gsea_dir, "jdk")
    if os.path.exists(bundled_jdk):
        # Handle different structures (bin/java.exe for Windows, bin/java for Unix)
        ext = ".exe" if os.name == "nt" else ""
        java_cmd = os.path.join(bundled_jdk, "bin", f"java{ext}")
        if os.path.exists(java_cmd):
            return java_cmd
    return "java"

def run_gsea_preranked(rnk_file, gene_set_file, output_dir, label, memory="4g",
                       permutations=1000, min_size=15, max_size=500, seed=42,
                       plot_top_x=0, scoring_scheme="weighted", make_sets="true", 
                       quiet=True):
    """
    Run GSEA Preranked analysis.
    """
    rnk_file = os.path.abspath(rnk_file)
    gene_set_file = os.path.abspath(gene_set_file)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    gsea_cli = find_gsea_cli()
    if not gsea_cli:
        logging.error("GSEA CLI not found.")
        return False
    
    if not quiet:
        logging.info(f"RNK: {os.path.basename(rnk_file)}, GeneSet: {os.path.basename(gene_set_file)}")
    
    gsea_dir = find_gsea_dir()
    java_cmd = find_java_cmd(gsea_dir)
    modules_path = os.path.join(gsea_dir, "modules")
    logging_properties = os.path.join(gsea_dir, "logging.properties")
    
    cmd = [
        java_cmd,
        f"--module-path={modules_path}",
        f"-Xmx{memory}",
        "-Djava.awt.headless=true",
        f"-Djava.util.logging.config.file={logging_properties}",
        "--module=org.gsea_msigdb.gsea/xapps.gsea.CLI",
        "GSEAPreranked",
        "-rnk", rnk_file,
        "-gmx", gene_set_file,
        "-out", output_dir,
        "-rpt_label", label,
        "-plot_top_x", str(plot_top_x),
        "-set_max", str(max_size),
        "-set_min", str(min_size),
        "-nperm", str(permutations),
        "-scoring_scheme", scoring_scheme,
        "-make_sets", make_sets,
        "-rnd_seed", str(seed),
        "-zip_report", "false"
    ]
    
    original_cwd = os.getcwd()
    try:
        os.chdir(gsea_dir)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,
            shell=False
        )
        
        if result.returncode != 0:
            error_msg = "GSEA Process Error"
            # Try to extract meaningful error from stderr
            if result.stderr:
                lines = [line.strip() for line in result.stderr.split('\n') if line.strip()]
                # Look for common GSEA error markers
                relevant_lines = [l for l in lines if "ERROR" in l.upper() or "Exception" in l]
                if relevant_lines:
                    error_msg = relevant_lines[-1]
                elif lines:
                    error_msg = lines[-1]
            return False, error_msg
            
        # Success verification
        result_pattern = f"{label}.GseaPreranked.*"
        
        # Check for results
        results = list(Path(output_dir).glob(result_pattern))
        if results:
            # Check if index.html exists in at least one matching dir
            for rdir in results:
                if (rdir / "index.html").exists():
                    return True, "Success"
            
        return False, "Output directory or index.html not found."
            
    except Exception as e:
        return False, str(e)
    finally:
        os.chdir(original_cwd)
