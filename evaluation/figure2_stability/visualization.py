#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2].absolute()
OUT_DIR = PROJECT_ROOT / "results" / "figure1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Data
methods = ["LaCoGSEA", "PCA"]
counts = [0, 0]

plt.figure(figsize=(6, 5))

# Draw bars (very thin as requested)
bars = plt.bar(methods, counts, color=["#2E86AB", "#A23B72"], alpha=0.6, width=0.2, edgecolor="black", linewidth=1)

# Label "0" on top of each bar location
for i, method in enumerate(methods):
    plt.text(i, 0.01, '0', 
             ha='center', va='bottom', fontsize=18, fontweight='bold', color="black")

plt.ylabel("Detected significant pathways per component", fontsize=12)
plt.xlabel("")
plt.title("Negative Control: Random Gaussian Noise", fontsize=14, fontweight='bold')

# Set Y axis limits to show clearly that it's 0
plt.ylim(-0.05, 0.5)
plt.grid(axis="y", alpha=0.3)

# Main Annotation in the center
plt.text(
    0.5, 0.5,
    "LaCoGSEA=0, PCA=0\n(no false positives)",
    ha="center",
    va="center",
    fontsize=18,
    fontweight="bold",
    color="#d62728", # A refined red
    transform=plt.gca().transAxes,
    bbox=dict(facecolor='white', alpha=0.9, edgecolor='#d62728', boxstyle='round,pad=0.5', linewidth=2)
)

plt.tight_layout()
out_path = OUT_DIR / "Figure1B_negative_control.png"
plt.savefig(out_path, dpi=300)
plt.close()

print(f"Figure 1B plot saved to: {out_path}")
