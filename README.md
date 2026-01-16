# LaCoGSEA: Latent Correlation-gene set enrichment analysis

LaCoGSEA is a powerful tool designed to interpret the latent space of autoencoders trained on gene expression data. It identifies biological pathways associated with each latent dimension through Pearson correlation and GSEA, providing a systematic way to understand what features your model has learned.

---

### 📋 Prerequisites
- **Python**: 3.8 to 3.12 (Recommended: 3.10)
- **Java**: JRE 11+ or 17+ (Will be installed automatically by the GUI if missing)

---

## 🎨 User Interface (Easiest Way to Run)

The easiest way to use LaCoGSEA is through its built-in Graphical User Interface. 

### 🚀 Windows Users
1. **Double-click `LaCoGSEA_run.bat`**.
2. If Python is missing, it will provide you with a download link.
3. Your browser will open the interface automatically.

### 🐧 Linux / macOS Users
1. Open terminal in the project folder.
2. Run: `bash LaCoGSEA_run.sh`
3. (Optional) You can also install manually: `pip install -e .` followed by `lacogsea-gui`.

### ❓ No Python Environment?
LaCoGSEA requires Python 3.8-3.12. If you don't have it:
- **Windows**: [Download Python from here](https://www.python.org/downloads/). **Crucial**: Tick "Add Python to PATH" during install.
- **Linux**: Usually pre-installed. If not: `sudo apt install python3 python3-pip`.

*Note: Environment checks (like Java) and data transformations are handled automatically upon clicking 'Run' in the GUI.*

### Manual Launch (Command Line)
```bash
lacogsea-gui
```

---

## ✨ Key Features

- **Consolidated GSEA Engine**: Powered exclusively by the high-performance GSEA Java CLI (bundled or system-installed).
- **Auto Data Transformation**: Intelligently detects data scale and applies `Log2(x + 1)` only when necessary.
- **Robustness & Resume**: Built-in "Resume Mode" skips already completed dimensions if a run is interrupted.
- **Detailed Error Diagnosis**: Captures and reports specific GSEA errors directly in the console/GUI.

---

## 🛠️ CLI Usage

For advanced users, LaCoGSEA provides a comprehensive command line interface.

### 1. Setup Environment
Ensure Java is ready:
```bash
lacogsea setup --yes
```

### 2. Run Full Pipeline
```bash
lacogsea run --train-csv data.csv --dim 4 --gene-set kegg --output result
```

### 3. Step-by-Step (Internal Commands)
- `lacogsea train`: Train the Autoencoder and save embedding.
- `lacogsea rnks`: Generate `.rnk` files for GSEA based on correlations.
- `lacogsea gsea`: Run GSEA for specific dimensions.
- `lacogsea summarize`: Aggregate GSEA results into a NES matrix.

---

## 📑 Result Interpretation

After running, the `result/` folder will contain:
- `nes.tsv`: The Normalized Enrichment Score (NES) matrix (Dimensions vs Pathways).
- `pathway_activity.tsv`: The calculated Pathway Activity Score for each sample.
- `top_pathways_heatmap.png`: High-resolution summary plot.
- `gsea/`: Raw output from the GSEA software for deep dives into specific pathways.

---

## 📄 License & Attribution

- **License**: LaCoGSEA is released under the MIT License.
- **Third-party software**: This tool interfaces with the Broad Institute's **GSEA software**. User of GSEA is subject to GSEA's license (Academic use only).

Developed by the LaCoGSEA Team.
