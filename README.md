# LaCoGSEA: Latent Correlation-gene set enrichment analysis

LaCoGSEA is a powerful tool designed to interpret the latent space of autoencoders trained on gene expression data. It identifies biological pathways associated with each latent dimension through Pearson correlation and GSEA, providing a systematic way to understand what features your model has learned.

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
- **Visual Interpretation**: Automatically generates a "Top Pathways Heatmap" focusing on significant (FDR < 0.05) biological signals.
- **Built-in Pathways**: Quick access to `kegg`, `reactome`, `go_bp`, and `c6` via aliases.

---

## 💻 CLI Usage (For Power Users)

### 1. Installation & Setup
```bash
pip install -e .
lacogsea setup --yes
```

### 2. Run Full Pipeline
```bash
# Run the built-in demo
lacogsea run

# Run with custom data
lacogsea run --train-csv my_data.csv --gene-set kegg --dim 32
```

### 3. Output Structure
`result/{gmt_name}_{timestamp}/`
- `nes.tsv`: Master matrix of NES scores (Dimensions x Pathways).
- `pathway_activity.tsv`: Sample-level activity matrix (Samples x Pathways).
- `top_pathways_heatmap.png`: High-quality summary visualization.
- `all_gsea_reports.zip`: Packaged raw GSEA HTML reports.

---

## 🛠️ Advanced Features Explained

### Smart Resume Mode
If your job crashes, pointing to the same output folder will skip already completed dimensions.

### Automatic Log Detection
If the maximum value of your input data is `> 50`, it automatically applies `Log2(x+1)`.

---

## ⚖️ Third-party software

LaCoGSEA interfaces with the Gene Set Enrichment Analysis (GSEA) software developed by the Broad Institute. Use of GSEA is subject to the Broad Institute GSEA License and is limited to academic, non-commercial use.

## 📄 License
MIT License.
