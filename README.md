# LaCoGSEA: Latent Correlation-gene set enrichment analysis

LaCoGSEA is a powerful tool designed to interpret the latent space of autoencoders trained on gene expression data. It identifies biological pathways associated with each latent dimension through Pearson correlation and GSEA, providing a systematic way to understand what features your model has learned.

---

### 📋 Prerequisites
- **Python**: 3.8 to 3.13
  - **Recommended for GUI**: Python 3.8 to 3.11
  - **⚠️ Python 3.12+ Users**: The GUI has known compatibility issues due to Gradio framework limitations. 
    - **Recommended**: Use **CLI mode** (see below) or downgrade to **Python 3.11**
- **Java**: JRE 11+ or 17+ (Will be installed automatically if missing)

---

## 🎨 User Interface (For Python 3.8-3.11)

> **⚠️ Python 3.12+ Users**: GUI is **not compatible**. Please use [CLI mode](#️-cli-usage-recommended-for-python-312) instead.

The easiest way to use LaCoGSEA is through its built-in Graphical User Interface. 

### 🚀 Quick Start
1. **Double-click `LaCoGSEA_run.bat`**.
2. If Python is missing, follow the download link provided in the console.
3. Your browser will open the interface automatically.

### ❓ Troubleshooting Python
LaCoGSEA requires Python 3.8-3.13. If you don't have it:
- [Download Python from here](https://www.python.org/downloads/). **Crucial**: Tick "Add Python to PATH" during install.

*Note: Environment checks (like Java) and data transformations are handled automatically upon clicking 'Run' in the GUI.*

### Manual Launch (Already Installed)
If you have already run the automatic script once, activate the environment based on your terminal:

- **CMD**: `.venv\Scripts\activate`
- **PowerShell**: `.\.venv\Scripts\Activate.ps1`
- **Linux / macOS / Git Bash**: `source .venv/bin/activate`  (or `source .venv/Scripts/activate` on Windows Git Bash)

Then run:
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

## 🛠️ CLI Usage (Recommended for Python 3.12+)

**✅ Fully compatible with all Python versions (3.8-3.13)**

For users who prefer the terminal, or for **Python 3.12+ users** where the GUI is not compatible.

### 1. Initial Setup
If you haven't run the `LaCoGSEA_run.bat` (Windows), you can initialize the environment manually:

```bash
# 1. Create environment
python -m venv .venv

# 2. Activate based on your terminal:
# CMD:          .venv\Scripts\activate
# PowerShell:   .\.venv\Scripts\Activate.ps1
# Linux/macOS:  source .venv/bin/activate

# 3. Install core dependencies (Fastest way)
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu --prefer-binary
python -m pip install -r requirements.txt --prefer-binary
python -m pip install -e . --no-deps
```

### 2. Basic Commands
Once the environment is activated, the `lacogsea` command is ready.

### 1. Setup Environment
Ensure Java is ready:
```bash
lacogsea setup --yes
```

### 2. Run Full Pipeline
To start the analysis with **built-in example data** and default settings, simply run:
```bash
lacogsea run
```
*Tip: This is the best way to verify your installation is working correctly.*

---

## 📂 Built-in Example Data
To help you get started immediately, LaCoGSEA includes a pre-processed dataset (**GSE126848**) from the NCBI GEO database.
- **Source**: Liver biopsy samples from patients with NASH and healthy controls.
- **Status**: 
    - **Gene Symbols**: All gene IDs have been mapped to standardized symbols.
    - **Pre-log**: The data is already **Log2-transformed** (`Log2(x + 1)`).
- **Format**: Standard CSV where **Rows** are samples and **Columns** are gene symbols.

---

### 📋 CLI Parameters
If you want to use your own data or change settings, use the following arguments. **If an argument is omitted, the default value below is used.**

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--train-csv` / `--train` | *(Example)* | Path to training data (CSV, TXT, or TSV). <br> **Rows**: Samples, **Columns**: Genes. <br> **Note**: Gene Symbols recommended. Example uses **GSE126848**. |
| `--dim` | `4` | Number of latent dimensions to extract. |
| `--epochs` | `100` | AE training epochs. |
| `--batch-size` | `128` | Training batch size. |
| `--gene-set` | `kegg` | Gene set alias (`kegg`, `go_bp`, etc.) or custom `.gmt` path. |
| `--scoring-scheme`| `weighted`| GSEA scoring method (`weighted`, `classic`, etc.). |
| `--output` | `result` | Output directory. |
| `--permutations`| `1000` | GSEA permutation count. |
| `--workers` | `None` | Max parallel GSEA processes (auto-calculated if None). |
| `--min-size` | `15` | Minimum gene set size. |
| `--max-size` | `500` | Maximum gene set size. |
| `--no-make-sets`| `False` | Disable detailed GSEA reports (faster). |

Example with custom parameters:
```bash
lacogsea run --dim 10 --epochs 200 --gene-set go_bp
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

## 🔬 Evaluation & Figure Generation

The `evaluation/` directory contains the source code used to generate the figures presented in the manuscript. These scripts are organized into encapsulated pipelines for reproduction.

### 📦 Installation for Evaluation

To run the evaluation scripts, ensure you have installed the package and all additional dependencies:

```bash
# Install core package
pip install -e .

# Install evaluation dependencies
pip install -r requirements.txt
```

### 🚀 Running Figure Pipelines

The scripts for each figure are located in their respective subdirectories within `evaluation/`:

| Figure | Directory | Description |
| :--- | :--- | :--- |
| **Figure 2** | `evaluation/figure2_stability/` | **Stability & Negative Control**: Saturation analysis and specificity verification. |
| **Figure 3** | `evaluation/figure3_subtypes/` | **Biological Representation**: Breast cancer subtype clustering. |
| **Figure 4** | `evaluation/figure4_clinical/` | **Clinical Relevance**: Survival analysis and cross-cohort generalization. |
| **Figure 5** | `evaluation/figure5_benchmarking/` | **Benchmarking**: Comparison against standard DE/GSEA baselines. |

### 📂 Prerequisites for Data
By default, these scripts attempt to use processed results. If you are generating results from scratch, ensure you have downloaded the required datasets (DLBCL, Heart Failure, AD, Trauma) as described in the supplementary documentation.

---

## 📄 License & Attribution

- **License**: LaCoGSEA is released under the MIT License.
- **Third-party software**: This tool interfaces with the Broad Institute's **GSEA software**. User of GSEA is subject to GSEA's license (Academic use only).

