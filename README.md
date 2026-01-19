# LaCoGSEA: Latent Correlation-gene set enrichment analysis

LaCoGSEA is a powerful tool designed to interpret the latent space of autoencoders trained on gene expression data. It identifies biological pathways associated with each latent dimension through Pearson correlation and GSEA, providing a systematic way to understand what features your model has learned.

---

## � Quick Start (Windows)

LaCoGSEA features a **Zero-Setup** environment manager. You don't need to install Python or manage libraries yourself.

1. **Download & Extract** the project.
2. **Double-click `LaCoGSEA_run.bat`**.
3. **Wait for First-run**: The first launch will automatically configure a dedicated, verified environment. This takes about **5-10 minutes** depending on your internet speed.
4. **Instant Launch**: Once setup is complete, future runs will start instantly.

---

## 🎨 User Interface & 🛠️ CLI

Both the Graphical Interface and Command Line Interface are ready to use out of the box.

### 1. Graphical User Interface (GUI)
Simply run the `LaCoGSEA_run.bat`. It will automatically launch the web interface in your default browser.

### 2. Command Line Interface (CLI)
You can use the CLI directly via the internally managed environment. Open your terminal in the project folder.

#### Basic Usage
```bash
.python_runtime\python.exe -m lacogsea.cli run --dim 10 --epochs 200
```

#### Run with Example Data (GSE126848)
To verify everything is working, you can run the full pipeline using the built-in **GSE126848** (NASH vs Healthy liver) dataset:
```bash
.python_runtime\python.exe -m lacogsea.cli run --dim 4 --epochs 100 --gene-set kegg
```
*No path is required for the built-in data.*

#### Full Parameter List
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--train` / `--train-csv` | *(Example)* | Path to training data (CSV, TXT, or TSV). <br> **Rows**: Samples, **Columns**: Genes. |
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

#### Step-by-Step (Internal Commands)
If you need more control, you can run individual steps:
- `.python_runtime\python.exe -m lacogsea.cli train`: Train Autoencoder.
- `.python_runtime\python.exe -m lacogsea.cli rnks`: Generate `.rnk` files.
- `.python_runtime\python.exe -m lacogsea.cli gsea`: Run GSEA.
- `.python_runtime\python.exe -m lacogsea.cli summarize`: Aggregate results.

---

## ✨ Key Features

- **Zero-Setup Environment**: Self-contained runtime ensures it works on any Windows machine without version conflicts.
- **Consolidated GSEA Engine**: Powered by high-performance GSEA Java CLI (automatically configured).
- **Auto Data Transformation**: Intelligently detects data scale and applies `Log2(x + 1)` only when necessary.
- **Robustness & Resume**: Built-in "Resume Mode" skips already completed dimensions if a run is interrupted.

---

## 📂 Built-in Example Data
To help you get started immediately, LaCoGSEA includes a pre-processed dataset (**GSE126848**) from the NCBI GEO database.
- **Source**: Liver biopsy samples from patients with NASH and healthy controls.
- **Status**: Already Log2-transformed and mapped to gene symbols.

---

---

## 📑 Result Interpretation
The `result/` folder will contain:
- `nes.tsv`: The Normalized Enrichment Score (NES) matrix.
- `pathway_activity.tsv`: The calculated Pathway Activity Score for each sample.
- `top_pathways_heatmap.png`: High-resolution summary plot.

---

## 📄 License & Attribution

- **LaCoGSEA License**: Released under the MIT License.
- **Third-Party Software Attribution**:
  - This tool is a **third-party wrapper** and high-level interface. It is **not** an official product of the Broad Institute.
  - LaCoGSEA interfaces with the **GSEA (Gene Set Enrichment Analysis)** software developed by the Broad Institute.
  - Users of LaCoGSEA are bound by the [GSEA license terms](https://www.gsea-msigdb.org/gsea/license.jsp) (Free for academic use; commercial users require a license from the Broad Institute).
  - All GSEA algorithms and MSigDB gene sets are property of the Broad Institute and UC San Diego.
