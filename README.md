# LaCoGSEA: Latent Correlation-gene set enrichment analysis

LaCoGSEA is a powerful tool designed to interpret the latent space of autoencoders trained on gene expression data. It identifies biological pathways associated with each latent dimension through Pearson correlation and GSEA, providing a systematic way to understand what features your model has learned.

## ✨ Key Features

- **Consolidated GSEA Engine**: Powered exclusively by the high-performance GSEA Java CLI (bundled or system-installed).
- **Auto Data Transformation**: Intelligently detects data scale and applies `Log2(x + 1)` only when necessary.
- **Robustness & Resume**: Built-in "Resume Mode" skips already completed dimensions if a run is interrupted.
- **Detailed Error Diagnosis**: Captures and reports specific GSEA errors (e.g., empty gene sets) directly in the console.
- **Visual Interpretation**: Automatically generates a "Top Pathways Heatmap" focusing on the most significant (FDR < 0.05) and recurring biological signals.
- **Smart Directory Naming**: Each run is isolated in a unique folder named by its GMT file and timestamp.
- **Built-in Pathways**: Quick access to `kegg`, `reactome`, `go_bp`, and `c6` via aliases.

---

## 🚀 Quick Start

### 1. Installation

```bash
# Install for development
pip install -e .
```

### 2. Environment Setup
LaCoGSEA requires **Java 11+**. Run the setup check to verify your environment:
```bash
lacogsea setup
```
If Java is missing, you can install a portable version automatically into the package directory:
```bash
lacogsea install-java
```

---

## 🎨 Graphical User Interface (GUI)

For users who prefer a visual interface (e.g., biologists), LaCoGSEA provides a built-in web-based GUI.

### Windows (Quick Start)
1. Double-click `install_windows.bat` (only needed once).
2. Double-click `run_gui.bat`.
3. Your browser will automatically open the LaCoGSEA interface.

### Manual Launch
```bash
lacogsea-gui
```
The GUI allows you to upload datasets, select gene sets, adjust parameters, and view heatmaps/NES tables interactively.

---

## 💻 Usage

### Subcommand 1: `run` (Standard Workflow)
Executes the full pipeline: Train -> Correlation -> GSEA -> Summary -> Activity -> Visualization.

```bash
# Run the built-in demo (uses bundled GSE126848 data and KEGG pathways)
lacogsea run

# Run with a specific built-in gene set (kegg, go_bp, reactome, c6)
lacogsea run --gene-set go_bp

# Run with custom data and a custom GMT file
lacogsea run --train-csv train.csv --test-csv test.csv --gene-set ./my_pathways.gmt --dim 32
```

**Output Structure:**
`result/{gmt_name}_{timestamp}/`
- `nes.tsv`: Master matrix of NES scores (Dimensions x Pathways).
- `pathway_activity.tsv`: Sample-level pathway activity matrix (Samples x Pathways).
- `top_pathways_heatmap.png`: High-quality summary visualization.
- `gsea/`: Raw output folders for every dimension.

### Advanced Parameters for `run`
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--dim` | 32 | Number of latent dimensions. |
| `--epochs` | 100 | Training epochs for the Autoencoder. |
| `--permutations`| 1000 | GSEA permutations. |
| `--scoring-scheme` | `weighted` | GSEA scoring: `weighted`, `classic`, `weighted_p2`, etc. |
| `--no-make-sets` | False | Skip generating detailed GSEA HTML reports (saves space). |

---

### Subcommand 2: `gsea` (Standalone GSEA)
Run GSEA Preranked on a single correlation `.rnk` file.

```bash
lacogsea gsea --rnk input.rnk --gene-set reactome --output ./gsea_out
```

---

### Subcommand 3: `summarize` & `activity`
Manually aggregate results or compute activity matrices from existing files.

```bash
# Build NES matrix from a folder of GSEA results
lacogsea summarize --gsea-dir ./my_results/gsea --dims 32 --output nes.tsv

# Calculate sample-level activities
lacogsea activity --embedding embedding.csv --nes nes.tsv --output activity.tsv
```

---

## 🛠️ Advanced Features Explained

### Smart Resume Mode
If your job crashes or you want to rerun with the same settings, point to the same output folder. LaCoGSEA will detect existing results and display `- Cached`, skipping directly to the remaining dimensions.

### Automatic Log Detection
LaCoGSEA checks the maximum value of your input data. If `max > 50`, it assumes the data is in raw count or TPM/FPKM format and applies `Log2(x+1)`. Otherwise, it assumes the data is already pre-logged and proceeds.

### Heatmap Visualization Logic
The automatically generated heatmap (`top_pathways_heatmap.png`) displays the top 5 pathways for the first 16 dimensions (FDR < 0.05). It prioritizes pathways that appear repeatedly across different dimensions, grouping them at the top to highlight consistently captured biological signals.

---

## ⚖️ Third-party software

LaCoGSEA interfaces with the Gene Set Enrichment Analysis (GSEA) software developed by the Broad Institute.

GSEA is a third-party tool and is not included as part of the LaCoGSEA source distribution core. When required, LaCoGSEA can automatically download the official GSEA Java binary from Broad Institute servers, or users may provide their own installation.

Use of GSEA is governed by the Broad Institute GSEA License and is limited to academic, non-commercial use.

## 📄 License
MIT License.
