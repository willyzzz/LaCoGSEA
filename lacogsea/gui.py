from __future__ import annotations
import os
import gradio as gr
import pandas as pd
from pathlib import Path
from .pipeline import run_full_pipeline
from .cli import resolve_gene_set
from .run_gsea_java import find_gsea_dir, find_java_cmd
from .utils import install_internal_java
import logging
import sys
import io

# Setup logs for GUI display
# We use a custom stream to capture logs in real-time
class LogStringStream(io.StringIO):
    def __init__(self):
        super().__init__()
        self.new_logs = []

    def write(self, s):
        if s.strip():
            self.new_logs.append(s.strip())
        super().write(s)

def run_pipeline_gui_stream(
    input_file, 
    gene_set_alias, 
    custom_gmt, 
    dim, 
    epochs, 
    batch_size,
    permutations, 
    min_size,
    max_size,
    scoring_scheme,
    no_make_sets
):
    log_stream = LogStringStream()
    handler = logging.StreamHandler(log_stream)
    logger = logging.getLogger("lacogsea")
    
    # Clear existing handlers to avoid duplicates
    for h in logger.handlers[:]:
        logger.removeHandler(h)
        
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    accumulated_logs = []
    
    def get_log_update():
        while log_stream.new_logs:
            accumulated_logs.append(log_stream.new_logs.pop(0))
        return "\n".join(accumulated_logs)

    yield get_log_update(), None, None, None, None, None

    try:
        logger.info("--- Environment Check ---")
        gsea_dir = find_gsea_dir()
        if not gsea_dir:
            logger.error("GSEA vendor directory not found. Please reinstall the package.")
            yield get_log_update(), None, None, None, None, None
            return

        java_cmd = find_java_cmd(gsea_dir)
        java_ready = False
        try:
            import subprocess
            subprocess.run([java_cmd, "-version"], capture_output=True, check=True)
            java_ready = True
            logger.info(f"Java found: {java_cmd}")
        except Exception:
            logger.info("Java not found in system or package. Initiating automatic portable JRE installation...")
            yield get_log_update(), None, None, None, None, None
            if install_internal_java(gsea_dir):
                logger.info("Portable JRE installed successfully.")
                java_ready = True
            else:
                logger.error("Failed to install portable JRE automatically.")
                yield get_log_update(), None, None, None, None, None
                return

        logger.info("--- Pipeline Starting ---")
        
        # Determine input data
        if input_file is None:
            package_dir = os.path.dirname(os.path.abspath(__file__))
            if gene_set_alias == "simulated_test":
                data_path = os.path.join(package_dir, "data", "simulated_unlogged.csv")
                logger.info("Using simulated UN-LOGGED dataset")
            else:
                data_path = os.path.join(package_dir, "data", "gse126848.csv")
                logger.info("Using default example dataset (GSE126848)")
            train_path = test_path = data_path
        else:
            train_path = test_path = input_file.name
            logger.info(f"Using uploaded dataset: {os.path.basename(input_file.name)}")
        
        yield get_log_update(), None, None, None, None, None

        # Resolve gene set
        if custom_gmt is not None:
            gene_set = custom_gmt.name
            logger.info(f"Using custom GMT: {os.path.basename(gene_set)}")
        else:
            gene_set = resolve_gene_set(gene_set_alias)
            logger.info(f"Using built-in gene set: {gene_set_alias}")
        
        yield get_log_update(), None, None, None, None, None

        output_dir = Path("result/gui_run")
        
        nes_path = run_full_pipeline(
            train_csv=train_path,
            test_csv=test_path,
            gene_set=gene_set,
            output_dir=output_dir,
            dim=int(dim),
            epochs=int(epochs),
            batch_size=int(batch_size),
            permutations=int(permutations),
            min_size=int(min_size),
            max_size=int(max_size),
            scoring_scheme=scoring_scheme,
            make_sets=not no_make_sets
        )
        
        yield get_log_update(), None, None, None, None, None

        actual_out_dir = nes_path.parent
        heatmap_path = actual_out_dir / "top_pathways_heatmap.png"
        activity_path = actual_out_dir / "pathway_activity.tsv"
        
        nes_df = pd.read_csv(nes_path, sep="\t", index_col=0)
        
        # Package raw reports
        logger.info("Packaging raw GSEA reports...")
        zip_path = actual_out_dir / "all_gsea_reports.zip"
        import shutil
        shutil.make_archive(str(zip_path).replace(".zip", ""), 'zip', actual_out_dir / "gsea")
        
        logger.info("--- Pipeline Completed Successfully ---")
        yield (
            get_log_update(), 
            nes_df.head(20), 
            str(heatmap_path) if heatmap_path.exists() else None,
            str(nes_path),
            str(activity_path) if activity_path.exists() else None,
            str(zip_path)
        )
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        yield get_log_update(), None, None, None, None, None
    finally:
        logger.removeHandler(handler)

def main():
    with gr.Blocks(title="LaCoGSEA Graphical Interface") as demo:
        gr.Markdown("# 🚀 LaCoGSEA: Latent Correlation-GSEA")
        gr.Markdown("Interpret your Autoencoder latent space without touching the command line.")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 1. Data Upload")
                data_input = gr.File(label="Dataset (CSV) - Leave empty to use example data", file_types=[".csv"])
                
                gr.Markdown("### 2. Gene Set Selection")
                gs_alias = gr.Dropdown(
                    choices=["kegg", "reactome", "go_bp", "c6", "simulated_test"], 
                    value="kegg", 
                    label="Built-in Gene Set / Test Mode"
                )
                custom_gmt = gr.File(label="OR Upload Custom GMT File", file_types=[".gmt"])
                
            with gr.Column():
                gr.Markdown("### 3. Parameters")
                with gr.Row():
                    dim = gr.Number(label="Latent Dimensions", value=32, precision=0)
                    epochs = gr.Number(label="AE Training Epochs", value=100, precision=0)
                    batch_size = gr.Number(label="Batch Size", value=128, precision=0)
                
                with gr.Row():
                    perms = gr.Number(label="GSEA Permutations", value=1000, precision=0)
                    min_sz = gr.Number(label="Gene Set Min Size", value=15, precision=0)
                    max_sz = gr.Number(label="Gene Set Max Size", value=500, precision=0)
                
                scoring = gr.Radio(
                    choices=["weighted", "classic", "weighted_p2"], 
                    value="weighted", 
                    label="GSEA Scoring Scheme"
                )
                no_make_sets = gr.Checkbox(label="Disable detailed HTML reports (Faster)", value=False)
                
                run_btn = gr.Button("🔥 Run Full Pipeline", variant="primary")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 💡 Execution Logs")
                log_output = gr.Textbox(label="Real-time Logs", lines=10, interactive=False)
            
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📊 Results: Top NES Scores")
                nes_table = gr.DataFrame()
                dl_nes = gr.File(label="Download NES Matrix (TSV)")
            with gr.Column():
                gr.Markdown("### 🎨 Visualization: Top Pathways Heatmap")
                heatmap_output = gr.Image(label="Summary Heatmap")
                dl_act = gr.File(label="Download Activity Matrix (TSV)")
                dl_reports = gr.File(label="Download Raw GSEA Reports (ZIP)")

        run_btn.click(
            fn=run_pipeline_gui_stream,
            inputs=[data_input, gs_alias, custom_gmt, dim, epochs, batch_size, perms, min_sz, max_sz, scoring, no_make_sets],
            outputs=[log_output, nes_table, heatmap_output, dl_nes, dl_act, dl_reports]
        )

        gr.Markdown("---")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📜 License")
                gr.Markdown("LaCoGSEA is licensed under the **MIT License**.")
            with gr.Column():
                gr.Markdown("### ⚖️ Third-party Software")
                gr.Markdown(
                    "This tool interfaces with the **GSEA software** developed by the Broad Institute. "
                    "Use of GSEA is subject to the Broad Institute GSEA License (Academic, non-commercial use only). "
                    "[Learn more](https://www.gsea-msigdb.org/)"
                )

    demo.launch(share=False, inbrowser=True)

if __name__ == "__main__":
    main()
