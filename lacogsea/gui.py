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
import threading
import time
from .run_gsea_java import find_gsea_dir, find_java_cmd
from .utils import install_internal_java

# Setup logs for GUI display
# We use a custom stream to capture logs in real-time
class LogStringStream(io.StringIO):
    def __init__(self):
        super().__init__()
        self.lines = [""]

    def write(self, s):
        if not s: return
        for char in s:
            if char == '\n':
                self.lines.append("")
            elif char == '\r':
                self.lines[-1] = ""
            else:
                self.lines[-1] += char
        super().write(s)

    def get_logs(self):
        # Filter out trailing empty lines to keep it clean, but keep the current line
        return "\n".join(self.lines)

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
    no_make_sets,
    workers
):
    log_stream = LogStringStream()
    handler = logging.StreamHandler(log_stream)
    logger = logging.getLogger("lacogsea")
    
    # Clear existing handlers to avoid duplicates
    for h in logger.handlers[:]:
        logger.removeHandler(h)
        
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    accumulated_logs = ""
    result_container = {"nes_path": None, "error": None, "done": False}

    def get_log_update():
        return log_stream.get_logs()

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
    
    # Resolve gene set
    if custom_gmt is not None:
        gene_set = custom_gmt.name
        logger.info(f"Using custom GMT: {os.path.basename(gene_set)}")
    else:
        gene_set = resolve_gene_set(gene_set_alias)
        logger.info(f"Using built-in gene set: {gene_set_alias}")

    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    gmt_stem = Path(gene_set).stem
    output_dir = Path("result/gui_run") / f"{gmt_stem}_d{dim}_{timestamp}"

    def worker():
        try:
            logger.info("--- Environment Check ---")
            gsea_dir = find_gsea_dir()
            if not gsea_dir:
                raise ValueError("GSEA vendor directory not found. Please reinstall the package.")

            java_cmd = find_java_cmd(gsea_dir)
            java_ready = False
            try:
                import subprocess
                subprocess.run([java_cmd, "-version"], capture_output=True, check=True)
                java_ready = True
                logger.info(f"Java found: {java_cmd}")
            except Exception:
                logger.info("Java not found in system or package. Initiating automatic portable JRE installation...")
                if install_internal_java(gsea_dir):
                    logger.info("Portable JRE installed successfully.")
                    java_ready = True
                else:
                    raise ValueError("Failed to install portable JRE automatically.")

            logger.info("--- Pipeline Starting ---")
            
            res = run_full_pipeline(
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
                make_sets=not no_make_sets,
                workers=int(workers) if workers > 0 else None
            )
            result_container["nes_path"] = res
        except Exception as e:
            result_container["error"] = str(e)
        finally:
            result_container["done"] = True

    # Redirect stdout for progress bars
    old_stdout = sys.stdout
    sys.stdout = log_stream

    t = threading.Thread(target=worker)
    t.start()

    last_log_content = ""
    while not result_container["done"]:
        current_logs = get_log_update()
        if current_logs != last_log_content:
            yield current_logs, None, None, None, None, None
            last_log_content = current_logs
        time.sleep(0.05) # Increased polling frequency for smoother updates

    sys.stdout = old_stdout
    
    final_logs = get_log_update()
    if result_container["error"]:
        logger.error(f"Error: {result_container['error']}")
        yield final_logs + f"\nError: {result_container['error']}", None, None, None, None, None
    else:
        nes_path = result_container["nes_path"]
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
            final_logs, 
            nes_df.head(20), 
            str(heatmap_path) if heatmap_path.exists() else None,
            str(nes_path),
            str(activity_path) if activity_path.exists() else None,
            str(zip_path)
        )
    logger.removeHandler(handler)

def main():
    css = """
    #log-container { 
        background-color: #ffffff !important; 
        border: 2px solid #000000 !important;
        border-radius: 8px !important;
    }
    #log-container textarea {
        background-color: #ffffff !important;
        color: #000000 !important; /* Pure Black */
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        line-height: 1.6 !important;
    }
    footer {display: none !important;}
    .show-api {display: none !important;}
    .upload-container .or, .upload-container .upload-button {font-size: 0 !important;}
    .upload-container .or::after {content: 'OR' !important; font-size: 14px !important;}
    .upload-container .upload-button::after {content: 'Upload' !important; font-size: 14px !important;}
    .upload-container p {font-size: 0 !important;}
    .upload-container p::after {content: 'Drop File Here' !important; font-size: 16px !important;}
    """
    with gr.Blocks(title="LaCoGSEA Graphical Interface", css=css) as demo:
        gr.Markdown("# 🚀 LaCoGSEA: Latent Correlation-GSEA")
        gr.Markdown("Interpret your Autoencoder latent space without touching the command line.")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 1. Data Upload")
                data_input = gr.File(label="Dataset (CSV/TXT/TSV) - Leave empty to use example data", file_types=[".csv", ".txt", ".tsv"])
                gr.Markdown("""
                **📋 Data Requirements:**
                - **Gene Symbols**: e.g., *TP53* (Required for built-in Gene Sets).
                - **Ensembl IDs**: e.g., *ENSG...* (**Will cause an error and stop**).
                - **Rows**: Samples | **Columns**: Genes.
                - **Example**: Powered by **GSE126848**.
                """)
                
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
                    dim = gr.Number(label="Latent Dimensions", value=4, precision=0)
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
                with gr.Row():
                    workers = gr.Slider(minimum=0, maximum=16, step=1, value=0, label="Parallel Workers (0 = Auto)")
                    no_make_sets = gr.Checkbox(label="Disable detailed HTML reports (Faster)", value=True)
                
                run_btn = gr.Button("🔥 Run Full Pipeline", variant="primary")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 💡 Execution Logs")
                log_output = gr.Textbox(
                    label="Terminal Output", 
                    lines=14, 
                    interactive=False, 
                    autoscroll=True, 
                    elem_id="log-container"
                )
            
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
            inputs=[data_input, gs_alias, custom_gmt, dim, epochs, batch_size, perms, min_sz, max_sz, scoring, no_make_sets, workers],
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
