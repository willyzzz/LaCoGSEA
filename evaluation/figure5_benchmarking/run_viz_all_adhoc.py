
import subprocess
import sys
from pathlib import Path

datasets = ["GSE10846", "GSE48350", "GSE11375", "GSE126848", "GSE116250"]
script_path = Path("scripts/figure2/figure2_visualize.py")

for ds in datasets:
    print(f"Running visualization for {ds}...")
    cmd = [sys.executable, str(script_path), "--dataset", ds]
    subprocess.run(cmd, check=True)

print("Done all.")
