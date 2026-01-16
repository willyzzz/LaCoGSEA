from __future__ import annotations
import os
import sys
import zipfile
import tarfile
import urllib.request
import logging
from pathlib import Path
from typing import Union

LOGGER = logging.getLogger(__name__)

# Adoptium JRE 17 (LTS) Portable Links
JAVA_URLS = {
    "nt": "https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.10%2B7/OpenJDK17U-jre_x64_windows_hotspot_17.0.10_7.zip",
    "posix": "https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.10%2B7/OpenJDK17U-jre_x64_linux_hotspot_17.0.10_7.tar.gz"
}

def download_file(url: str, dest: Path):
    """Download a file using built-in urllib."""
    LOGGER.info(f"Downloading: {url} ...")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        LOGGER.error(f"Download error: {e}")
        raise

def install_internal_java(target_dir: Union[str, Path]) -> bool:
    """
    Downloads and installs a portable JRE into the specified directory.
    Target directory should be the GSEA vendor folder.
    """
    target_dir = Path(target_dir)
    os_type = "nt" if os.name == "nt" else "posix"
    url = JAVA_URLS.get(os_type)
    
    if not url:
        LOGGER.error(f"Unsupported OS for auto-java installation: {os.name}")
        return False

    archive_name = "jre.zip" if os_type == "nt" else "jre.tar.gz"
    archive_path = target_dir / archive_name
    
    try:
        # 1. Download
        download_file(url, archive_path)
        
        # 2. Extract
        LOGGER.info(f"Extracting Java archive...")
        if archive_name.endswith(".zip"):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                # Find the top-level folder in zip
                top_folder = zip_ref.namelist()[0].split('/')[0]
                zip_ref.extractall(target_dir)
        else:
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                top_folder = tar_ref.getnames()[0].split('/')[0]
                tar_ref.extractall(target_dir)
        
        # 3. Rename top folder to 'jdk' for consistency with find_java_cmd
        extracted_path = target_dir / top_folder
        final_path = target_dir / "jdk"
        
        import shutil
        if final_path.exists():
            LOGGER.info(f"Removing old JDK directory: {final_path}")
            shutil.rmtree(final_path, ignore_errors=True)
            
        LOGGER.info(f"Moving {extracted_path} to {final_path}")
        try:
            shutil.move(str(extracted_path), str(final_path))
        except Exception as move_err:
            LOGGER.warning(f"shutil.move failed ({move_err}), trying simple rename...")
            extracted_path.rename(final_path)
        
        # 4. Set executable permissions on Unix
        if os_type == "posix":
            java_bin = final_path / "bin" / "java"
            if java_bin.exists():
                LOGGER.info(f"Setting executable permissions on {java_bin}")
                java_bin.chmod(0o755)
        
        # 5. Cleanup
        archive_path.unlink()
        
        LOGGER.info("[OK] Portable Java installed successfully.")
        return True
        
    except Exception as e:
        LOGGER.error(f"Failed to install portable Java: {e}")
        if archive_path.exists(): archive_path.unlink()
        return False
