"""
Colab Utilities for Modal Logic Mechanistic Interpretability
Provides helper functions for environment setup, GPU memory telemetry,
Hugging Face authentication, file visualization, and result synchronization to Google Drive.
"""
from __future__ import annotations

import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Optional, Union, List, Dict, Any

try:
    import torch
except ImportError:
    torch = None


def is_colab() -> bool:
    """Checks if code is executing inside a Google Colab notebook environment."""
    return "google.colab" in sys.modules or "COLAB_GPU" in os.environ


def setup_colab_environment(
    repo_root: Optional[Union[str, Path]] = None,
    mount_drive: bool = False,
    gdrive_mount_point: str = "/content/drive",
    create_output_dirs: bool = True,
) -> Path:
    """
    Configures paths, mounts Google Drive if requested, and adds project packages to sys.path.
    
    Returns:
        Path: Absolute path to project workspace root.
    """
    if is_colab() and mount_drive:
        try:
            from google.colab import drive
            print("Mounting Google Drive at", gdrive_mount_point, "...")
            drive.mount(gdrive_mount_point)
            print("[OK] Google Drive mounted successfully.")
        except Exception as e:
            print(f"[WARN] Could not mount Google Drive: {e}")

    # Determine root directory
    if repo_root is not None:
        root_path = Path(repo_root).resolve()
    else:
        # If in colab/ directory, parent is repo root
        current_dir = Path.cwd()
        if (current_dir / "modal-logic-mi").exists():
            root_path = current_dir
        elif (current_dir.parent / "modal-logic-mi").exists():
            root_path = current_dir.parent
        elif Path("/content/arcega-sfcon").exists():
            root_path = Path("/content/arcega-sfcon")
        elif Path("/content/tblm-modal-reasoning").exists():
            root_path = Path("/content/tblm-modal-reasoning")
        else:
            root_path = current_dir

    # Add core packages to sys.path
    modal_mi_dir = root_path / "modal-logic-mi"
    transformer_circuit_dir = root_path / "modal-logic-transformer-circuit"
    
    for d in [root_path, modal_mi_dir, transformer_circuit_dir]:
        if d.exists() and str(d) not in sys.path:
            sys.path.insert(0, str(d))

    if create_output_dirs:
        (root_path / "results" / "colab_runs").mkdir(parents=True, exist_ok=True)
        (modal_mi_dir / "results" / "part_a").mkdir(parents=True, exist_ok=True)
        (modal_mi_dir / "results" / "part_b").mkdir(parents=True, exist_ok=True)

    print(f"[OK] Environment configured. Workspace root: {root_path}")
    print_gpu_info()
    return root_path


def print_gpu_info() -> None:
    """Prints GPU accelerator specifications and memory status."""
    print("=" * 60)
    print("  HARDWARE ACCELERATION TELEMETRY")
    print("=" * 60)
    if torch is not None and torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        print(f"CUDA is Available: True ({num_devices} device(s) found)")
        for i in range(num_devices):
            name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            total_vram = props.total_memory / (1024 ** 3)
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
            print(f"  [GPU {i}] {name}")
            print(f"         Total VRAM     : {total_vram:.2f} GB")
            print(f"         Allocated VRAM : {allocated:.2f} GB")
            print(f"         Reserved VRAM  : {reserved:.2f} GB")
    else:
        print("CUDA is Available: False (Running on CPU)")
    print("=" * 60)


def login_huggingface(token: Optional[str] = None) -> bool:
    """
    Authenticates with Hugging Face Hub using an access token.
    Useful when accessing gated models or writing checkpoints.
    """
    try:
        from huggingface_hub import login
        if token and token.strip():
            login(token=token.strip(), add_to_git_credential=True)
            print("[OK] Hugging Face Hub authentication successful.")
            return True
        else:
            print("[INFO] No Hugging Face token provided. Public models will load without token.")
            return True
    except Exception as e:
        print(f"[WARN] Hugging Face authentication note: {e}")
        return False


def display_image(image_path: Union[str, Path], width: Optional[int] = None) -> None:
    """Displays an image file within Jupyter/Colab output cell."""
    path = Path(image_path)
    if not path.exists():
        print(f"[WARN] Image file not found: {path}")
        return
    try:
        from IPython.display import Image, display
        display(Image(filename=str(path), width=width))
    except Exception:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg
            img = mpimg.imread(str(path))
            plt.figure(figsize=(10, 8))
            plt.imshow(img)
            plt.axis("off")
            plt.show()
        except Exception as e:
            print(f"[INFO] Could not display image inline: {e}")


def export_results_zip(
    output_dir: Union[str, Path] = "results",
    zip_filename: str = "modal_logic_results.zip",
    trigger_download: bool = True,
) -> Path:
    """
    Archives experiment results into a zip file and triggers browser download in Colab.
    """
    source_dir = Path(output_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory {source_dir} does not exist.")

    zip_path = Path(zip_filename).resolve()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(source_dir):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(source_dir.parent if source_dir.parent.exists() else source_dir)
                zipf.write(file_path, arcname)

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"[OK] Archive created: {zip_path} ({size_mb:.2f} MB)")

    if is_colab() and trigger_download:
        try:
            from google.colab import files
            print("Triggering browser download in Colab...")
            files.download(str(zip_path))
        except Exception as e:
            print(f"[INFO] Automatic browser download note: {e}")

    return zip_path


def sync_to_gdrive(
    source_dir: Union[str, Path] = "results",
    gdrive_dest_dir: str = "/content/drive/MyDrive/modal_logic_results",
) -> bool:
    """
    Copies experiment figures, tables, and JSON summaries to Google Drive.
    """
    src = Path(source_dir)
    dest = Path(gdrive_dest_dir)
    if not src.exists():
        print(f"[WARN] Source directory {src} does not exist.")
        return False
    try:
        dest.mkdir(parents=True, exist_ok=True)
        for item in src.glob("**/*"):
            if item.is_file():
                rel = item.relative_to(src)
                target = dest / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)
        print(f"[OK] Successfully synced {src} -> {dest}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed syncing to Google Drive: {e}")
        return False
