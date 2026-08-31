# ?? Google Colab Cloud Execution Guide

This directory (`colab/`) contains all notebooks, environment configuration scripts, and utilities designed specifically for executing the **Modal Logic Mechanistic Interpretability** project in **Google Colab** using cloud GPU acceleration.

> [!NOTE]
> **File Safety Guarantee**: This directory is completely isolated. All Colab-specific setup cells, Drive syncing helpers, and interactive widgets reside here, ensuring that non-notebook and local workspace files remain clean and unaffected.

---

## ?? Colab Directory Layout

| File | Description | Recommended GPU |
| :--- | :--- | :--- |
| [`00_quickstart_colab_runner.ipynb`](file:///C:/Users/Art/Documents/arcega-sfcon/colab/00_quickstart_colab_runner.ipynb) | **Master Runner**: Interactive UI form to configure model (`2B`, `4B`, `9B`), select pipeline stage (`part_a`, `part_b`, `all`, `comparative`), run tests, and download results. | T4 / L4 / A100 |
| [`01_part_a_modal_circuit_discovery.ipynb`](file:///C:/Users/Art/Documents/arcega-sfcon/colab/01_part_a_modal_circuit_discovery.ipynb) | **Part A Deep Dive**: Controlled Mediation Analysis (CMA) activation patching, discovering 7 head families (`MOH`, `MPH`, `CRH`, etc.), sufficiency tables, and publication circuit diagram. | T4 / L4 / A100 |
| [`02_part_b_macroscopic_mechanisms.ipynb`](file:///C:/Users/Art/Documents/arcega-sfcon/colab/02_part_b_macroscopic_mechanisms.ipynb) | **Part B Deep Dive**: 4-Region MLP Staging, Token-wise Residual Patching, Accessible vs Inaccessible Fact Retrospection contrast, and 11 modal rule/axiom categories (**B, D, 4, 5, K, T**). | T4 / L4 / A100 |
| [`03_comparative_propositional_vs_modal.ipynb`](file:///C:/Users/Art/Documents/arcega-sfcon/colab/03_comparative_propositional_vs_modal.ipynb) | **Comparative Baseline**: Direct side-by-side benchmark comparing Propositional Logic vs. Modal Logic across circuits, staging, and gating. | Any (CPU / GPU) |
| [`requirements-colab.txt`](file:///C:/Users/Art/Documents/arcega-sfcon/colab/requirements-colab.txt) | Pinned Python packages optimized for the Google Colab runtime. | ? |
| [`colab_utils.py`](file:///C:/Users/Art/Documents/arcega-sfcon/colab/colab_utils.py) | Helper module for environment detection, GPU VRAM telemetry, Google Drive mounting, and results archiving. | ? |

---

## ?? Quickstart: Running in Google Colab

### Step 1: Open Notebook in Colab
1. Navigate to [Google Colab](https://colab.research.google.com).
2. Select **GitHub** tab and paste the repository URL: `https://github.com/artemiui/tblm-modal-reasoning`
3. Select `colab/00_quickstart_colab_runner.ipynb` (or any individual notebook).

### Step 2: Configure Hardware Accelerator
1. In the Colab top menu, go to **Runtime** $\rightarrow$ **Change runtime type**.
2. Under **Hardware accelerator**, select **T4 GPU** (or **A100 / L4 GPU** if on Colab Pro).
3. Click **Save**.

### Step 3: Execute Setup & Install Dependencies
Run the initial setup cells in the notebook. Dependencies will automatically install via:
```bash
!pip install -q -r colab/requirements-colab.txt
```

### Step 4: Run Experiments & View Results
- Use the Colab interactive Form dropdowns to choose the model (`Qwen/Qwen3.5-2B`, `Qwen/Qwen3.5-4B`, or `Qwen/Qwen3.5-9B`) and the desired pipeline stage.
- Visualizations (heatmaps, bar charts, and circuit architecture diagrams) and sufficiency tables render directly inline.

### Step 5: Export Artifacts
- The final cell packages all figures, JSON summaries, and CSV tables into a zip archive and triggers a browser download, or syncs them directly to `/content/drive/MyDrive/modal_logic_results/`.

---

## ?? Hardware & VRAM Recommendations

| Model Preset | Parameter Size | Minimum VRAM | Recommended Colab Tier |
| :--- | :--- | :--- | :--- |
| `Qwen/Qwen3.5-2B` | 2.5 Billion | ~4 GB VRAM | **Free Tier (T4 GPU 15GB)** |
| `Qwen/Qwen3.5-4B` | 4.0 Billion | ~8 GB VRAM | **Free Tier (T4 GPU 15GB)** |
| `Qwen/Qwen3.5-9B` | 9.0 Billion | ~18 GB VRAM | **Colab Pro (L4 / A100 GPU)** |

---

## ??? Troubleshooting & Tips

- **Out of Memory (OOM)**:
  If running on a Free Tier T4 GPU with a larger model, decrease the batch size / sample count in the notebook cell (`n_per_type=10` instead of `30`), or select `Qwen/Qwen3.5-2B`.
- **Hugging Face Model Access**:
  Most models (`Qwen3.5`, `Gemma-2`, `Mistral`) download automatically from Hugging Face Hub. For gated checkpoints, provide your Hugging Face User Access Token in the auth cell.
- **Clearing GPU Memory**:
  If re-running cells multiple times, run:
  ```python
  import torch
  torch.cuda.empty_cache()
  ```
