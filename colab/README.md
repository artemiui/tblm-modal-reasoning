# 🚀 Google Colab Cloud Execution Guide

This directory (`colab/`) contains the complete cloud execution suite for the **Modal Logic Mechanistic Interpretability** project in **Google Colab** using cloud GPU acceleration.

> [!NOTE]
> **Unified Free-Tier Colab Master Notebook**: Because Google Colab Free Tier does not allow connecting multiple concurrent runtimes, all experiment stages (Part A Circuit Discovery, Part B Macroscopic Staging, Comparative Propositional Baseline, and Publication Visualizations) have been unified into a single all-in-one master notebook: [`modal_logic_reasoning_master.ipynb`](file:///C:/Users/Art/Documents/arcega-sfcon/colab/modal_logic_reasoning_master.ipynb) (and [`00_quickstart_colab_runner.ipynb`](file:///C:/Users/Art/Documents/arcega-sfcon/colab/00_quickstart_colab_runner.ipynb)). You can run the entire research suite from start to finish on a single T4 GPU runtime session with zero reconnection overhead!

---

## 📂 Colab Directory Layout

| File | Description | Recommended GPU |
| :--- | :--- | :--- |
| [`modal_logic_reasoning_master.ipynb`](file:///C:/Users/Art/Documents/arcega-sfcon/colab/modal_logic_reasoning_master.ipynb) | **Combined Master All-in-One Notebook**: Unifies Environment Setup, Universal Model Loading (`2B`, `4B`, `9B`), Part A CMA Circuit Discovery, Part B 4-Region Staging & Axioms, Comparative Baseline, Inline Publication Figures, Sufficiency Tables, and Result Export. | **Free Tier T4** / L4 / A100 |
| [`00_quickstart_colab_runner.ipynb`](file:///C:/Users/Art/Documents/arcega-sfcon/colab/00_quickstart_colab_runner.ipynb) | **Master Runner (Quickstart)**: Identical unified all-in-one runner for standard GitHub Colab quick-launch. | **Free Tier T4** / L4 / A100 |
| [`01_part_a_modal_circuit_discovery.ipynb`](file:///C:/Users/Art/Documents/arcega-sfcon/colab/01_part_a_modal_circuit_discovery.ipynb) | **Part A Deep Dive**: Standalone walkthrough for Controlled Mediation Analysis (CMA) activation patching, discovering 7 head families (`MOH`, `MPH`, `CRH`, etc.), sufficiency tables, and publication circuit diagram. | T4 / L4 / A100 |
| [`02_part_b_macroscopic_mechanisms.ipynb`](file:///C:/Users/Art/Documents/arcega-sfcon/colab/02_part_b_macroscopic_mechanisms.ipynb) | **Part B Deep Dive**: Standalone walkthrough for 4-Region MLP Staging, Token-wise Residual Patching, Accessible vs Inaccessible Fact Retrospection contrast, and 11 modal rule/axiom categories (**B, D, 4, 5, K, T**). | T4 / L4 / A100 |
| [`03_comparative_propositional_vs_modal.ipynb`](file:///C:/Users/Art/Documents/arcega-sfcon/colab/03_comparative_propositional_vs_modal.ipynb) | **Comparative Baseline**: Standalone direct side-by-side benchmark comparing Propositional Logic vs. Modal Logic across circuits, staging, and gating. | Any (CPU / GPU) |
| [`requirements-colab.txt`](file:///C:/Users/Art/Documents/arcega-sfcon/colab/requirements-colab.txt) | Pinned Python packages optimized for the Google Colab runtime. | — |
| [`colab_utils.py`](file:///C:/Users/Art/Documents/arcega-sfcon/colab/colab_utils.py) | Helper module for environment detection, GPU VRAM telemetry, Google Drive mounting, and results archiving. | — |

---

## ⚡ Quickstart: Running in Google Colab (Free Tier)

### Step 1: Open Master Notebook in Colab
1. Navigate to [Google Colab](https://colab.research.google.com).
2. Select **GitHub** tab and paste the repository URL: `https://github.com/artemiui/tblm-modal-reasoning`
3. Select `colab/modal_logic_reasoning_master.ipynb` (or `colab/00_quickstart_colab_runner.ipynb`).

### Step 2: Configure Hardware Accelerator
1. In the Colab top menu, go to **Runtime** $\rightarrow$ **Change runtime type**.
2. Under **Hardware accelerator**, select **T4 GPU** (available on the Free Tier).
3. Click **Save**.

### Step 3: Run All Cells Sequentially
- Run the cells in order from top to bottom.
- Dependencies install automatically via `!pip install -q -r colab/requirements-colab.txt`.
- Select your target model preset (`Qwen/Qwen3.5-2B`, `Qwen/Qwen3.5-4B`, or `Qwen/Qwen3.5-9B`).

### Step 4: Inspect Inline Data & Visualizations (Matching Reference Repos)
The combined master notebook renders all outputs in the exact format of the reference repos:
- **Sample Inspection**: Clean vs. Counterfactual prompts, truth valuations, and answer token pairs `(Clean Target, Corrupt Target)`.
- **Part A (Hong et al. 2025)**:
  * 2D Layer $\times$ Head Indirect Effect ($\text{cLD}$) Heatmap.
  * Discovered Modal Head Taxonomy breakdown across 7 families (`MOH`, `MPH`, `CRH`, `QRLH`, `QRMH`, `FPH`, `DH`).
  * Negative-Control Specificity Test ($p < 0.05$, ratio $> 3\times$).
  * Sufficiency Ablation Matrix Table (Markdown & Pandas) + Horizontal Bar Chart.
  * Publication Circuit Architecture Diagram (`modal_circuit_architecture.png`).
- **Part B (Chen et al. 2026)**:
  * 11 Modal Rule Categories & Modal Axioms (**B, D, 4, 5, K, T**) sample prompt display with 4-region formatting.
  * 4-Region Staged MLP Computation Bar Chart across Early, Middle, and Late layer groups.
  * Residual Stream Information Transmission 2D Heatmap across 9 token types.
  * Selective Fact Retrospection Bar Chart (Accessible vs Inaccessible Worlds contrast ratio).
  * Specialized Attention Heads Taxonomy & Multi-Head Validation Curves ($k=1 \dots 64$).
- **Part C (Comparative Evaluation)**:
  * Side-by-side Comparative Metrics DataFrame.
  * Attention Head Family Allocation Comparison Bar Chart.
  * Retrospection Selectivity Discrimination Bar Chart.

### Step 5: Export Artifacts
- Automatically packages all generated figures, JSON summaries, and CSV tables into `modal_logic_experiment_results.zip` with browser download trigger and optional Google Drive synchronization (`/content/drive/MyDrive/modal_logic_results/`).

---

## 📊 Hardware & VRAM Recommendations

| Model Preset | Parameter Size | Minimum VRAM | Recommended Colab Tier |
| :--- | :--- | :--- | :--- |
| `Qwen/Qwen3.5-2B` | 2.5 Billion | ~4 GB VRAM | **Free Tier (T4 GPU 15GB)** |
| `Qwen/Qwen3.5-4B` | 4.0 Billion | ~8 GB VRAM | **Free Tier (T4 GPU 15GB)** |
| `Qwen/Qwen3.5-9B` | 9.0 Billion | ~18 GB VRAM | **Colab Pro (L4 / A100 GPU)** |

---

## 💡 Troubleshooting & Tips

- **Out of Memory (OOM)**:
  If running on a Free Tier T4 GPU, choose `Qwen/Qwen3.5-2B` or `Qwen/Qwen3.5-4B`.
- **Hugging Face Model Access**:
  Public models (`Qwen3.5`, `Gemma-2`, `Mistral`) download automatically. For gated models, supply your Hugging Face User Access Token in the auth cell.
- **Clearing GPU Memory**:
  ```python
  import torch
  torch.cuda.empty_cache()
  ```
