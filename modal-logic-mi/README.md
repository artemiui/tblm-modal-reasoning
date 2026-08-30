# Modal Logic Mechanistic Interpretability (ModalLogic-MI)

<p align="center">
  <b>Experimental code, datasets, and discovery pipelines for mechanistic interpretability of modal logical reasoning in Large Language Models.</b>
</p>

---

## 1. Project Overview

This repository extends two foundational mechanistic interpretability studies of propositional logic reasoning to **Modal Logic** ($\\Box$ necessity, $\\Diamond$ possibility, and Kripke possible-world accessibility relations $R \\subseteq W \\times W$):
1. **Part A (Circuit Discovery -- Hong et al. 2025 Extension)**: Controlled Mediation Analysis (CMA) circuit-discovery pipeline adapted for modal-logic prompts, identifying standard reasoning heads along with novel **Modal-Operator Heads (MOH)** and **World-Accessibility Heads (WAH)**. Evaluated on `Mistral-7B`, `Gemma-2-9B`, and `Gemma-2-27B`.
2. **Part B (Mechanistic Principles -- Chen et al. 2026 Extension)**: Macroscopic pattern analysis investigating staged computation across 4 prompt regions (Facts, **Accessibility**, Expression, Query), residual-stream information transmission, selective fact retrospection (accessible vs. inaccessible worlds), and specialized attention head taxonomy (including **Accessibility-Filtering Heads**). Evaluated on `Qwen3-8B` and `Qwen3-14B`.

---

## 2. Current Project Status & Verification Matrix

| Component | Status | Details & Metrics |
|:---|:---:|:---|
| **Data Generation (Part B)** | **COMPLETE** | 1,000 samples generated (`facts_first`), 1,000 samples (`expr_first`), 7 rule categories, 50% 1-hop / 50% 2-hop, 502 fact corruptions vs 498 accessibility corruptions. |
| **Data Generation (Part A)** | **COMPLETE** | 500 controlled counterfactual pairs generated across 5 flip modes (100 query, 100 operator $\\Box\\leftrightarrow\\Diamond$, 100 accessibility, 100 fact, 100 rule swap). |
| **Patching Engine & Metrics** | **COMPLETE** | Residual stream, MLP zero/mean ablation, attention head output ($z$), and sub-component ($q, k, v$) patching with GQA group handling. |
| **Part A Circuit Discovery** | **COMPLETE** | CMA necessity sweep, head classification (QRLH, MOH, WAH, FPH, QRMH, DH), and complement patching sufficiency table generator. |
| **Part B Staged Computation** | **COMPLETE** | 4-region MLP staging, token-wise transmission, accessible vs inaccessible fact retrospection contrast ($accessible \\gg inaccessible$), and Accessibility-Filtering Heads. |
| **Visualizations & Plots** | **COMPLETE** | Publication-grade circuit diagrams, 2D layer-by-head/token heatmaps, and stage bar charts with SEM error bars. |
| **Unit & Integration Tests** | **PASSED** | 17/17 tests passing across grammar, pairing functions, metrics, corruptions, staging, and head taxonomies. |

---

## 3. Generated Datasets Summary

The production datasets are generated under `data/`:

| File Path | Sample Count | Description |
|:---|---:|:---|
| `data/modal_mi/modal_mi_facts_first.jsonl` | 1,000 | 4-Region structure (`facts_first`), balanced across all 7 modal categories (~143 per category), 50% 1-hop / 50% 2-hop. |
| `data/modal_mi/modal_mi_expr_first.jsonl` | 1,000 | 4-Region structure (`expr_first`), balanced across all 7 modal categories, 50% 1-hop / 50% 2-hop. |
| `data/modal_circuit/modal_circuit_pairs.jsonl` | 500 | 100 `query_flip`, 100 `modal_operator_flip` ($\\Box\\leftrightarrow\\Diamond$), 100 `accessibility_flip`, 100 `fact_flip`, 100 `rule_location_swap`. |

---

## 4. Repository Layout

```text
modal-logic-mi/
|-- configs/                   # Experiment configurations for Part A and Part B
|   |-- part_a_mistral7b.yaml
|   |-- part_a_gemma9b.yaml
|   |-- part_a_gemma27b.yaml
|   |-- part_b_qwen8b.yaml
|   |-- part_b_qwen14b.yaml
|   \-- calibration_proplogic.yaml
|-- data/
|   |-- modal_circuit/         # Part A controlled counterfactual prompt pairs
|   \-- modal_mi/              # Part B ModalLogic-MI datasets (1-hop & 2-hop)
|-- results/
|   |-- part_a/{model}/        # Discovered circuits, sufficiency tables, findings
|   \-- part_b/{model}/        # MLP staging, retrospection contrast, findings
|-- scripts/                   # End-to-end execution scripts
|   |-- dataset_create.sh
|   |-- run_part_a_mistral7b.sh
|   |-- run_part_a_gemma9b.sh
|   |-- run_part_b_qwen8b.sh
|   \-- run_smoke_tests.sh
|-- src/
|   |-- data_gen/              # Kripke semantics, AST, pair generators, formatters
|   |   |-- modal_grammar.py
|   |   |-- circuit_pairs.py
|   |   |-- corruptions.py
|   |   |-- formatters.py
|   |   |-- mi_pairs.py
|   |   \-- generate_dataset.py
|   |-- patching/              # Core activation patching and metrics engine
|   |   |-- activation_patch.py
|   |   |-- metrics.py
|   |   \-- cma.py
|   |-- circuits/              # Part A circuit discovery and classification
|   |   |-- head_discovery.py
|   |   |-- head_classify.py
|   |   |-- sufficiency_table.py
|   |   \-- run.py
|   |-- staged/                # Part B macroscopic mechanistic analyses
|   |   |-- mlp_staging.py
|   |   |-- info_transmission.py
|   |   |-- fact_retrospection.py
|   |   |-- specialized_heads.py
|   |   \-- run.py
|   |-- viz/                   # Publication-ready figure visualizers
|   |   |-- bar_charts.py
|   |   |-- circuit_diagram.py
|   |   \-- heatmaps.py
|   |-- model_loading.py       # HookedTransformer universal loader with GQA support
|   |-- plot_style.py          # Consistent publication plot styling
|   \-- progress.py            # Structured logging and progress tracking
|-- tests/                     # Unit and integration test suite
|-- requirements.txt
\-- README.md
```

---

## 5. Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Generate Datasets
```bash
bash scripts/dataset_create.sh
```

### 3. Run Part A: Modal Circuit Discovery
```bash
python -m src.circuits.run --config configs/part_a_mistral7b.yaml
```

### 4. Run Part B: Mechanistic Principles Analysis
```bash
python -m src.staged.run --config configs/part_b_qwen8b.yaml
```

### 5. Run Unit Tests
```bash
python tests/run_all.py
```

---

## 6. Core Methodologies & Novel Extensions

### Part A: Circuit Discovery on Modal Logic
- **Controlled Counterfactual Pairs**: 5 pairing regimes holding all context byte-identical except target features: `query_flip`, `modal_operator_flip` ($\\Box \\leftrightarrow \\Diamond$), `accessibility_flip`, `fact_flip`, `rule_location_swap`.
- **New Head Families**:
  - **Modal-Operator Heads (MOH)**: High Indirect Effect (IE) under $\\Box \\leftrightarrow \\Diamond$ flips.
  - **World-Accessibility Heads (WAH)**: High IE under accessibility modifications, combined with a negative-control specificity assertion (near-zero attention mass on inaccessible-world facts).
- **Sufficiency Ablation**: Complement patching verifying that Circuit $C$ retains >85% calibrated logit diff, while dropping MOH or WAH degrades performance significantly.

### Part B: Mechanistic Principles on Modal Logic
- **4-Region Staged Computation**: Partitioning prompts into Facts, **Accessibility**, Expression, and Query regions. Confirms that Accessibility reasoning peaks in the intermediate layer band between facts and modal rule synthesis.
- **Selective Fact Retrospection**: Fact representations in late layers selectively retain accessible-world information ($accessible \\gg inaccessible$, contrast ratio > 3.5x).
- **Accessibility-Filtering Attention Heads**: Attention heads whose fact retrieval is conditionally gated by Kripke frame accessibility.

---

## 7. License & Attribution
Extends Hong et al. (2025) *A Implies B: Circuit Analysis in LLMs for Propositional Logical Reasoning* (NeurIPS 2025) and Chen et al. (2026) *Towards a Mechanistic Understanding of Propositional Logical Reasoning in Large Language Models*.
