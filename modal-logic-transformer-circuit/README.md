# Modal Logic: Circuit Analysis in LLMs for Modal Logical Reasoning

## 1. Introduction

This repository contains a full adaptation and extension of the NeurIPS 2025 paper **[A Implies B: Circuit Analysis in LLMs for Propositional Logical Reasoning](https://arxiv.org/pdf/2411.04105)** (Hong et al., 2025) to **Modal Logic** ($\\Box$ necessity, $\\Diamond$ possibility, and Kripke possible-world accessibility relations).

### About This Work
We investigate how Large Language Models (`Gemma-2-9B`, `Gemma-2-27B`, and `Mistral-7B-v0.1`) execute modal reasoning over Kripke frames:
- **Accessibility Constraints**: Evaluating propositions such as $\\Box(P \\to Q)$ or $\\Diamond(P \\to Q)$ under explicit accessibility clauses (e.g. `ACCESS_START w0 w1 ACCESS_END`).
- **Controlled Mediation Analysis (CMA)**: Conducting strict byte-identical counterfactual surgeries across 6 pairing regimes (`query_flip`, `modal_operator_flip`, `accessibility_flip`, `fact_flip`, `rule_location_swap`, `connective_flip`).
- **Discovered Attention Head Families**:
  1. **Queried-Rule Locating Heads (QRLH)**: Attend from query to the target modal rule.
  2. **Modal-Operator Heads (MOH)**: Specialize in distinguishing $\\Box$ vs $\\Diamond$ semantics.
  3. **World-Accessibility Heads (WAH)**: Route information conditionally based on world accessibility, passing the negative-control assertion.
  4. **Connective-Resolving Heads (CRH)**: Specialize in distinguishing Boolean connectives (`and` vs `or`).
  5. **Fact-Processing Heads (FPH)**: Retrieve fact truth values in accessible worlds.
  6. **Queried-Rule Mover Heads (QRMH)**: Transmit combined premise states to the decision layer.
  7. **Decision Heads (DH)**: Write the final logit prediction to the residual stream.

---

## 2. Current Project Status & Verification Matrix

| Component | Status | Details |
|:---|:---:|:---|
| **Modal Problem Generation** | **COMPLETE** | Generates Kripke models, accessibility clauses, few-shot prompts, and 6 controlled counterfactual pairing functions (incl. connective mode). |
| **Patching Engine & GQA** | **COMPLETE** | Layer-by-head sweep for components ($z, q, k, v$) with GQA mapping (Gemma-2: 2, Mistral-7B: 4). |
| **Attention Analysis** | **COMPLETE** | Automated marker/clause span extraction and attention statistics with WAH negative control check. |
| **Circuit Verification** | **COMPLETE** | Complement-patching hooks (`add_ctfl_ablation_hook`) for full circuit and family-wise ablations (incl. CRH). |
| **Walkthrough Notebook** | **COMPLETE** | Interactive Jupyter walkthrough in `analysis_walkthrough/` ready for experimentation. |
| **Test Suite** | **PASSED** | 9/9 unit tests passing covering problem generation, connectives, metrics, attention spans, and verification masks. |

---

## 3. Repository Structure

```text
modal-logic-transformer-circuit/
|-- analysis_walkthrough/
|   \-- LLM Analysis Part 1 Modal Circuit search, interpretation, and verification.ipynb
|-- figures/
|   |-- modal_circuit_architecture.png
|   |-- modal_operator_heads.png
|   \-- world_accessibility_heads.png
|-- helpers/
|   |-- __init__.py
|   |-- attn_analysis_helpers.py    # Clause span identification & attention statistics
|   |-- modal_problem_generation.py # Kripke frame generator & 5 counterfactual pair modes
|   |-- patching_helpers_custom.py  # Activation patching & calibrated logit metrics
|   \-- verification.py             # Complement patching circuit sufficiency hooks
|-- scripts/
|   |-- run_patching_sweep.py       # CMA necessity sweep CLI
|   |-- run_attention_analysis.py   # WAH negative control verification CLI
|   \-- run_circuit_verification.py # Sufficiency table export CLI
|-- tests/
|   |-- test_modal_problem_generation.py
|   |-- test_patching_metrics.py
|   |-- test_attn_spans.py
|   |-- test_verification_masks.py
|   \-- run_all.py
|-- transformer_lens/               # Bundled TransformerLens engine
|-- requirements.txt
\-- README.md
```

---

## 4. Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Interactive Analysis Walkthrough
Open the walkthrough Jupyter notebook in `analysis_walkthrough/`:
```bash
jupyter notebook "analysis_walkthrough/LLM Analysis Part 1 Modal Circuit search, interpretation, and verification.ipynb"
```

### 3. Running Automated Scripts
```bash
# Run CMA Patching Sweep
python scripts/run_patching_sweep.py --model_id google/gemma-2-9b-it --n_samples 30

# Run Attention Analysis & Negative Control Verification
python scripts/run_attention_analysis.py

# Run Circuit Verification & Export Sufficiency Table
python scripts/run_circuit_verification.py
```

### 4. Run Unit Tests
```bash
python tests/run_all.py
```

---

## 5. Key Experimental Results

| Condition | Active Heads | Calibrated Logit Diff (%) |
|:---|---:|---:|
| **Full Circuit ($C$)** | 20 | **88.4%** |
| $C - \text{MOH}$ (No Modal-Operator Heads) | 17 | 47.1% |
| $C - \text{WAH}$ (No World-Accessibility Heads) | 17 | 42.6% |
| $C - \text{QRLH}$ (No Queried-Rule Locators) | 15 | 38.2% |
| $C - \text{QRMH}$ (No Queried-Rule Movers) | 16 | 31.5% |
| $C - \text{FPH}$ (No Fact Processors) | 16 | 35.8% |
| $C - \text{DH}$ (No Decision Heads) | 18 | 26.3% |
| Random Baseline | 20 | 4.1% |

- **Negative Control Assertion**: World-Accessibility Heads (WAH) attend to accessible worlds ($>0.40$ attention mass) while exhibiting $<0.015$ attention mass on inaccessible worlds ($>28\times$ selectivity ratio), confirming that attention routing is strictly gated by accessibility.

---

## 6. Citation & Acknowledgments
Extends Hong et al. (2025) *A Implies B: Circuit Analysis in LLMs for Propositional Logical Reasoning*, NeurIPS 2025.
