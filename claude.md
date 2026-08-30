# CLAUDE.md -- Modal Logic Mechanistic Interpretability Project

## 1. Project Overview

This project extends two foundational mechanistic-interpretability studies of propositional logic reasoning in Large Language Models to **MODAL LOGIC** (incorporating $\Box$ necessity, $\Diamond$ possibility, and Kripke possible-world accessibility relations $R \subseteq W \times W$):

1. **Part A (Circuit Discovery -- Hong et al. 2025 Extension)**:
   - Reference Paper: *A Implies B: Circuit Analysis in LLMs for Propositional Logical Reasoning* (NeurIPS 2025).
   - Reference Repository: `prop-logic-transformer-circuit-main/`.
   - Core Mechanism: Controlled Mediation Analysis (CMA) circuit discovery, identifying functional attention head families (`QRLH`, `QRMH`, `FPH`, `DH`) plus novel modal and connective families: **Modal-Operator Heads (MOH)**, **World-Accessibility Heads (WAH)**, and **Connective-Resolving Heads (CRH)**.
   - Evaluated Models: `Mistral-7B-Instruct-v0.2`, `Gemma-2-9B-it`, `Gemma-2-27B-it`.

2. **Part B (Mechanistic Principles -- Chen et al. 2026 Extension)**:
   - Reference Paper: *Towards a Mechanistic Understanding of Propositional Logical Reasoning in Large Language Models* (2026).
   - Reference Repository: `anomy_repo_CDSW74VD/`.
   - Core Mechanism: Macroscopic pattern analysis including 4-region staged computation (Facts, **Accessibility**, Expression, Query), residual-stream information transmission with accessibility boundary and operator tokens, selective fact retrospection (accessible vs. inaccessible contrast), and specialized attention head taxonomy (including **Accessibility-Filtering Heads** across 8 rule categories).
   - Evaluated Models: `Qwen/Qwen3-8B`, `Qwen/Qwen3-14B`.

---

## 2. Workspace Repository Layouts

### A. Unified Modal Interpretability Repository (`modal-logic-mi/`)
Modular end-to-end framework implementing both Part A and Part B pipelines:

```text
modal-logic-mi/
|-- configs/
|   |-- part_a_mistral7b.yaml      # Part A config for Mistral-7B
|   |-- part_a_gemma9b.yaml        # Part A config for Gemma-2-9B
|   |-- part_a_gemma27b.yaml       # Part A config for Gemma-2-27B
|   |-- part_b_qwen8b.yaml         # Part B config for Qwen3-8B
|   |-- part_b_qwen14b.yaml        # Part B config for Qwen3-14B
|   \-- calibration_proplogic.yaml # Propositional baseline calibration
|-- data/
|   |-- modal_circuit/             # Part A 6 controlled counterfactual prompt pair sets
|   \-- modal_mi/                  # Part B 1-hop & 2-hop ModalLogic-MI 8-category datasets
|-- results/
|   |-- part_a/{model}/            # Discovered circuits, sufficiency tables, findings.md
|   \-- part_b/{model}/            # MLP staging, retrospection contrast, findings.md
|-- scripts/
|   |-- dataset_create.sh          # Dataset generation CLI script
|   |-- run_part_a_mistral7b.sh    # Part A runner for Mistral-7B
|   |-- run_part_a_gemma9b.sh      # Part A runner for Gemma-2-9B
|   |-- run_part_b_qwen8b.sh       # Part B runner for Qwen3-8B
|   \-- run_smoke_tests.sh         # Test suite runner
|-- src/
|   |-- data_gen/
|   |   |-- modal_grammar.py       # Kripke semantics, AST, evaluators (Box, Diamond, And, Or, Xor, Iff)
|   |   |-- circuit_pairs.py       # 6 Part A controlled counterfactual pairing functions
|   |   |-- corruptions.py         # Minimal fact & accessibility edge label-flipping corruptions
|   |   |-- formatters.py          # 4-region prompt formatting (facts, access, expr, query)
|   |   |-- mi_pairs.py            # 8 Modal rule categories with dual-corruption modes
|   |   \-- generate_dataset.py    # Dataset generation CLI entrypoint
|   |-- patching/
|   |   |-- activation_patch.py    # Hook engine (residual, MLP, attn head, Q/K/V, complement patching)
|   |   |-- metrics.py             # Logit diff, Calibrated LD (cLD / IE), dPD, dLD, R_LD
|   |   \-- cma.py                 # Controlled Mediation Analysis necessity and sufficiency engines
|   |-- circuits/
|   |   |-- head_discovery.py      # Layer x Head CMA necessity sweep
|   |   |-- head_classify.py       # Classification into QRLH, QRMH, FPH, DH, MOH, WAH, CRH
|   |   |-- sufficiency_table.py   # Sufficiency ablation table exporter (CSV, MD, LaTeX)
|   |   \-- run.py                 # Part A CLI entrypoint
|   |-- staged/
|   |   |-- mlp_staging.py         # 4-region MLP ablation with accessibility timing hypothesis test
|   |   |-- info_transmission.py   # Token-wise residual patching across 9 refined categories
|   |   |-- fact_retrospection.py  # Accessible vs Inaccessible fact retrospection contrast
|   |   |-- specialized_heads.py   # Discovery, 4-role taxonomy, multi-head validation curves (k=1..64)
|   |   \-- run.py                 # Part B CLI entrypoint
|   |-- viz/
|   |   |-- bar_charts.py          # Grouped stage bar charts with SEM error bars
|   |   |-- circuit_diagram.py     # Publication-grade circuit architecture diagram generator
|   |   \-- heatmaps.py            # 2D layer-by-head and layer-by-token heatmaps
|   |-- model_loading.py           # HookedTransformer universal loader with GQA support
|   |-- plot_style.py              # Consistent publication matplotlib styling
|   \-- progress.py                # Structured logging and progress utilities
|-- tests/                         # Unit and integration test suite (20 tests)
|-- requirements.txt
\-- README.md
```

### B. Modal Transformer Circuit Interactive Clone (`modal-logic-transformer-circuit/`)
High-fidelity interactive clone of `prop-logic-transformer-circuit-main/` tailored for Part A:

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
|   |-- modal_problem_generation.py # Kripke frame generator & 6 counterfactual pair modes
|   |-- patching_helpers_custom.py  # Activation patching with GQA handling & calibrated logit metrics
|   |-- attn_analysis_helpers.py    # Region/marker finders, clause spans & negative control check
|   \-- verification.py             # Complement patching hooks & family ablation specifications (incl. CRH)
|-- scripts/
|   |-- run_patching_sweep.py       # CLI runner for CMA necessity sweep
|   |-- run_attention_analysis.py   # CLI runner for attention mass & negative control verification
|   \-- run_circuit_verification.py # CLI runner for sufficiency table export
|-- tests/
|   |-- test_modal_problem_generation.py
|   |-- test_patching_metrics.py
|   |-- test_attn_spans.py
|   |-- test_verification_masks.py
|   \-- run_all.py                  # All unit tests (9 tests)
|-- transformer_lens/               # Bundled TransformerLens package
|-- requirements.txt
\-- README.md
```

---

## 3. Part A: Circuit Discovery (Hong et al. 2025 Adaptation + Connective Extensions)

### A1. Modal Kripke Grammar & Token Formatting
- Evaluates propositions under a Kripke frame $\mathcal{M} = \langle W, R, V \rangle$ where $W = \{w_0, w_1, \dots\}$, $R \subseteq W \times W$.
- Supports crisp modalities ($\Box, \Diamond$) and multi-arity Boolean connectives (`and`, `or`, `xor`, `iff`, `not`).
- Prompt Format:
  - Accessibility Clause: `ACCESS_START w0 w1 ACCESS_END` (or `From w0, accessible worlds are [w0, w1].`).
  - World Facts: `In world w0: P is True, Q is False. In world w1: P is True, Q is True.`
  - Rules: `[Rule 1] necessarily P implies Q. [Rule 2] R implies S.`
  - Query: `Query: Is Q necessarily true from w0?`
  - Answer: `Answer: True.`
- Few-shot sets: 4-shot for Mistral-7B, 6-shot for Gemma-2 (Hong et al. §B.1).

### A2. 6 Controlled Counterfactual Pairing Regimes
Strict byte-identical counterfactual surgeries (`src/data_gen/circuit_pairs.py` & `helpers/modal_problem_generation.py`):
1. `query_flip`: Flips QUERY variable between modal conclusion and linear distractor conclusion.
2. `modal_operator_flip`: Flips $\Box \leftrightarrow \Diamond$ while holding facts and accessibility identical.
3. `accessibility_flip`: Modifies accessibility relation (e.g. $w_0 \to \{w_0, w_1\}$ vs $w_0 \to \{w_0\}$).
4. `fact_flip`: Flips fact truth value in a fixed world.
5. `rule_location_swap`: Swaps textual ordering of modal rule and linear rule.
6. `connective_flip`: Flips Boolean connectives `or` $\leftrightarrow$ `and` under modal context ($\Box(P \lor Q)$ vs $\Box(P \land Q)$).

### A3. CMA Patching Engine & GQA Support
- Implements attention-head output patching ($z$) and sub-component ($q, k, v$) patching (`src/patching/activation_patch.py`, `helpers/patching_helpers_custom.py`).
- Supports Grouped-Query Attention (GQA):
  - `Gemma-2-9B` / `Gemma-2-27B`: 2 query heads per KV head (`GQA_constant = 2`).
  - `Mistral-7B`: 4 query heads per KV head (`GQA_constant = 4`).
- Calibrated Logit Difference Metric (Hong et al. Eq. 1):
  $$\text{cLD} = \frac{\text{LD}_{\text{patched}} - \text{LD}_{\text{corrupted}}}{\text{LD}_{\text{clean}} - \text{LD}_{\text{corrupted}}}$$

### A4. Discovered Head Taxonomy & Negative-Control Specificity
- **Queried-Rule Locating Heads (QRLH)**: Attend from query to target modal rule.
- **Modal-Operator Heads (MOH)**: High indirect effect under $\Box \leftrightarrow \Diamond$ flips.
- **World-Accessibility Heads (WAH)**: High indirect effect under accessibility flips AND satisfies the **negative-control specificity assertion**:
  $$\text{AttnMass}(\text{WAH} \to \text{AccessibleFacts}) \gg \text{AttnMass}(\text{WAH} \to \text{InaccessibleFacts})$$
  (Assertion enforced in code: Inaccessible attention mass $< 0.05$).
- **Connective-Resolving Heads (CRH)**: High indirect effect under `and` $\leftrightarrow$ `or` connective flips.
- **Fact-Processing Heads (FPH)**: Retrieve fact truth values in accessible worlds.
- **Queried-Rule Mover Heads (QRMH)**: Transmit combined premise states to the decision position.
- **Decision Heads (DH)**: Write final output logits for True vs False.

### A5. Circuit Sufficiency Ablation
- Complement patching retaining only circuit heads while ablating the rest (`src/circuits/sufficiency_table.py`, `helpers/verification.py`).
- Conditions evaluated: Full Circuit ($C$), $C - \text{MOH}$, $C - \text{WAH}$, $C - \text{CRH}$, $C - \text{QRLH}$, $C - \text{QRMH}$, $C - \text{FPH}$, $C - \text{DH}$, Random Baseline.

---

## 4. Part B: Macroscopic Mechanistic Principles (Chen et al. 2026 Adaptation + Connective Extensions)

### B1. ModalLogic-MI 8 Rule Categories
Generates 1-hop and 2-hop modal logic samples across 8 fundamental categories (`src/data_gen/mi_pairs.py`):
1. `necessitation_implication`: $\Box(P \to Q)$ under accessibility.
2. `possibility_implication`: $\Diamond(P \to Q)$ and $\Diamond P \to Q$.
3. `duality`: $\Box P \leftrightarrow \neg \Diamond \neg P$ and $\Diamond P \leftrightarrow \neg \Box \neg P$.
4. `t_axiom`: $\Box P \to P$ on reflexive frames ($w_0 \in R(w_0)$).
5. `k_axiom`: $\Box(P \to Q) \to (\Box P \to \Box Q)$.
6. `cross_world_composition`: Multi-hop chained accessibility evaluation ($w_0 \to w_1 \to w_2$).
7. `modal_commutative_associative`: $\Box(P \land Q) \leftrightarrow \Box P \land \Box Q$, $\Diamond(P \lor Q) \leftrightarrow \Diamond P \lor \Diamond Q$.
8. `connective_disjunction`: $\Box(P \lor Q)$ evaluating disjunctive closure over accessible worlds.

### B2. 4-Region Staged Computation (`src/staged/mlp_staging.py`)
- Partitions prompts into 4 functional regions:
  1. `facts_region`: Facts per world.
  2. **`accessibility_region`**: Frame accessibility statements.
  3. `expression_region`: Modal rule / proposition.
  4. `query_region`: Question / suffix instruction.
- Zero/mean ablation per region per layer calculating $dPD$, $BMI$ (Band Mean Impact), $BCR$ (Band Concentration Ratio), and $R_{LD}$.
- **Accessibility Hypothesis Test**: Confirms that the Accessibility Region causally peaks in intermediate layer bands (between early Fact retrieval and late Expression evaluation).

### B3. Information Transmission (`src/staged/info_transmission.py`)
- Token-wise residual-stream patching tracking causal convergence across 9 refined categories:
  `facts_value`, `accessibility_boundary`, `variable_in_facts`, `variable_in_expr`, `operator`, `expr_last`, `derived_assignment`, `query_token`, `others`.
- Recognizes modal and Boolean operators (`box`, `diamond`, `necessarily`, `possibly`, `and`, `or`, `xor`, `iff`, `implies`, `not`).
- Grouped stage bar charts across Early, Middle, and Late layer groups with SEM error bars.

### B4. Selective Fact Retrospection (`src/staged/fact_retrospection.py`)
- Measures late-layer fact retrieval with an explicit modal contrast:
  $$\text{Mean}|dPD|(\text{accessible\_world\_facts}) \gg \text{Mean}|dPD|(\text{inaccessible\_world\_facts})$$
- Evaluates contrast ratio (predicted $> 3.0\times$), confirming LLMs selectively retrospected facts constrained by accessibility.

### B5. Specialized Attention Heads (`src/staged/specialized_heads.py`)
- Screened and classified into 4 functional families:
  - `fact_retrieval`
  - `splitting`
  - `transmission`
  - **`accessibility_filtering`** (heads whose fact attention is conditionally gated by world accessibility).
- Multi-head validation curves ($k=1, 2, 4, 8, 16, 32, 64$) comparing role ablations against random baselines.

---

## 5. Execution and Replication Commands

### 1. Dataset Generation
```bash
# In modal-logic-mi/
bash scripts/dataset_create.sh
```

### 2. Part A Circuit Discovery Execution
```bash
# In modal-logic-mi/
python -m src.circuits.run --config configs/part_a_mistral7b.yaml
python -m src.circuits.run --config configs/part_a_gemma9b.yaml
python -m src.circuits.run --config configs/part_a_gemma27b.yaml

# In modal-logic-transformer-circuit/
python scripts/run_patching_sweep.py --model_id google/gemma-2-9b-it
python scripts/run_attention_analysis.py
python scripts/run_circuit_verification.py
```

### 3. Part B Mechanistic Principles Execution
```bash
# In modal-logic-mi/
python -m src.staged.run --config configs/part_b_qwen8b.yaml
python -m src.staged.run --config configs/part_b_qwen14b.yaml
```

### 4. Running Test Suites
```bash
# Run all unit tests for modal-logic-mi (20 tests)
python modal-logic-mi/tests/run_all.py

# Run all unit tests for modal-logic-transformer-circuit (9 tests)
python modal-logic-transformer-circuit/tests/run_all.py
```

---

## 6. Definition of Done & Quality Checklist

- [x] **Part A Modal Circuit Discovery**: Full pipeline adapted from Hong et al., supporting MOH, WAH (with negative control specificity check), CRH, QRLH, QRMH, FPH, DH, GQA handling, and sufficiency tables.
- [x] **Part B Modal Staged Principles**: Full pipeline adapted from Chen et al., supporting 4-region MLP staging (with Accessibility Region), token-wise transmission, accessible vs. inaccessible fact retrospection contrast, and Accessibility-Filtering Heads across 8 rule categories.
- [x] **Systematic Boolean Connectives under Modality**: Boolean connectives under modal context (`and`, `or`, `xor`, `iff`, `not`) fully supported in AST, recursive evaluators, counterfactual pairing, and head classification.
- [x] **Standalone High-Fidelity Clone**: `modal-logic-transformer-circuit` created as a self-contained clone of `prop-logic-transformer-circuit-main` with interactive notebook walkthrough.
- [x] **Negative Controls**: Every modal-specific mechanism (MOH, WAH, CRH, Accessibility Region, Accessibility-Filtering Heads) includes explicit negative controls and selectivity ratio metrics.
- [x] **Unit & Integration Tests**: All 29 unit tests (20 in `modal-logic-mi`, 9 in `modal-logic-transformer-circuit`) passing cleanly.
- [x] **Documentation & Hygiene**: Complete READMEs, findings reports, requirements, and configuration files provided.
