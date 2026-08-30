#!/usr/bin/env bash
set -e

echo "=== Running Comparative Baseline: Modal Logic vs First-Order Propositional Logic ==="
echo "Models: Qwen3.5-2B, Qwen3.5-4B, Qwen3.5-9B"

# 1. Baseline Propositional Calibration on reference task
echo "[1/4] Running Propositional Baseline Calibration..."
python -m src.staged.mlp_staging --config configs/calibration_proplogic.yaml

# 2. Part A Circuit Comparative Sweeps
echo "[2/4] Running Part A Modal Proposition Circuit Discovery..."
python -m src.circuits.run --config configs/part_a_qwen3.5_2b.yaml
python -m src.circuits.run --config configs/part_a_qwen3.5_4b.yaml
python -m src.circuits.run --config configs/part_a_qwen3.5_9b.yaml

# 3. Part B Staged Macroscopic Comparative Sweeps
echo "[3/4] Running Part B Modal Macroscopic Staging Sweeps..."
python -m src.staged.run --config configs/part_b_qwen3.5_2b.yaml
python -m src.staged.run --config configs/part_b_qwen3.5_4b.yaml
python -m src.staged.run --config configs/part_b_qwen3.5_9b.yaml

echo "[4/4] Comparative evaluation complete across Qwen3.5-2B, Qwen3.5-4B, Qwen3.5-9B."
