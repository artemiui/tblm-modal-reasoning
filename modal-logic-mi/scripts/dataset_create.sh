#!/bin/bash
set -e

echo "=== Generating ModalLogic-MI and Modal Circuit Datasets ==="
python -m src.data_gen.generate_dataset --output data/modal_mi/modal_mi_facts_first.jsonl --dataset_type modal_mi --n_samples 1000 --prompt_order facts_first
python -m src.data_gen.generate_dataset --output data/modal_mi/modal_mi_expr_first.jsonl --dataset_type modal_mi --n_samples 1000 --prompt_order expr_first
python -m src.data_gen.generate_dataset --output data/modal_circuit/modal_circuit_pairs.jsonl --dataset_type modal_circuit --n_samples 500
echo "Datasets generated successfully."
