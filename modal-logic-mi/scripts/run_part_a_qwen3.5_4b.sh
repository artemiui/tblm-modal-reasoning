#!/usr/bin/env bash
set -e

echo "Running Part A Circuit Discovery on Qwen3.5-4B..."
python -m src.circuits.run --config configs/part_a_qwen3.5_4b.yaml
