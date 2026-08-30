#!/usr/bin/env bash
set -e

echo "Running Part B Mechanistic Analysis on Qwen3.5-4B..."
python -m src.staged.run --config configs/part_b_qwen3.5_4b.yaml
