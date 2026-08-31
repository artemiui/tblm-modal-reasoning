# Generate all datasets for Modal Logic MI experiments
$ErrorActionPreference = 'Stop'

# Create output directories
New-Item -ItemType Directory -Force -Path dataset | Out-Null

Write-Host '=== Generating Propositional Baseline Datasets ===' -ForegroundColor Cyan

python -m src.data.generate_dataset `
    --output dataset/proplogic_baseline.jsonl `
    --mode propositional `
    --per_rule_per_hop 200 `
    --seed 42 `
    --prompt_order facts_first

python -m src.data.generate_dataset `
    --output dataset/proplogic_baseline_expr_first.jsonl `
    --mode propositional `
    --per_rule_per_hop 200 `
    --seed 42 `
    --prompt_order expr_first

Write-Host '=== Generating Modal Logic Datasets ===' -ForegroundColor Cyan

# Per-axiom datasets with different world counts
foreach ($nw in @(2, 3, 4)) {
    python -m src.data.generate_dataset `
        --output "dataset/modal_w${nw}.jsonl" `
        --mode modal `
        --per_axiom_per_hop 200 `
        --n_worlds $nw `
        --seed 42 `
        --prompt_style symbolic
}

# Combined dataset
python -m src.data.generate_dataset `
    --output dataset/combined_prop_modal.jsonl `
    --mode combined `
    --per_rule_per_hop 100 `
    --per_axiom_per_hop 100 `
    --n_worlds 3 `
    --seed 42

Write-Host '=== All datasets generated ===' -ForegroundColor Green
