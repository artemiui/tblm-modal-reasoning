# Run propositional baseline experiments across all Qwen3 models
$ErrorActionPreference = 'Stop'

$models = @('Qwen/Qwen3-0.6B', 'Qwen/Qwen3-1.7B', 'Qwen/Qwen3-4B', 'Qwen/Qwen3-8B')
$modelTags = @('Qwen3-0.6B', 'Qwen3-1.7B', 'Qwen3-4B', 'Qwen3-8B')

New-Item -ItemType Directory -Force -Path artifacts | Out-Null
New-Item -ItemType Directory -Force -Path reports | Out-Null

for ($i = 0; $i -lt $models.Count; $i++) {
    $model = $models[$i]
    $tag = $modelTags[$i]
    
    foreach ($order in @('facts_first', 'expr_first')) {
        if ($order -eq 'facts_first') {
            $ds = 'dataset/proplogic_baseline.jsonl'
        } else {
            $ds = 'dataset/proplogic_baseline_expr_first.jsonl'
        }
        
        Write-Host "=== Propositional baseline: $tag ($order) ===" -ForegroundColor Cyan
        
        python -m src.eval.inference `
            --model_id $model `
            --input $ds `
            --output "artifacts/preds_prop_${tag}_${order}.jsonl" `
            --prompt_style symbolic `
            --mode nocot `
            --max_new_tokens 8 `
            --temperature 0.0 `
            --backend api `
            --max_samples 0
        
        python -m src.eval.filtering `
            --input "artifacts/preds_prop_${tag}_${order}.jsonl" `
            --output "artifacts/filtered_prop_${tag}_${order}.jsonl"
        
        python -m src.eval.metrics `
            --input "artifacts/preds_prop_${tag}_${order}.jsonl" `
            --output "reports/metrics_prop_${tag}_${order}.json"
    }
}

Write-Host '=== Propositional baselines completed ===' -ForegroundColor Green
