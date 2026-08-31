# Run modal logic experiments across all Qwen3 models
$ErrorActionPreference = 'Stop'

$models = @('Qwen/Qwen3-0.6B', 'Qwen/Qwen3-1.7B', 'Qwen/Qwen3-4B', 'Qwen/Qwen3-8B')
$modelTags = @('Qwen3-0.6B', 'Qwen3-1.7B', 'Qwen3-4B', 'Qwen3-8B')
$datasets = @('dataset/modal_w3.jsonl')

New-Item -ItemType Directory -Force -Path artifacts | Out-Null
New-Item -ItemType Directory -Force -Path reports | Out-Null

for ($i = 0; $i -lt $models.Count; $i++) {
    $model = $models[$i]
    $tag = $modelTags[$i]
    
    foreach ($ds in $datasets) {
        $dsName = [System.IO.Path]::GetFileNameWithoutExtension($ds)
        
        Write-Host "=== Running inference: $tag on $dsName ===" -ForegroundColor Cyan
        
        python -m src.eval.inference `
            --model_id $model `
            --input $ds `
            --output "artifacts/preds_${tag}_${dsName}.jsonl" `
            --prompt_style symbolic `
            --mode nocot `
            --max_new_tokens 8 `
            --temperature 0.0 `
            --backend api `
            --max_samples 0
        
        Write-Host "=== Filtering: $tag on $dsName ===" -ForegroundColor Yellow
        
        python -m src.eval.filtering `
            --input "artifacts/preds_${tag}_${dsName}.jsonl" `
            --output "artifacts/filtered_${tag}_${dsName}.jsonl"
        
        Write-Host "=== Metrics: $tag on $dsName ===" -ForegroundColor Green
        
        python -m src.eval.metrics `
            --input "artifacts/preds_${tag}_${dsName}.jsonl" `
            --output "reports/metrics_${tag}_${dsName}.json"
    }
}

Write-Host '=== All modal experiments completed ===' -ForegroundColor Green
