from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.progress import log_event, make_tqdm, resolve_log_path, setup_file_logger
from .io_utils import balanced_sample_by_rule, read_jsonl, write_jsonl

_BOOL_RE = re.compile(r'\b(true|false)\b', re.IGNORECASE)

def _parse_bool(text: str) -> Optional[bool]:
    matches = _BOOL_RE.findall(text)
    if not matches:
        return None
    return matches[-1].lower() == 'true'

def _resolve_prompt(row: Dict[str, object], prompt_style: str, kind: str, mode: str) -> str:
    prompt_key = f"{kind}_prompt_{prompt_style}"
    if prompt_key in row:
        return str(row[prompt_key])
    
    # Fallback to symbolic if the specific style wasn't pre-generated
    fallback_key = f"{kind}_prompt_symbolic"
    if fallback_key in row:
        return str(row[fallback_key])
        
    raise ValueError(f"Could not find prompt key {prompt_key} or {fallback_key} in row")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prompt_style', choices=['symbolic', 'semi_natural', 'verbose'], default='symbolic')
    parser.add_argument('--mode', choices=['nocot', 'cot'], default='nocot')
    parser.add_argument('--max_new_tokens', type=int, default=8)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max_samples', type=int, default=0)
    parser.add_argument('--backend', choices=['api', 'local'], default='api')
    parser.add_argument('--hf_token', type=str, default=None)
    parser.add_argument('--progress_every', type=int, default=50)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if args.max_samples > 0:
        rows = balanced_sample_by_rule(rows, args.max_samples)

    if args.backend == 'api':
        from .inference_hf_api import create_inference_client
        client = create_inference_client(args.model_id, token=args.hf_token, max_new_tokens=args.max_new_tokens)
    else:
        import torch
        from transformers import pipeline
        print(f"Loading local model {args.model_id} on Colab GPU...")
        client = pipeline(
            "text-generation", 
            model=args.model_id, 
            model_kwargs={"torch_dtype": torch.float16}, 
            device_map="auto"
        )

    results = []
    for row in make_tqdm(rows, desc=f"Evaluating {args.model_id}"):
        prompt_clean = _resolve_prompt(row, args.prompt_style, 'clean', args.mode)
        prompt_corrupted = _resolve_prompt(row, args.prompt_style, 'corrupted', args.mode)
        
        if args.backend == 'api' and client is not None:
            pred_clean, raw_clean = client.predict_chat(prompt_clean, temperature=args.temperature)
            pred_corrupted, raw_corrupted = client.predict_chat(prompt_corrupted, temperature=args.temperature)
        else:
            # Local Inference
            def _local_predict(prompt: str) -> Tuple[Optional[bool], str]:
                messages = [{"role": "user", "content": prompt}]
                do_sample = args.temperature > 0
                out = client(
                    messages, 
                    max_new_tokens=args.max_new_tokens, 
                    temperature=args.temperature if do_sample else None, 
                    do_sample=do_sample
                )
                raw_text = out[0]['generated_text'][-1]['content']
                return _parse_bool(raw_text), raw_text

            pred_clean, raw_clean = _local_predict(prompt_clean)
            pred_corrupted, raw_corrupted = _local_predict(prompt_corrupted)
            
        row_out = dict(row)
        row_out.update({
            'pred_clean': pred_clean,
            'pred_corrupted': pred_corrupted,
            'raw_clean': raw_clean,
            'raw_corrupted': raw_corrupted,
            'correct_clean': pred_clean == row['label'] if pred_clean is not None and 'label' in row else False,
            'correct_corrupted': pred_corrupted == row['label_corrupted'] if pred_corrupted is not None and 'label_corrupted' in row else False,
            'eval_mode': args.mode,
            'prompt_style': args.prompt_style,
            'model_id': args.model_id,
            'backend': args.backend
        })
        results.append(row_out)

    write_jsonl(args.output, results)

if __name__ == '__main__':
    main()
