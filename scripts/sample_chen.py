import argparse
import random
from pathlib import Path

from src.eval.io_utils import read_jsonl, write_jsonl, balanced_sample_by_rule
from src.data.split_by_hop import split_rows_by_hop

def main():
    parser = argparse.ArgumentParser(description="Sample 370 examples (74 one-hop, 296 two-hop) matching Chen et al.")
    parser.add_argument('--input', type=Path, required=True, help="Input dataset JSONL file")
    parser.add_argument('--output', type=Path, required=True, help="Output sampled JSONL file")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    
    # Split by hop
    one_hop_rows, two_hop_rows = split_rows_by_hop(rows)
    
    # Check if we have enough data
    if len(one_hop_rows) < 74:
        raise ValueError(f"Not enough one-hop examples. Need 74, but only found {len(one_hop_rows)}.")
    if len(two_hop_rows) < 296:
        raise ValueError(f"Not enough two-hop examples. Need 296, but only found {len(two_hop_rows)}.")
        
    # Sample balanced by rule categories
    sampled_one_hop = balanced_sample_by_rule(one_hop_rows, max_samples=74, seed=args.seed)
    sampled_two_hop = balanced_sample_by_rule(two_hop_rows, max_samples=296, seed=args.seed)
    
    # Combine and shuffle
    combined = sampled_one_hop + sampled_two_hop
    rng = random.Random(args.seed)
    rng.shuffle(combined)
    
    write_jsonl(args.output, combined)
    
    print(f"Successfully sampled {len(combined)} total rows.")
    print(f"One-hop: {len(sampled_one_hop)} (Target: 74)")
    print(f"Two-hop: {len(sampled_two_hop)} (Target: 296)")
    print(f"Output saved to: {args.output}")

if __name__ == '__main__':
    main()
