import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def balanced_sample(df_subset: pd.DataFrame, n_total: int, seed: int = 42) -> pd.DataFrame:
    """Sample exactly n_total rows, balanced across rule categories."""
    if len(df_subset) <= n_total:
        return df_subset
    
    # Shuffle first to ensure randomness within groups
    df_subset = df_subset.sample(frac=1, random_state=seed)
    
    # Calculate how many to sample per group
    n_groups = df_subset['rule_category'].nunique()
    n_per_group = max(1, n_total // n_groups)
    
    # Sample equally from each category
    sampled = df_subset.groupby('rule_category', group_keys=False).apply(
        lambda x: x.head(n_per_group)
    )
    
    # If division leaves a shortfall, randomly sample remainder from the unselected pool
    if len(sampled) < n_total:
        remaining = df_subset[~df_subset.index.isin(sampled.index)]
        shortfall = n_total - len(sampled)
        sampled = pd.concat([sampled, remaining.sample(n=shortfall, random_state=seed)])
        
    # Return exactly n_total (in case n_per_group * n_groups exceeded n_total)
    return sampled.head(n_total)

def main():
    parser = argparse.ArgumentParser(description="Sample 370 examples matching Chen et al. using Pandas")
    parser.add_argument('--input', type=Path, required=True, help="Input dataset JSONL file")
    parser.add_argument('--output', type=Path, required=True, help="Output sampled JSONL file")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    input_df = pd.read_json(args.input, lines=True)

    # Create a unified rule category column for balancing
    if 'axiom' in input_df.columns and 'rule' in input_df.columns:
        input_df['rule_category'] = input_df['rule'].fillna(input_df['axiom'])
    elif 'axiom' in input_df.columns:
        input_df['rule_category'] = input_df['axiom']
    elif 'rule' in input_df.columns:
        input_df['rule_category'] = input_df['rule']
    else:
        raise ValueError("Dataset must contain either 'rule' or 'axiom' column for balancing.")

    # Filter by hop (handles both string representations and integers)
    one_hop_df = input_df[input_df['hop'].astype(str).isin(['1', 'one_hop'])]
    two_hop_df = input_df[input_df['hop'].astype(str).isin(['2', 'two_hop'])]

    if len(one_hop_df) < 74 or len(two_hop_df) < 296:
        print(f"Warning: Not enough data! Have {len(one_hop_df)} 1-hop and {len(two_hop_df)} 2-hop.")

    # Apply balanced sampling
    sampled_one_hop = balanced_sample(one_hop_df, 74, seed=args.seed)
    sampled_two_hop = balanced_sample(two_hop_df, 296, seed=args.seed)

    # Combine, shuffle, and clean up
    proplogic_df = pd.concat([sampled_one_hop, sampled_two_hop])
    proplogic_df = proplogic_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    proplogic_df = proplogic_df.drop(columns=['rule_category'])

    # Validate output
    final_size = len(proplogic_df)
    one_hop_count = len(proplogic_df[proplogic_df['hop'].astype(str).isin(['1', 'one_hop'])])
    two_hop_count = len(proplogic_df[proplogic_df['hop'].astype(str).isin(['2', 'two_hop'])])
    
    print(f"Final proplogic_df size: {final_size}")
    print(f"1-hop count: {one_hop_count} (Target: 74)")
    print(f"2-hop count: {two_hop_count} (Target: 296)")

    # Export to jsonl
    print(f"Saving to {args.output}...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    proplogic_df.to_json(args.output, orient="records", lines=True)
    print("Done!")

if __name__ == '__main__':
    main()
