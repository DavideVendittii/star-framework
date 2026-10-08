"""
Ablation Experiment Runner (STAR Paper Table 3).

Executes all 4 ablation configurations:
1. Sentinel-only
2. Sentinel + Abstractor
3. Sentinel + Meta-Policy
4. STAR Full (Sentinel + Abstractor + Meta-Policy)

Usage:
    python ablation/run_ablation.py --model_id meta-llama/Llama-3.1-8B-Instruct --sentinel_path sentinel_model.pkl --abstractor_weights abstractor_weights.pt --thresholds_path meta_policy_thresholds.json --test_csv data/star_dataset_test.csv
"""

import argparse
import os
import pandas as pd
from tqdm import tqdm
import torch

from ablation.ablation_guard import AblationGuard, ABLATION_CONFIGS
from benchmark.evaluate_guard_vs_baseline import is_refusal, DEVICE, LAYERS_TO_MONITOR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--sentinel_path", required=True)
    parser.add_argument("--abstractor_weights", required=True)
    parser.add_argument("--thresholds_path", required=True)
    parser.add_argument("--test_csv", default="data/star_dataset_test.csv")
    parser.add_argument("--out_dir", default="ablation_results/")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.test_csv)
    df.columns = df.columns.str.strip().str.lower()

    guard = AblationGuard(
        model_id=args.model_id,
        sentinel_path=args.sentinel_path,
        abstractor_weights_path=args.abstractor_weights,
        thresholds_path=args.thresholds_path,
        target_layers=LAYERS_TO_MONITOR,
        config=ABLATION_CONFIGS["star_full"],
        device=DEVICE,
    )

    print("\n--- Running Baseline Evaluation ---")
    base_results = []
    for row in tqdm(df.itertuples(), total=len(df), desc="Baseline"):
        text = guard.generate_baseline(row.prompt)
        refused = is_refusal(text)
        base_results.append({
            "prompt": row.prompt, "true_label": int(row.label),
            "generated_text": text, "refused": int(refused)
        })
    pd.DataFrame(base_results).to_csv(os.path.join(args.out_dir, "ablation_baseline.csv"), index=False)

    for config_key, cfg in ABLATION_CONFIGS.items():
        print(f"\n--- Running Ablation Cell: {cfg.name} ---")
        guard.set_config(cfg)

        cell_results = []
        for row in tqdm(df.itertuples(), total=len(df), desc=f"Ablation ({cfg.name})"):
            text, action, score = guard.generate_guarded(row.prompt)
            refused = is_refusal(text)
            cell_results.append({
                "prompt": row.prompt, "true_label": int(row.label),
                "generated_text": text, "action": action,
                "risk_score": score, "refused": int(refused)
            })

        pd.DataFrame(cell_results).to_csv(
            os.path.join(args.out_dir, f"ablation_{cfg.name}.csv"), index=False
        )

    print(f"\nAll ablation runs completed. Saved in {args.out_dir}")


if __name__ == "__main__":
    main()