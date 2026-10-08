"""
Results Aggregator for STAR Framework (Tables 2 & 3).

Usage:
    python utils/aggregate_star_results.py --results_dir benchmark_results/ --out_dir paper_artifacts/
"""

import argparse
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(df, pred_col="star_pred_class", refused_col="star_refused"):
    acc = accuracy_score(df["true_label"], df[pred_col])
    risky = df[df["true_label"] == 1]
    benign = df[df["true_label"] == 0]

    lvr = (1.0 - risky[refused_col].mean()) * 100.0
    frr = benign[refused_col].mean() * 100.0
    return acc * 100.0, lvr, frr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="./")
    parser.add_argument("--out_dir", default="paper_artifacts/")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    files = [f for f in os.listdir(args.results_dir) if f.endswith("benchmark_results_raw_vs_star.csv")]
    if not files:
        print(f"No benchmark_results_raw_vs_star.csv files found in {args.results_dir}")
        return

    table3_rows = []
    table2_rows = []

    for fname in sorted(files):
        model_name = fname.replace("_benchmark_results_raw_vs_star.csv", "").replace("benchmark_results_raw_vs_star.csv", "Model")
        df = pd.read_csv(os.path.join(args.results_dir, fname))

        b_acc, b_lvr, b_frr = compute_metrics(df, "raw_pred_class", "raw_refused")
        table3_rows.append({"Model": model_name, "Config": "Baseline", "Acc (%)": round(b_acc, 2), "LVR (%)": round(b_lvr, 2), "FRR (%)": round(b_frr, 2)})

        s_acc, s_lvr, s_frr = compute_metrics(df, "star_pred_class", "star_refused")
        table3_rows.append({"Model": model_name, "Config": "STAR", "Acc (%)": round(s_acc, 2), "LVR (%)": round(s_lvr, 2), "FRR (%)": round(s_frr, 2)})

        raw_p, raw_r, _, _ = precision_recall_fscore_support(df["true_label"], df["raw_pred_class"], average=None)
        star_p, star_r, _, _ = precision_recall_fscore_support(df["true_label"], df["star_pred_class"], average=None)

        for c in [0, 1]:
            table2_rows.append({
                "Model": model_name, "Class": c,
                "Baseline_P": round(raw_p[c], 2), "Baseline_R": round(raw_r[c], 2),
                "STAR_P": round(star_p[c], 2), "STAR_R": round(star_r[c], 2)
            })

    t3_df = pd.DataFrame(table3_rows)
    t2_df = pd.DataFrame(table2_rows)

    t3_df.to_csv(os.path.join(args.out_dir, "table3_main_results.csv"), index=False)
    t2_df.to_csv(os.path.join(args.out_dir, "table2_per_class.csv"), index=False)

    print("=== TABLE 3: MAIN RESULTS ===")
    print(t3_df.to_string(index=False))

    print("\n=== TABLE 2: PER-CLASS PRECISION/RECALL ===")
    print(t2_df.to_string(index=False))


if __name__ == "__main__":
    main()