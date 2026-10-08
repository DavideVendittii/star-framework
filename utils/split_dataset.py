"""
Dataset Splitting Tool for STAR Framework (Section 3).

Partitions the corpus into:
- D_train (70%, N=4,351) -> further split into:
  - D_sentinel (80% of D_train, N=3,481) for Sentinel & Abstractor training
  - D_calib (20% of D_train, N=870) for Meta-Policy threshold calibration
- D_test (30%, N=1,865) for final evaluation

Usage:
    python utils/split_dataset.py --input_csv data/star_full_dataset.csv --out_dir data/
"""

import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="Path to raw full dataset CSV")
    parser.add_argument("--out_dir", default="data/", help="Directory to save dataset splits")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    df.columns = df.columns.str.strip().str.lower()

    # 70% Train, 30% Test
    df_train, df_test = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["label"]
    )

    # 80% D_sentinel, 20% D_calib from D_train
    df_sentinel, df_calib = train_test_split(
        df_train, test_size=0.20, random_state=42, stratify=df_train["label"]
    )

    p_sentinel = os.path.join(args.out_dir, "star_dataset_sentinel.csv")
    p_calib = os.path.join(args.out_dir, "star_dataset_calib.csv")
    p_test = os.path.join(args.out_dir, "star_dataset_test.csv")

    df_sentinel.to_csv(p_sentinel, index=False)
    df_calib.to_csv(p_calib, index=False)
    df_test.to_csv(p_test, index=False)

    print(f"Splits generated in {args.out_dir}:")
    print(f"  - D_sentinel: {len(df_sentinel)} samples -> {p_sentinel}")
    print(f"  - D_calib:    {len(df_calib)} samples -> {p_calib}")
    print(f"  - D_test:     {len(df_test)} samples -> {p_test}")


if __name__ == "__main__":
    main()