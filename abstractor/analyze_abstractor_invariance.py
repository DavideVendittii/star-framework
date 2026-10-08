"""
Quantifies content-invariance of Abstractor projections on held-out risky activations.

Usage:
    python abstractor/analyze_abstractor_invariance.py --activations sentinel_activations_X.npy --labels sentinel_activations_y.npy --sentinel sentinel_model.pkl --abstractor_weights abstractor_weights.pt --prompts_csv data/star_dataset_train.csv
"""

import argparse
import os
import re
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


class PrivacyAbstractor(nn.Module):
    def __init__(self, hidden_size, device, dtype=torch.float32):
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False).to(device=device, dtype=dtype)
        nn.init.eye_(self.projection.weight)
        self.dtype = dtype

    def forward(self, hidden_states):
        return self.projection(hidden_states.to(self.dtype))


def load_abstractor(weights_path, hidden_size, target_layers, device="cpu"):
    state_dicts = torch.load(weights_path, map_location=device)
    abstractors = {}
    for layer in target_layers:
        m = PrivacyAbstractor(hidden_size, device, dtype=torch.float32)
        m.load_state_dict(state_dicts[layer])
        m.eval()
        abstractors[layer] = m
    return abstractors


def project_activations(X, abstractors, target_layers, hidden_size):
    chunks = np.split(X, len(target_layers), axis=-1)
    projected_chunks = []
    for i, layer in enumerate(target_layers):
        with torch.no_grad():
            t = torch.tensor(chunks[i], dtype=torch.float32)
            proj = abstractors[layer](t).numpy()
        projected_chunks.append(proj)
    return np.concatenate(projected_chunks, axis=-1)


def cosine_to_centroid(X):
    centroid = X.mean(axis=0, keepdims=True)
    num = (X * centroid).sum(axis=1)
    den = np.linalg.norm(X, axis=1) * np.linalg.norm(centroid)
    return num / (den + 1e-8)


SURFACE_KEYWORDS = {
    "email": r"\bemail\b",
    "address": r"\baddress\b",
    "password": r"\bpassword\b",
    "ssn": r"\bssn\b|\bsocial security\b",
    "phone": r"\bphone\b|\bnumber\b",
    "financial": r"\bcredit card\b|\bbank\b|\biban\b",
}


def build_surface_pseudolabels(prompts):
    labels = []
    for p in prompts:
        p_low = str(p).lower()
        assigned = "other"
        for cat, pattern in SURFACE_KEYWORDS.items():
            if re.search(pattern, p_low):
                assigned = cat
                break
        labels.append(assigned)
    return np.array(labels)


def probe_accuracy(X, y, cv=5):
    if len(set(y)) < 2:
        return float("nan")
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    return scores.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--sentinel", required=True)
    parser.add_argument("--abstractor_weights", required=True)
    parser.add_argument("--prompts_csv", required=True)
    parser.add_argument("--target_layers", nargs="+", type=int, default=[12, 13, 14, 15, 16])
    parser.add_argument("--hidden_size", type=int, default=4096)
    parser.add_argument("--out_dir", default="invariance_analysis")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    X = np.load(args.activations)
    y = np.load(args.labels)
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)

    df = pd.read_csv(args.prompts_csv)
    df.columns = df.columns.str.strip().str.lower()
    prompts = df["prompt"].values

    risky_mask = y == 1
    X_risky = X[risky_mask]
    prompts_risky = prompts[risky_mask]

    sentinel = joblib.load(args.sentinel)
    abstractors = load_abstractor(args.abstractor_weights, args.hidden_size, args.target_layers)

    X_risky_proj = project_activations(X_risky, abstractors, args.target_layers, args.hidden_size)

    cos_pre = cosine_to_centroid(X_risky)
    cos_post = cosine_to_centroid(X_risky_proj)

    surface_labels = build_surface_pseudolabels(prompts_risky)
    acc_pre = probe_accuracy(X_risky, surface_labels)
    acc_post = probe_accuracy(X_risky_proj, surface_labels)

    sentinel_acc_pre = sentinel.score(X_risky, np.ones(len(X_risky)))
    sentinel_acc_post = sentinel.score(X_risky_proj, np.ones(len(X_risky_proj)))

    summary_df = pd.DataFrame([
        {"Metric": "Cosine similarity to centroid (mean)", "Pre-Abstractor": round(cos_pre.mean(), 4), "Post-Abstractor": round(cos_post.mean(), 4)},
        {"Metric": "Cosine similarity to centroid (std)", "Pre-Abstractor": round(cos_pre.std(), 4), "Post-Abstractor": round(cos_post.std(), 4)},
        {"Metric": "Surface-content probe accuracy", "Pre-Abstractor": round(acc_pre, 4), "Post-Abstractor": round(acc_post, 4)},
        {"Metric": "Sentinel recall on risky class", "Pre-Abstractor": round(sentinel_acc_pre, 4), "Post-Abstractor": round(sentinel_acc_post, 4)},
    ])

    summary_df.to_csv(os.path.join(args.out_dir, "invariance_summary.csv"), index=False)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()