"""
Calibrates tau_low and tau_high decision thresholds on the calibration split.

Usage:
    python meta_policy/calibrate_meta_policy.py --model_id meta-llama/Llama-3.1-8B-Instruct --sentinel_path sentinel_model.pkl --csv data/star_dataset_calib.csv
"""

import argparse
import json
import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

LOW_PERCENTILE = 98
HIGH_PERCENTILE = 2
TARGET_LAYERS = list(range(12, 17))


def extract_features(model, tokenizer, prompt: str, device: str = "cuda:0") -> np.ndarray:
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False).to(device)

    activations = {}
    def make_hook(layer_idx):
        def hook(module, input_tensor, output_tensor):
            hidden = output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor
            activations[layer_idx] = hidden[0, -1, :].detach().to(torch.float32).cpu().numpy()
        return hook

    handles = [model.model.layers[idx].register_forward_hook(make_hook(idx)) for idx in TARGET_LAYERS]

    with torch.no_grad():
        _ = model(**inputs)

    for h in handles:
        h.remove()

    return np.concatenate([activations[idx] for idx in TARGET_LAYERS])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--sentinel_path", required=True)
    parser.add_argument("--csv", default="data/star_dataset_calib.csv")
    parser.add_argument("--output_json", default="meta_policy_thresholds.json")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(args.csv)
    df.columns = df.columns.str.strip().str.lower()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, device_map={"": device})
    model.eval()

    sentinel = joblib.load(args.sentinel_path)

    scores = {0: [], 1: []}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Calculating risk scores"):
        feats = extract_features(model, tokenizer, row["prompt"], device).reshape(1, -1)
        score = sentinel.predict_proba(feats)[0][1]
        scores[int(row["label"])].append(score)

    benign_scores = np.array(scores[0])
    risky_scores = np.array(scores[1])

    tau_low = float(np.percentile(benign_scores, LOW_PERCENTILE))
    tau_high = float(np.percentile(risky_scores, HIGH_PERCENTILE))
    if tau_low >= tau_high:
        tau_low, tau_high = min(tau_low, tau_high), max(tau_low, tau_high)

    with open(args.output_json, "w") as f:
        json.dump({"tau_low": tau_low, "tau_high": tau_high}, f, indent=2)
    print(f"Calibrated thresholds: tau_low={tau_low:.4f}, tau_high={tau_high:.4f} -> {args.output_json}")


if __name__ == "__main__":
    main()