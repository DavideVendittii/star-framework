"""
Extracts geometric residual stream activations (layers 12-16) for training Sentinel probes.

Usage:
    python sentinel/build_sentinel_dataset.py --model_id meta-llama/Llama-3.1-8B-Instruct --dataset_csv data/star_dataset_train.csv
"""

import argparse
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

HF_TOKEN = os.environ.get("HF_TOKEN")
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
    parser.add_argument("--model_id", required=True, help="HuggingFace model ID")
    parser.add_argument("--dataset_csv", default="data/star_dataset_train.csv")
    parser.add_argument("--out_prefix", default="sentinel_activations")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(args.dataset_csv)
    df.columns = df.columns.str.strip().str.lower()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map={"": device}, token=HF_TOKEN
    )
    model.eval()

    X_list = []
    for p in tqdm(df["prompt"], desc="Extracting activations"):
        X_list.append(extract_features(model, tokenizer, p, device))

    X = np.array(X_list)
    y = df["label"].values.astype(int)

    np.save(f"{args.out_prefix}_X.npy", X)
    np.save(f"{args.out_prefix}_y.npy", y)
    print(f"Saved activations: {args.out_prefix}_X.npy {X.shape}, {args.out_prefix}_y.npy {y.shape}")


if __name__ == "__main__":
    main()