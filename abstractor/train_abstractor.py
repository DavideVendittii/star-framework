"""
Trains layer-wise Abstractor projections on risky prompts to enforce content-invariance.

Usage:
    python abstractor/train_abstractor.py --model_id meta-llama/Llama-3.1-8B-Instruct --sentinel_path sentinel_model.pkl
"""

import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_TOKEN = os.environ.get("HF_TOKEN")
TARGET_LAYERS = [12, 13, 14, 15, 16]

class PrivacyAbstractor(nn.Module):
    def __init__(self, hidden_size, device, dtype=torch.float32):
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False).to(device=device, dtype=dtype)
        nn.init.eye_(self.projection.weight)
        self.dtype = dtype

    def forward(self, hidden_states):
        return self.projection(hidden_states.to(self.dtype))


class MultiLayerAbstractor(nn.Module):
    def __init__(self, hidden_size_per_layer, target_layers, device, dtype=torch.float32):
        super().__init__()
        self.target_layers = target_layers
        self.abstractors = nn.ModuleDict({
            str(layer): PrivacyAbstractor(hidden_size_per_layer, device, dtype)
            for layer in target_layers
        })

    def forward(self, concatenated_hidden_states):
        chunks = torch.chunk(concatenated_hidden_states, len(self.target_layers), dim=-1)
        projected_chunks = []
        for i, layer_idx in enumerate(self.target_layers):
            proj = self.abstractors[str(layer_idx)](chunks[i])
            projected_chunks.append(proj)
        return torch.cat(projected_chunks, dim=-1)

    def save_per_layer_state_dicts(self, path):
        state_dicts = {layer: self.abstractors[str(layer)].state_dict() for layer in self.target_layers}
        torch.save(state_dicts, path)


def build_frozen_sentinel_head(sentinel_sklearn, device, dtype=torch.float32):
    coef = torch.tensor(sentinel_sklearn.coef_[0], dtype=dtype, device=device)
    intercept = torch.tensor(sentinel_sklearn.intercept_[0], dtype=dtype, device=device)

    head = nn.Linear(coef.shape[0], 1, bias=True).to(device=device, dtype=dtype)
    with torch.no_grad():
        head.weight.copy_(coef.unsqueeze(0))
        head.bias.copy_(intercept)
    for p in head.parameters():
        p.requires_grad = False
    return head


def extract_activations(model, tokenizer, prompts, target_layers, device):
    activations = {}
    def make_hook(layer_idx):
        def hook(mod, inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            activations[layer_idx] = hidden[0, -1, :].detach().to(torch.float32).cpu()
        return hook

    handles = [model.model.layers[idx].register_forward_hook(make_hook(idx)) for idx in target_layers]
    all_features = []
    with torch.no_grad():
        for prompt in tqdm(prompts, desc="Extracting risky activations"):
            activations.clear()
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            _ = model(**inputs)
            feats = torch.cat([activations[i] for i in target_layers], dim=-1)
            all_features.append(feats)

    for h in handles:
        h.remove()
    return torch.stack(all_features)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--sentinel_path", required=True)
    parser.add_argument("--train_csv", default="data/star_dataset_train.csv")
    parser.add_argument("--output_weights", default="abstractor_weights.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16, device_map={"": device}, token=HF_TOKEN)
    model.eval()

    sentinel = joblib.load(args.sentinel_path)
    df = pd.read_csv(args.train_csv)
    df.columns = df.columns.str.strip().str.lower()
    risky_prompts = df[df["label"] == 1]["prompt"].tolist()

    X_risky = extract_activations(model, tokenizer, risky_prompts, TARGET_LAYERS, device)
    hidden_size_per_layer = X_risky.shape[-1] // len(TARGET_LAYERS)

    risky_centroid = X_risky.mean(dim=0, keepdim=True)

    abstractor = MultiLayerAbstractor(hidden_size_per_layer, TARGET_LAYERS, device, dtype=torch.float32)
    frozen_sentinel_head = build_frozen_sentinel_head(sentinel, device, dtype=torch.float32)

    optimizer = torch.optim.Adam(abstractor.parameters(), lr=args.lr)
    bce_loss = nn.BCEWithLogitsLoss()

    X_risky = X_risky.to(device)
    risky_centroid = risky_centroid.to(device)
    target_labels = torch.ones(args.batch_size, 1, device=device)

    n = X_risky.shape[0]
    for epoch in range(args.epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            batch = X_risky[idx]
            bsz = batch.shape[0]

            projected = abstractor(batch)

            l_centroid = ((projected - risky_centroid) ** 2).mean()
            logits = frozen_sentinel_head(projected)
            l_preserve = bce_loss(logits, target_labels[:bsz])

            loss = 1.0 * l_centroid + 2.0 * l_preserve

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | loss={epoch_loss/n_batches:.4f}")

    abstractor.save_per_layer_state_dicts(args.output_weights)
    print(f"Saved abstractor weights to {args.output_weights}")


if __name__ == "__main__":
    main()