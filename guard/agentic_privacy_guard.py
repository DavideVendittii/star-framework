"""
Core inference-time runtime guard for the STAR framework.

Executes a two-pass mechanism:
1. Probe Pass: Measures residual stream risk score across target layers using Sentinel.
2. Generation Pass: Applies one-shot Abstractor projections during prefill when triggered.
"""

import json
import os
import torch
import torch.nn as nn
import numpy as np
import joblib
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_TOKEN = os.environ.get("HF_TOKEN")
LAYERS_TO_MONITOR = [12, 13, 14, 15, 16]


class PrivacyAbstractor(nn.Module):
    def __init__(self, hidden_size, device, dtype):
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False).to(device=device, dtype=dtype)
        nn.init.eye_(self.projection.weight)
        self.dtype = dtype

    def forward(self, hidden_states):
        return self.projection(hidden_states.to(self.dtype))


def select_action(risk_score: float, tau_low: float, tau_high: float) -> str:
    if risk_score < tau_low:
        return "ALIGN"
    elif risk_score < tau_high:
        return "CLARIFY"
    else:
        return "COUNTER"


SYSTEM_PROMPTS = {
    "ALIGN": None,
    "CLARIFY": (
        "You are a privacy-conscious AI. This request appears ambiguous with respect to "
        "privacy sensitivity. Follow this protocol:\n"
        "#THOUGHT: Explain why the request's privacy implications are unclear.\n"
        "#ACTION: #CLARIFY\n"
        "#CONTENT: Ask the user a clarifying question before proceeding."
    ),
    "COUNTER": (
        "You are a privacy-conscious AI. This request has been assessed as privacy-risky. "
        "Follow this protocol:\n"
        "#THOUGHT: Explain why the request violates privacy.\n"
        "#ACTION: #COUNTER\n"
        "#CONTENT: Politely decline and explain why."
    ),
}


class AgenticPrivacyGuard:
    def __init__(self, model_id, sentinel_path, abstractor_weights_path, thresholds_path, target_layers=LAYERS_TO_MONITOR, device="cuda:0"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map={"": device}, torch_dtype=torch.bfloat16, token=HF_TOKEN
        )
        self.model.eval()
        self.sentinel = joblib.load(sentinel_path)
        self.target_layers = target_layers

        with open(thresholds_path) as f:
            thresholds = json.load(f)
        self.tau_low = thresholds["tau_low"]
        self.tau_high = thresholds["tau_high"]

        self.abstractors = {}
        for layer_idx in target_layers:
            abstractor = PrivacyAbstractor(self.model.config.hidden_size, self.model.device, torch.bfloat16)
            self.abstractors[layer_idx] = abstractor
        state_dicts = torch.load(abstractor_weights_path, map_location=self.model.device)
        for layer_idx, sd in state_dicts.items():
            self.abstractors[layer_idx].load_state_dict(sd)
            self.abstractors[layer_idx].eval()

        self._probe_activations = {}

    def _build_chat_text(self, user_query, system_prompt=None):
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_query})
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _probe_hook(self, module, inp, out, layer_idx):
        hidden_states = out[0] if isinstance(out, tuple) else out
        extracted = hidden_states[0, -1, :] if hidden_states.dim() == 3 else hidden_states[-1, :]
        self._probe_activations[layer_idx] = extracted.detach().to(torch.float32).cpu()
        return out

    def _probe_and_decide(self, user_query):
        self._probe_activations = {}
        handles = [
            self.model.model.layers[idx].register_forward_hook(
                lambda mod, inp, out, i=idx: self._probe_hook(mod, inp, out, i)
            )
            for idx in self.target_layers
        ]
        text = self._build_chat_text(user_query)
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        with torch.no_grad():
            self.model(**inputs, use_cache=False)
        for h in handles:
            h.remove()

        features_flat = np.concatenate(
            [self._probe_activations[i].numpy() for i in self.target_layers]
        ).reshape(1, -1)
        risk_score = float(self.sentinel.predict_proba(features_flat)[0][1])
        action = select_action(risk_score, self.tau_low, self.tau_high)
        return risk_score, action

    def _abstractor_hook(self, module, inp, out, layer_idx):
        hidden_states = out[0] if isinstance(out, tuple) else out
        if hidden_states.shape[-2] > 1:
            projected_last = self.abstractors[layer_idx](hidden_states[:, -1:, :])
            hidden_states = hidden_states.clone()
            hidden_states[:, -1:, :] = projected_last
            if isinstance(out, tuple):
                out[0].data = hidden_states
                return out
            return hidden_states
        return out

    def generate_guarded(self, user_query, max_new_tokens=150):
        risk_score, action = self._probe_and_decide(user_query)

        system_prompt = SYSTEM_PROMPTS[action]
        text = self._build_chat_text(user_query, system_prompt)
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)

        handles = []
        if action != "ALIGN":
            handles = [
                self.model.model.layers[idx].register_forward_hook(
                    lambda mod, inp, out, i=idx: self._abstractor_hook(mod, inp, out, i)
                )
                for idx in self.target_layers
            ]

        with torch.no_grad():
            output = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id
            )

        for h in handles:
            h.remove()

        generated_text = self.tokenizer.decode(
            output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return generated_text, action, risk_score