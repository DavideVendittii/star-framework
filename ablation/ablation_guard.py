"""
Config-driven guard implementation for module-wise ablation studies.

Usage:
    python ablation/ablation_guard.py --model_id meta-llama/Llama-3.1-8B-Instruct --sentinel_path sentinel_model.pkl --thresholds_path meta_policy_thresholds.json --config_name star_full
"""

import argparse
import json
from dataclasses import dataclass
from typing import Optional
import joblib
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class AblationConfig:
    name: str
    use_abstractor: bool
    use_meta_policy: bool
    binary_threshold_key: str = "tau_high"


ABLATION_CONFIGS = {
    "sentinel": AblationConfig(name="sentinel", use_abstractor=False, use_meta_policy=False),
    "sentinel_abstractor": AblationConfig(name="sentinel_abstractor", use_abstractor=True, use_meta_policy=False),
    "sentinel_metapolicy": AblationConfig(name="sentinel_metapolicy", use_abstractor=False, use_meta_policy=True),
    "star_full": AblationConfig(name="star_full", use_abstractor=True, use_meta_policy=True),
}

SYSTEM_PROMPTS_3WAY = {
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

SYSTEM_PROMPTS_BINARY = {
    "ALIGN": None,
    "COUNTER": SYSTEM_PROMPTS_3WAY["COUNTER"],
}


def select_action_3way(risk_score: float, tau_low: float, tau_high: float) -> str:
    if risk_score < tau_low:
        return "ALIGN"
    elif risk_score < tau_high:
        return "CLARIFY"
    else:
        return "COUNTER"


def select_action_binary(risk_score: float, tau: float) -> str:
    return "COUNTER" if risk_score >= tau else "ALIGN"


class PrivacyAbstractor(nn.Module):
    def __init__(self, hidden_size, device, dtype):
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False).to(device=device, dtype=dtype)
        nn.init.eye_(self.projection.weight)
        self.dtype = dtype

    def forward(self, hidden_states):
        return self.projection(hidden_states.to(self.dtype))


class AblationGuard:
    def __init__(
        self,
        model_id: str,
        sentinel_path: str,
        abstractor_weights_path: Optional[str],
        thresholds_path: str,
        target_layers: list,
        config: AblationConfig,
        hf_token: Optional[str] = None,
        device: str = "cuda:0",
    ):
        self.config = config
        self.target_layers = target_layers
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map={"": device}, torch_dtype=torch.bfloat16, token=hf_token
        )
        self.model.eval()

        self.sentinel = joblib.load(sentinel_path)

        with open(thresholds_path) as f:
            thresholds = json.load(f)
        self.tau_low = thresholds["tau_low"]
        self.tau_high = thresholds["tau_high"]
        self.tau_binary = thresholds[config.binary_threshold_key]

        self.abstractors = None
        if abstractor_weights_path is not None:
            self.abstractors = {}
            for layer_idx in target_layers:
                abst = PrivacyAbstractor(self.model.config.hidden_size, self.model.device, torch.bfloat16)
                self.abstractors[layer_idx] = abst
            state_dicts = torch.load(abstractor_weights_path, map_location=self.model.device)
            for layer_idx, sd in state_dicts.items():
                self.abstractors[layer_idx].load_state_dict(sd)
                self.abstractors[layer_idx].eval()

        self._probe_activations = {}

    def _probe_hook(self, module, inp, out, layer_idx):
        hidden_states = out[0] if isinstance(out, tuple) else out
        extracted = hidden_states[0, -1, :] if hidden_states.dim() == 3 else hidden_states[-1, :]
        self._probe_activations[layer_idx] = extracted.detach().to(torch.float32).cpu()
        return out

    def _probe_and_decide(self, user_query: str):
        self._probe_activations = {}
        handles = [
            self.model.model.layers[idx].register_forward_hook(
                lambda mod, inp, out, i=idx: self._probe_hook(mod, inp, out, i)
            )
            for idx in self.target_layers
        ]
        msgs = [{"role": "user", "content": user_query}]
        formatted = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(formatted, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        with torch.no_grad():
            _ = self.model(**inputs)
        for h in handles:
            h.remove()

        feats = np.concatenate(
            [self._probe_activations[i].numpy() for i in self.target_layers]
        ).reshape(1, -1)
        risk_score = float(self.sentinel.predict_proba(feats)[0][1])

        if self.config.use_meta_policy:
            action = select_action_3way(risk_score, self.tau_low, self.tau_high)
        else:
            action = select_action_binary(risk_score, self.tau_binary)

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

    def generate_guarded(self, user_query: str, max_new_tokens: int = 150):
        risk_score, action = self._probe_and_decide(user_query)

        apply_abstractor = self.config.use_abstractor and action != "ALIGN"

        system_prompts = SYSTEM_PROMPTS_3WAY if self.config.use_meta_policy else SYSTEM_PROMPTS_BINARY
        system_prompt = system_prompts[action]

        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_query})
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)

        handles = []
        if apply_abstractor:
            handles = [
                self.model.model.layers[idx].register_forward_hook(
                    lambda mod, i, out, li=idx: self._abstractor_hook(mod, i, out, li)
                )
                for idx in self.target_layers
            ]

        with torch.no_grad():
            output = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
            )

        for h in handles:
            h.remove()

        generated_text = self.tokenizer.decode(
            output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return generated_text, action, risk_score