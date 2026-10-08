# STAR: Secure Targeted Activation Regulator for Agentic Privacy Protection in LLMs

Official evaluation code and implementation for **STAR** (*Secure Targeted Activation Regulator*), an inference-time activation-steering framework for agentic privacy protection in Large Language Models.

---

## 📌 Repository Structure

```text
star-framework/
├── README.md
├── requirements.txt
├── sentinel/
│   ├── build_sentinel_dataset.py       # Residual stream activation extraction (layers 12-16)
│   └── train_sentinel.py               # Geometric risk probe training (Logistic Regression)
├── abstractor/
│   ├── train_abstractor.py             # Layer-wise content-invariant linear projection training
│   └── analyze_abstractor_invariance.py # Content-invariance evaluation metrics
├── meta_policy/
│   └── calibrate_meta_policy.py        # Empirical threshold calibration (tau_low, tau_high)
├── guard/
│   └── agentic_privacy_guard.py        # Inference-time two-pass runtime guard
├── benchmark/
│   ├── evaluate_guard_vs_baseline.py   # Main evaluation suite (STAR vs Baseline)
│   └── run_ood_eval.py                 # Out-Of-Distribution evaluation (JBB & WildJailbreak)
├── ablation/
│   ├── ablation_guard.py               # Config-driven guard for component ablations
│   └── run_ablation.py                 # Runner for full ablation matrix (Table 3)
└── utils/
    ├── split_dataset.py                # Dataset partitioning (D_sentinel, D_calib, D_test)
    └── aggregate_star_results.py       # Metrics aggregator and LaTeX table generator
