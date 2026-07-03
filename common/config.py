"""Shared configuration for all notebooks.

The two model tables are copied verbatim from the canonical pipeline notebooks:
- MODEL_CONFIGS_INSTRUCT: pipeline_instruct.ipynb, "Model configuration" cell
- MODELS_BASE:            pipeline_base.ipynb, "Model configuration" cell

Usage in every notebook:
    from common import config
    config.init(MODEL_KEY, VARIANT)   # VARIANT in {"instruct", "base"}
    ... load model/tokenizer ...
    config.model, config.tokenizer, config.first_device = model, tokenizer, first_device
"""
import os
import torch
from pathlib import Path

MODEL_CONFIGS_INSTRUCT = {
    "mistral": {
        "model_path":       "mistralai/Mistral-7B-Instruct-v0.3",
        "attn_module":      "transformers.models.mistral.modeling_mistral",
        "has_softcapping":  False,
        "chat_format":      "standard",   # system + user messages
        "heads":            {8: [16], 9: [23], 12: [9]},
        "label":            "Mistral-7B-Instruct-v0.3",
    },
    "llama": {
        "model_path":       "meta-llama/Llama-3.1-8B-Instruct",
        "attn_module":      "transformers.models.llama.modeling_llama",
        "has_softcapping":  False,
        "chat_format":      "standard",
        "heads":            {7: [7], 8: [17]},
        "label":            "Llama-3.1-8B-Instruct",
    },
    "gemma2": {
        "model_path":       "google/gemma-2-9b-it",
        "attn_module":      "transformers.models.gemma2.modeling_gemma2",
        "has_softcapping":  True,   # tanh(scores/cap)*cap before softmax
        "chat_format":      "gemma", # no system msg, leading space in user content
        "heads":            {11: [14], 13: [13], 16: [15]},
        "label":            "Gemma-2-9B-it",
    },
    "nemo": {
        "model_path":       "mistralai/Mistral-Nemo-Instruct-2407",
        "attn_module":      "transformers.models.mistral.modeling_mistral",
        "has_softcapping":  False,
        "chat_format":      "standard",
        "heads":            {8: [9], 10: [24]},
        "label":            "Mistral-Nemo-12B-Instruct",
    },
}

# NB: historical key naming kept as-is — the base table uses "gemma"
# where the instruct table uses "gemma2".
MODELS_BASE = {
    "mistral": {
        "path":  "mistralai/Mistral-7B-v0.3",
        "heads": {8: [16], 9: [23], 12: [9]},
        "attn_module": "transformers.models.mistral.modeling_mistral",
        "softcap": False,
        "cond_approach": "prefix",
        "cond_token_str": "(",
    },
    "llama": {
        "path":  "meta-llama/Meta-Llama-3.1-8B",
        "heads": {7: [7], 8: [17]},
        "attn_module": "transformers.models.llama.modeling_llama",
        "softcap": False,
        "cond_approach": "prefix",
        "cond_token_str": " (",
    },
    "gemma": {
        "path":  "google/gemma-2-9b",
        "heads": {11: [14], 13: [13], 16: [15]},
        "attn_module": "transformers.models.gemma2.modeling_gemma2",
        "softcap": True,
        "cond_approach": "prefix",
        "cond_token_str": "(",
    },
    "nemo": {
        "path":  "mistralai/Mistral-Nemo-Base-2407",
        "heads": {8: [9], 10: [24]},
        "attn_module": "transformers.models.mistral.modeling_mistral",
        "softcap": False,
        "cond_approach": "in_prompt",
        "cond_token_str": "(",
    },
}

SEED = 42

# All paths are anchored on the repo root (derived from this file's location)
# so notebooks work regardless of the kernel's working directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = str(REPO_ROOT / "data") + os.sep
RESULTS_ROOT = REPO_ROOT / "results"
HF_TOKEN = os.environ.get("HF_TOKEN")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ANSWER_SUFFIX = "\nAnswer with a single letter: A, B, or C.\nAnswer:"

# Wang et al. MC suffix (for generation-based evaluation).
# MC_WANG is the historical name used by the base pipeline; same string.
MC = ("Please answer by selecting only one multiple choice option, "
      "for example '(a)', '(b)', or '(c)'. Do not add any other words. ")
MC_WANG = MC

MAX_NEW_TOKENS = 5

# ── mutable state, set by init() ──────────────────────────────────
MODEL_KEY = None      # active model key
ACTIVE_MODEL = None   # historical alias used by the instruct notebooks
VARIANT = None        # "instruct" | "base"
CFG = None
HEADS = None
ANSWER_START = None   # base variant only
OUTPUT_DIR = None

# runtime singletons, set by the notebook after loading the model
model = None
tokenizer = None
tok = None            # historical alias used by some notebooks
first_device = None


def init(model_key, variant):
    """Select the active model/variant. Returns CFG."""
    global MODEL_KEY, ACTIVE_MODEL, VARIANT, CFG, HEADS, ANSWER_START, OUTPUT_DIR
    assert variant in ("instruct", "base")
    table = MODEL_CONFIGS_INSTRUCT if variant == "instruct" else MODELS_BASE
    assert model_key in table, f"unknown model key {model_key!r} for {variant}: {list(table)}"
    MODEL_KEY = ACTIVE_MODEL = model_key
    VARIANT = variant
    CFG = table[model_key]
    HEADS = CFG["heads"]
    if variant == "base":
        # verbatim from pipeline_base.ipynb: Nemo embeds '(' directly in
        # ANSWER_START; other models append it dynamically via
        # COND_PREFIX_TID in a second forward pass.
        if CFG["cond_approach"] == "in_prompt":
            ANSWER_START = (
                "Answering between the provided multiple choice "
                "letter options, I would choose letter ("
            )
        else:
            ANSWER_START = (
                "Answering between the provided multiple choice "
                "letter options, I would choose letter "
            )
    OUTPUT_DIR = RESULTS_ROOT / f"{model_key}_{variant}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    label = CFG.get("label", CFG.get("path", model_key))
    print(f"Active model: {label} ({variant})")
    print(f"  heads: {HEADS}")
    print(f"  output dir: {OUTPUT_DIR}")
    return CFG
