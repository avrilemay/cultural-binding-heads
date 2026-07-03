"""S/K-score computation for the base pipeline (baseline, knockout, scaled).

Copied verbatim from pipeline_base.ipynb (S-score baseline, edge-knockout,
standalone-KO and dose-response cells). The runtime singleton first_device
was rewritten to common.config.first_device.
"""


import numpy as np
import torch
import torch.nn.functional as F

from common import config
from .hooks import edge_knockout, edge_scale


def compute_s_scores(model, tokenizer, formatted_texts, after_ids,
                     cond_prefix_tid=None):
    """Compute S = log P(c) - logsumexp(log P(a), log P(b)) per prompt.

    Uses COND_AFTER_IDS (discovered from top-50 after prefix) for all models.
    For prefix models: appends prefix token, reads logits at next position.
    For in_prompt models: reads logits at last position directly.

    Returns (scores, coverages, all_lps): scores and coverages as arrays,
    all_lps as a list of per-option logprob dicts.
    """
    scores, coverages, all_lps = [], [], []
    use_prefix = cond_prefix_tid is not None

    for i, text in enumerate(formatted_texts):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = enc["input_ids"].to(config.first_device)
        attn_mask = enc["attention_mask"].to(config.first_device)

        if use_prefix:
            prefix_t = torch.tensor([[cond_prefix_tid]], device=config.first_device)
            input_ids = torch.cat([input_ids, prefix_t], dim=1)
            attn_mask = torch.cat([attn_mask,
                torch.ones(1, 1, device=config.first_device, dtype=attn_mask.dtype)], dim=1)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attn_mask).logits[0, -1, :].float()
        lp = F.log_softmax(logits, dim=-1)
        probs = lp.exp()

        cov = 0
        lps = {}
        for opt in ["a", "b", "c"]:
            tids = after_ids[opt]
            if tids:
                t = torch.tensor(tids, device=config.first_device)
                lps[opt] = torch.logsumexp(lp[t], 0).item()
                cov += probs[t].sum().item()
            else:
                lps[opt] = float("-inf")

        scores.append(lps["c"] - torch.logsumexp(torch.tensor([lps["a"], lps["b"]]), 0).item())
        coverages.append(cov)
        all_lps.append(lps)

        del input_ids, attn_mask, logits, lp, probs
        if i % 100 == 0 and i > 0:
            torch.cuda.empty_cache()

    return np.array(scores), np.array(coverages), all_lps


def compute_s_scores_with_ko(model, tokenizer, formatted_texts, positions_list,
                              heads, edge_mode, after_ids, cond_prefix_tid=None):
    """S-scores with edge knockout applied."""
    scores, coverages = [], []
    use_prefix = cond_prefix_tid is not None

    for i, text in enumerate(formatted_texts):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = enc["input_ids"].to(config.first_device)
        attn_mask = enc["attention_mask"].to(config.first_device)

        if use_prefix:
            prefix_t = torch.tensor([[cond_prefix_tid]], device=config.first_device)
            input_ids = torch.cat([input_ids, prefix_t], dim=1)
            attn_mask = torch.cat([attn_mask,
                torch.ones(1, 1, device=config.first_device, dtype=attn_mask.dtype)], dim=1)

        pos = positions_list[i]
        if pos is None or not heads or edge_mode == "none":
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attn_mask)
        else:
            if edge_mode == "B_to_item":
                src, tgt = pos["item_tokens"], pos["B_tokens"]
            elif edge_mode == "A_to_item":
                src, tgt = pos["item_tokens"], pos["A_tokens"]
            else:
                src, tgt = pos["item_tokens"], pos["B_tokens"] + pos["A_tokens"]
            with torch.no_grad(), edge_knockout(model, heads, src, tgt):
                out = model(input_ids=input_ids, attention_mask=attn_mask)

        logits = out.logits[0, -1, :].float()
        lp = F.log_softmax(logits, dim=-1)
        probs = lp.exp()

        cov = 0
        lps = {}
        for opt in ["a", "b", "c"]:
            tids = after_ids[opt]
            if tids:
                t = torch.tensor(tids, device=config.first_device)
                lps[opt] = torch.logsumexp(lp[t], 0).item()
                cov += probs[t].sum().item()
            else:
                lps[opt] = float("-inf")

        scores.append(lps["c"] - torch.logsumexp(torch.tensor([lps["a"], lps["b"]]), 0).item())
        coverages.append(cov)

        del input_ids, attn_mask, out, logits, lp, probs
        if i % 50 == 0 and i > 0:
            torch.cuda.empty_cache()

    return np.array(scores), np.array(coverages)


def _compute_scores(model, tokenizer, formatted_texts, positions_list,
                    heads, edge_mode, after_ids, cond_prefix_tid=None):
    """Compute S/K scores with optional edge knockout. Standalone.
    Returns (scores, all_lps): array of S/K-scores and list of per-option logprob dicts."""
    scores, all_lps = [], []
    use_prefix = cond_prefix_tid is not None

    for i, text in enumerate(formatted_texts):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = enc["input_ids"].to(config.first_device)
        attn_mask = enc["attention_mask"].to(config.first_device)

        if use_prefix:
            prefix_t = torch.tensor([[cond_prefix_tid]], device=config.first_device)
            input_ids = torch.cat([input_ids, prefix_t], dim=1)
            attn_mask = torch.cat([attn_mask,
                torch.ones(1, 1, device=config.first_device, dtype=attn_mask.dtype)], dim=1)

        pos = positions_list[i] if positions_list else None
        if pos is None or not heads or edge_mode == "none":
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attn_mask)
        else:
            if edge_mode == "B_to_item":
                src, tgt = pos["item_tokens"], pos["B_tokens"]
            elif edge_mode == "A_to_item":
                src, tgt = pos["item_tokens"], pos["A_tokens"]
            else:
                src, tgt = [], []
            with torch.no_grad(), edge_knockout(model, heads, src, tgt):
                out = model(input_ids=input_ids, attention_mask=attn_mask)

        logits = out.logits[0, -1, :].float()
        lp = F.log_softmax(logits, dim=-1)

        lps = {}
        for opt in ["a", "b", "c"]:
            tids = after_ids[opt]
            if tids:
                t = torch.tensor(tids, device=config.first_device)
                lps[opt] = torch.logsumexp(lp[t], 0).item()
            else:
                lps[opt] = float("-inf")

        scores.append(lps["c"] - torch.logsumexp(torch.tensor([lps["a"], lps["b"]]), 0).item())
        all_lps.append(lps)
        del input_ids, attn_mask, out, logits, lp
        if i % 50 == 0 and i > 0:
            torch.cuda.empty_cache()

    return np.array(scores), all_lps


def compute_s_scores_scaled(model, tokenizer, formatted_texts, positions_list,
                            heads, edge_mode, after_ids, alpha=0.0,
                            cond_prefix_tid=None):
    """S-scores with edge scaling at a given alpha."""
    scores = []
    use_prefix = cond_prefix_tid is not None

    for i, text in enumerate(formatted_texts):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = enc["input_ids"].to(config.first_device)
        attn_mask = enc["attention_mask"].to(config.first_device)

        if use_prefix:
            prefix_t = torch.tensor([[cond_prefix_tid]], device=config.first_device)
            input_ids = torch.cat([input_ids, prefix_t], dim=1)
            attn_mask = torch.cat([attn_mask,
                torch.ones(1, 1, device=config.first_device, dtype=attn_mask.dtype)], dim=1)

        pos = positions_list[i]
        if alpha == 1.0 or pos is None or not heads:
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attn_mask)
        else:
            if edge_mode == "B_to_item":
                src, tgt = pos["item_tokens"], pos["B_tokens"]
            else:
                src, tgt = pos["item_tokens"], pos["A_tokens"]
            with torch.no_grad(), edge_scale(model, heads, src, tgt, alpha=alpha):
                out = model(input_ids=input_ids, attention_mask=attn_mask)

        logits = out.logits[0, -1, :].float()
        lp = F.log_softmax(logits, dim=-1)

        lps = {}
        for opt in ["a", "b", "c"]:
            tids = after_ids[opt]
            if tids:
                t = torch.tensor(tids, device=config.first_device)
                lps[opt] = torch.logsumexp(lp[t], 0).item()
            else:
                lps[opt] = float("-inf")

        scores.append(lps["c"] - torch.logsumexp(torch.tensor([lps["a"], lps["b"]]), 0).item())
        del input_ids, attn_mask, out, logits, lp
        if i % 100 == 0 and i > 0:
            torch.cuda.empty_cache()

    return np.array(scores)
