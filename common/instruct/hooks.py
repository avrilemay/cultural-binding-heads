"""Attention-edge interventions: edge knockout, edge alpha-scaling, and scoring.

CFG lookups are read from common.config.

Note: knowledge_heads/khead_discovery_instruct.ipynb deliberately defines and
uses its own local variant of this machinery, whose leak-window semantics
differ; see the note in that notebook before importing from here in its place.
"""


import importlib
from contextlib import contextmanager
import numpy as np
import torch
import torch.nn.functional as F
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS as _AAF
from common import config


_EDGE_REGISTRY = {}


@contextmanager
def edge_knockout(model, heads, source_tokens, target_tokens):
    """Zero attention edges from source → target for specific heads.
    Automatically handles Gemma-2 soft-capping via config.CFG['has_softcapping'].
    """
    global _EDGE_REGISTRY
    if not heads or not source_tokens or not target_tokens:
        yield; return

    attn_module = importlib.import_module(config.CFG["attn_module"])
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    _EDGE_REGISTRY = {
        'heads': {k: set(v) for k, v in heads.items()},
        'source_tokens': sorted(set(source_tokens)),
        'target_tokens': sorted(set(target_tokens)),
    }

    original_eager = attn_module.eager_attention_forward
    n_heads = model.config.num_attention_heads
    n_kv_heads = getattr(model.config, 'num_key_value_heads', n_heads)
    kv_group_size = n_heads // n_kv_heads
    head_dim = getattr(model.config, "head_dim", None) or (
        model.config.hidden_size // n_heads)
    softcapping = getattr(model.config, 'attn_logit_softcapping', None) \
                  if config.CFG["has_softcapping"] else None

    def wrapped_eager(module, query, key, value, attention_mask, **kwargs):
        scaling = kwargs.pop('scaling', None)
        dropout = kwargs.pop('dropout', 0.0)
        layer_idx = getattr(module, 'layer_idx', None)
        heads_to_edit = _EDGE_REGISTRY['heads'].get(layer_idx, None)
        if not heads_to_edit:
            return original_eager(module, query, key, value, attention_mask,
                                  scaling=scaling, dropout=dropout, **kwargs)

        attn_output, attn_weights = original_eager(
            module, query, key, value, attention_mask,
            scaling=scaling, dropout=dropout, **kwargs)
        attn_output = attn_output.clone()

        # Detect head axis
        if query.shape[1] == n_heads:   qkv_head_axis = 1
        elif query.shape[2] == n_heads: qkv_head_axis = 2
        else: return attn_output, attn_weights

        if attn_output.shape[1] == n_heads:   out_head_axis = 1
        elif attn_output.shape[2] == n_heads: out_head_axis = 2
        else: return attn_output, attn_weights

        seq_len_q = query.shape[2] if qkv_head_axis == 1 else query.shape[1]
        seq_len_k = key.shape[2] if qkv_head_axis == 1 else key.shape[1]
        src = [s for s in _EDGE_REGISTRY['source_tokens'] if s < seq_len_k]
        tgt = [t for t in _EDGE_REGISTRY['target_tokens'] if t < seq_len_q]
        if not src or not tgt:
            return attn_output, attn_weights

        for h in heads_to_edit:
            kv_h = h // kv_group_size
            if qkv_head_axis == 1:
                q_h, k_h, v_h = query[:, h, :, :], key[:, kv_h, :, :], value[:, kv_h, :, :]
            else:
                q_h, k_h, v_h = query[:, :, h, :], key[:, :, kv_h, :], value[:, :, kv_h, :]

            attn_scores = torch.matmul(q_h, k_h.transpose(-2, -1)) * scaling
            # Gemma-2 soft-capping
            if softcapping is not None:
                attn_scores = attn_scores / softcapping
                attn_scores = torch.tanh(attn_scores)
                attn_scores = attn_scores * softcapping
            if attention_mask is not None and attention_mask.dim() == 4:
                if attention_mask.shape[1] == 1:
                    attn_scores = attn_scores + attention_mask[:, 0, :seq_len_q, :seq_len_k]
                else:
                    attn_scores = attn_scores + attention_mask[:, h, :seq_len_q, :seq_len_k]
            # Zero the target edges
            for t_idx in tgt:
                for s_idx in src:
                    attn_scores[:, t_idx, s_idx] = torch.finfo(attn_scores.dtype).min
            attn_probs = F.softmax(attn_scores, dim=-1)
            new_output = torch.matmul(attn_probs, v_h)
            if out_head_axis == 1:
                attn_output[:, h, :, :] = new_output
            else:
                attn_output[:, :, h, :] = new_output

        return attn_output, attn_weights

    # Monkey-patch
    attn_module.eager_attention_forward = wrapped_eager
    orig_entry = None
    try:
        orig_entry = ALL_ATTENTION_FUNCTIONS.get('eager', None)
        ALL_ATTENTION_FUNCTIONS['eager'] = wrapped_eager
    except: pass
    try:
        yield
    finally:
        attn_module.eager_attention_forward = original_eager
        if orig_entry is not None:
            try: ALL_ATTENTION_FUNCTIONS['eager'] = orig_entry
            except: pass


_SCALE_REGISTRY = {}


@contextmanager
def edge_scale(model, heads, source_tokens, target_tokens, alpha=0.0):
    """Scale attention edges: α=0 → full KO, α=1 → baseline, α>1 → amplify."""
    global _SCALE_REGISTRY
    if not heads or not source_tokens or not target_tokens or alpha == 1.0:
        yield; return

    attn_module = importlib.import_module(config.CFG["attn_module"])
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    _SCALE_REGISTRY = {
        'heads': {k: set(v) for k, v in heads.items()},
        'source_tokens': sorted(set(source_tokens)),
        'target_tokens': sorted(set(target_tokens)),
        'alpha': alpha,
    }

    original_eager = attn_module.eager_attention_forward
    n_heads = model.config.num_attention_heads
    n_kv_heads = getattr(model.config, 'num_key_value_heads', n_heads)
    kv_group_size = n_heads // n_kv_heads
    head_dim = getattr(model.config, "head_dim", None) or (
        model.config.hidden_size // n_heads)
    softcapping = getattr(model.config, 'attn_logit_softcapping', None) \
                  if config.CFG["has_softcapping"] else None

    def wrapped_eager(module, query, key, value, attention_mask, **kwargs):
        scaling = kwargs.pop('scaling', None)
        dropout = kwargs.pop('dropout', 0.0)
        layer_idx = getattr(module, 'layer_idx', None)
        heads_to_edit = _SCALE_REGISTRY['heads'].get(layer_idx, None)
        if not heads_to_edit:
            return original_eager(module, query, key, value, attention_mask,
                                  scaling=scaling, dropout=dropout, **kwargs)

        alpha_val = _SCALE_REGISTRY['alpha']

        # Normal forward
        attn_output, attn_weights = original_eager(
            module, query, key, value, attention_mask,
            scaling=scaling, dropout=dropout, **kwargs)
        attn_output = attn_output.clone()

        if query.shape[1] == n_heads:   qkv_head_axis = 1
        elif query.shape[2] == n_heads: qkv_head_axis = 2
        else: return attn_output, attn_weights

        if attn_output.shape[1] == n_heads:   out_head_axis = 1
        elif attn_output.shape[2] == n_heads: out_head_axis = 2
        else: return attn_output, attn_weights

        seq_len_q = query.shape[2] if qkv_head_axis == 1 else query.shape[1]
        seq_len_k = key.shape[2] if qkv_head_axis == 1 else key.shape[1]
        src = [s for s in _SCALE_REGISTRY['source_tokens'] if s < seq_len_k]
        tgt = [t for t in _SCALE_REGISTRY['target_tokens'] if t < seq_len_q]
        if not src or not tgt:
            return attn_output, attn_weights

        for h in heads_to_edit:
            kv_h = h // kv_group_size
            if qkv_head_axis == 1:
                q_h, k_h, v_h = query[:, h, :, :], key[:, kv_h, :, :], value[:, kv_h, :, :]
            else:
                q_h, k_h, v_h = query[:, :, h, :], key[:, :, kv_h, :], value[:, :, kv_h, :]

            attn_scores = torch.matmul(q_h, k_h.transpose(-2, -1)) * scaling
            if softcapping is not None:
                attn_scores = attn_scores / softcapping
                attn_scores = torch.tanh(attn_scores)
                attn_scores = attn_scores * softcapping
            if attention_mask is not None and attention_mask.dim() == 4:
                if attention_mask.shape[1] == 1:
                    attn_scores = attn_scores + attention_mask[:, 0, :seq_len_q, :seq_len_k]
                else:
                    attn_scores = attn_scores + attention_mask[:, h, :seq_len_q, :seq_len_k]

            # KO version: zero target edges
            attn_scores_ko = attn_scores.clone()
            for t_idx in tgt:
                for s_idx in src:
                    attn_scores_ko[:, t_idx, s_idx] = torch.finfo(attn_scores_ko.dtype).min
            attn_probs_ko = F.softmax(attn_scores_ko, dim=-1)
            out_ko = torch.matmul(attn_probs_ko, v_h)

            # Normal version
            attn_probs_normal = F.softmax(attn_scores, dim=-1)
            out_normal = torch.matmul(attn_probs_normal, v_h)

            # Blend: out = out_ko + α * (out_normal - out_ko)
            new_output = out_ko + alpha_val * (out_normal - out_ko)

            if out_head_axis == 1:
                attn_output[:, h, :, :] = new_output
            else:
                attn_output[:, :, h, :] = new_output

        return attn_output, attn_weights

    attn_module.eager_attention_forward = wrapped_eager
    orig_entry = None
    try:
        orig_entry = ALL_ATTENTION_FUNCTIONS.get('eager', None)
        ALL_ATTENTION_FUNCTIONS['eager'] = wrapped_eager
    except: pass
    try:
        yield
    finally:
        attn_module.eager_attention_forward = original_eager
        if orig_entry is not None:
            try: ALL_ATTENTION_FUNCTIONS['eager'] = orig_entry
            except: pass


def compute_logit_scores_edge(model, tokenizer, formatted_texts, raw_texts,
                               heads, positions_list, edge_mode, option_tokens,
                               return_lps=False):
    """Compute S = logP(c) - logsumexp(logP(a), logP(b)) with optional edge KO.
    If return_lps=True, also returns a list of per-option logprob dicts."""
    scores = []
    all_lps = []
    first_device = next(model.parameters()).device
    for i, text in enumerate(formatted_texts):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        enc = {k: v.to(first_device) for k, v in enc.items()}
        pos = positions_list[i]

        if not heads or pos is None:
            with torch.no_grad():
                outputs = model(**enc)
        else:
            if edge_mode == 'B_to_item':
                src, tgt = pos['item_tokens'], pos['B_tokens']
            elif edge_mode == 'A_to_item':
                src, tgt = pos['item_tokens'], pos['A_tokens']
            else:
                src, tgt = [], []
            with torch.no_grad(), edge_knockout(model, heads, src, tgt):
                outputs = model(**enc)

        logits = outputs.logits[0, -1, :].float()
        lp = F.log_softmax(logits, dim=-1)
        lp_a = torch.logsumexp(lp[option_tokens['a']], dim=0).item()
        lp_b = torch.logsumexp(lp[option_tokens['b']], dim=0).item()
        lp_c = torch.logsumexp(lp[option_tokens['c']], dim=0).item()
        scores.append(lp_c - torch.logsumexp(torch.tensor([lp_a, lp_b]), dim=0).item())
        if return_lps:
            all_lps.append({'a': lp_a, 'b': lp_b, 'c': lp_c})
        del enc, outputs, logits, lp
        if i % 100 == 0 and i > 0:
            torch.cuda.empty_cache()
    return (np.array(scores), all_lps) if return_lps else np.array(scores)


def compute_scores_scaled(model, tokenizer, formatted_texts, positions_list,
                           heads_dict, edge_mode, option_tokens, alpha=1.0):
    """Compute S-scores with edge_scale (α-scaling)."""
    scores = []
    first_device = next(model.parameters()).device
    for i, text in enumerate(formatted_texts):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        enc = {k: v.to(first_device) for k, v in enc.items()}
        pos = positions_list[i]

        if alpha == 1.0 or pos is None or not heads_dict:
            with torch.no_grad():
                outputs = model(**enc)
        else:
            if edge_mode == 'B_to_item':
                src, tgt = pos['item_tokens'], pos['B_tokens']
            elif edge_mode == 'A_to_item':
                src, tgt = pos['item_tokens'], pos['A_tokens']
            else:
                src, tgt = [], []
            with torch.no_grad(), edge_scale(model, heads_dict, src, tgt, alpha=alpha):
                outputs = model(**enc)

        logits = outputs.logits[0, -1, :].float()
        lp = F.log_softmax(logits, dim=-1)
        lp_a = torch.logsumexp(lp[option_tokens['a']], dim=0).item()
        lp_b = torch.logsumexp(lp[option_tokens['b']], dim=0).item()
        lp_c = torch.logsumexp(lp[option_tokens['c']], dim=0).item()
        scores.append(lp_c - torch.logsumexp(torch.tensor([lp_a, lp_b]), dim=0).item())
        del enc, outputs, logits, lp
        if i % 100 == 0 and i > 0:
            torch.cuda.empty_cache()
    return np.array(scores)


# leak fix from khead_discovery v4: edge_knockout leaves its wrapper
# registered; call between knockout contexts
def reset_eager_attention():
    """Purge the wrapper edge_knockout leaks into ALL_ATTENTION_FUNCTIONS
    and clear the stale edge registry (see header note)."""
    global _EDGE_REGISTRY
    if hasattr(_AAF, '_local_mapping'):
        _AAF._local_mapping.pop('eager', None)
    _EDGE_REGISTRY = {}
