"""Edge intervention functions (attention-edge knockout and scaling).

Model configuration is read from common.config.CFG.
"""


import importlib
import torch
import torch.nn.functional as F
from contextlib import contextmanager

from common import config


_EDGE_REGISTRY = {}


_SCALE_REGISTRY = {}


def _get_attn_module():
    """Import the model-specific attention module."""
    return importlib.import_module(config.CFG["attn_module"])


@contextmanager
def edge_knockout(model, heads, source_tokens, target_tokens):
    """Zero B->item attention edges in the specified heads.

    Monkey-patches eager_attention_forward for the target architecture.
    Gemma-2 soft-capping is applied when present in model config.
    """
    global _EDGE_REGISTRY
    if not heads or not source_tokens or not target_tokens:
        yield
        return

    attn_mod = _get_attn_module()
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    _EDGE_REGISTRY = {
        "heads": {k: set(v) for k, v in heads.items()},
        "source_tokens": sorted(set(source_tokens)),
        "target_tokens": sorted(set(target_tokens)),
    }

    original_eager = attn_mod.eager_attention_forward
    _n_heads = model.config.num_attention_heads
    _n_kv = getattr(model.config, "num_key_value_heads", _n_heads)
    _kv_group = _n_heads // _n_kv
    _head_dim = getattr(model.config, "head_dim", None) or (model.config.hidden_size // _n_heads)
    _softcap = getattr(model.config, "attn_logit_softcapping", None) if config.CFG["softcap"] else None

    if config.CFG["softcap"]:
        # Gemma-2: scaling and dropout passed as kwargs
        def wrapped_eager(module, query, key, value, attention_mask, **kwargs):
            scaling = kwargs.pop("scaling", None)
            dropout = kwargs.pop("dropout", 0.0)

            layer_idx = getattr(module, "layer_idx", None)
            heads_to_edit = _EDGE_REGISTRY["heads"].get(layer_idx, None)
            if not heads_to_edit:
                return original_eager(module, query, key, value, attention_mask,
                                      scaling=scaling, dropout=dropout, **kwargs)

            attn_output, attn_weights = original_eager(
                module, query, key, value, attention_mask,
                scaling=scaling, dropout=dropout, **kwargs)
            attn_output = attn_output.clone()

            if query.shape[1] == _n_heads: qkv_ax = 1
            elif query.shape[2] == _n_heads: qkv_ax = 2
            else: return attn_output, attn_weights
            out_ax = 1 if attn_output.shape[1] == _n_heads else 2
            seq_q = query.shape[2] if qkv_ax == 1 else query.shape[1]
            seq_k = key.shape[2] if qkv_ax == 1 else key.shape[1]
            src = [s for s in _EDGE_REGISTRY["source_tokens"] if s < seq_k]
            tgt = [t for t in _EDGE_REGISTRY["target_tokens"] if t < seq_q]
            if not src or not tgt: return attn_output, attn_weights

            for h in heads_to_edit:
                kv_h = h // _kv_group
                if qkv_ax == 1: q_h, k_h, v_h = query[:, h], key[:, kv_h], value[:, kv_h]
                else: q_h, k_h, v_h = query[:, :, h], key[:, :, kv_h], value[:, :, kv_h]
                scores = torch.matmul(q_h, k_h.transpose(-2, -1)) * scaling
                if _softcap is not None:
                    scores = _softcap * torch.tanh(scores / _softcap)
                if attention_mask is not None and attention_mask.dim() == 4:
                    m = attention_mask[:, 0, :seq_q, :seq_k] if attention_mask.shape[1] == 1 else attention_mask[:, h, :seq_q, :seq_k]
                    scores = scores + m
                for ti in tgt:
                    for si in src:
                        scores[:, ti, si] = torch.finfo(scores.dtype).min
                new_out = torch.matmul(F.softmax(scores, dim=-1), v_h)
                if out_ax == 1: attn_output[:, h] = new_out
                else: attn_output[:, :, h] = new_out
            return attn_output, attn_weights
    else:
        # Mistral / Llama / Nemo: scaling is a positional argument
        def wrapped_eager(module, query, key, value, attention_mask, scaling,
                          dropout=0.0, **kwargs):
            layer_idx = getattr(module, "layer_idx", None)
            heads_to_edit = _EDGE_REGISTRY["heads"].get(layer_idx, None)
            if not heads_to_edit:
                return original_eager(module, query, key, value, attention_mask,
                                      scaling, dropout=dropout, **kwargs)

            attn_output, attn_weights = original_eager(
                module, query, key, value, attention_mask, scaling,
                dropout=dropout, **kwargs)
            attn_output = attn_output.clone()

            if query.shape[1] == _n_heads: qkv_ax = 1
            elif query.shape[2] == _n_heads: qkv_ax = 2
            else: return attn_output, attn_weights
            out_ax = 1 if attn_output.shape[1] == _n_heads else 2
            seq_q = query.shape[2] if qkv_ax == 1 else query.shape[1]
            seq_k = key.shape[2] if qkv_ax == 1 else key.shape[1]
            src = [s for s in _EDGE_REGISTRY["source_tokens"] if s < seq_k]
            tgt = [t for t in _EDGE_REGISTRY["target_tokens"] if t < seq_q]
            if not src or not tgt: return attn_output, attn_weights

            for h in heads_to_edit:
                kv_h = h // _kv_group
                if qkv_ax == 1: q_h, k_h, v_h = query[:, h], key[:, kv_h], value[:, kv_h]
                else: q_h, k_h, v_h = query[:, :, h], key[:, :, kv_h], value[:, :, kv_h]
                scores = torch.matmul(q_h, k_h.transpose(-2, -1)) * scaling
                if attention_mask is not None and attention_mask.dim() == 4:
                    m = attention_mask[:, 0, :seq_q, :seq_k] if attention_mask.shape[1] == 1 else attention_mask[:, h, :seq_q, :seq_k]
                    scores = scores + m
                for ti in tgt:
                    for si in src:
                        scores[:, ti, si] = torch.finfo(scores.dtype).min
                new_out = torch.matmul(F.softmax(scores, dim=-1), v_h)
                if out_ax == 1: attn_output[:, h] = new_out
                else: attn_output[:, :, h] = new_out
            return attn_output, attn_weights

    attn_mod.eager_attention_forward = wrapped_eager
    orig_reg = ALL_ATTENTION_FUNCTIONS.get("eager", None)
    ALL_ATTENTION_FUNCTIONS["eager"] = wrapped_eager
    try:
        yield
    finally:
        attn_mod.eager_attention_forward = original_eager
        if orig_reg is not None:
            ALL_ATTENTION_FUNCTIONS["eager"] = orig_reg
        _EDGE_REGISTRY = {}


@contextmanager
def edge_scale(model, heads, source_tokens, target_tokens, alpha=0.0):
    """Scale B->item edges by alpha (0=KO, 1=baseline, >1=amplify).

    o_scaled = o_knockout + alpha * (o_normal - o_knockout)
    """
    global _SCALE_REGISTRY
    if not heads or not source_tokens or not target_tokens or alpha == 1.0:
        yield
        return

    attn_mod = _get_attn_module()
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    _SCALE_REGISTRY = {
        "heads": {k: set(v) for k, v in heads.items()},
        "source_tokens": sorted(set(source_tokens)),
        "target_tokens": sorted(set(target_tokens)),
        "alpha": alpha,
    }

    original_eager = attn_mod.eager_attention_forward
    _n_heads = model.config.num_attention_heads
    _n_kv = getattr(model.config, "num_key_value_heads", _n_heads)
    _kv_group = _n_heads // _n_kv
    _head_dim = getattr(model.config, "head_dim", None) or (model.config.hidden_size // _n_heads)
    _softcap = getattr(model.config, "attn_logit_softcapping", None) if config.CFG["softcap"] else None

    if config.CFG["softcap"]:
        # Gemma-2
        def wrapped_eager(module, query, key, value, attention_mask, **kwargs):
            scaling = kwargs.pop("scaling", None)
            dropout = kwargs.pop("dropout", 0.0)

            layer_idx = getattr(module, "layer_idx", None)
            heads_to_edit = _SCALE_REGISTRY["heads"].get(layer_idx, None)
            if not heads_to_edit:
                return original_eager(module, query, key, value, attention_mask,
                                      scaling=scaling, dropout=dropout, **kwargs)

            _alpha = _SCALE_REGISTRY["alpha"]
            attn_output, attn_weights = original_eager(
                module, query, key, value, attention_mask,
                scaling=scaling, dropout=dropout, **kwargs)
            attn_output_normal = attn_output.clone()

            if query.shape[1] == _n_heads: qkv_ax = 1
            elif query.shape[2] == _n_heads: qkv_ax = 2
            else: return attn_output, attn_weights
            out_ax = 1 if attn_output.shape[1] == _n_heads else 2
            seq_q = query.shape[2] if qkv_ax == 1 else query.shape[1]
            seq_k = key.shape[2] if qkv_ax == 1 else key.shape[1]
            src = [s for s in _SCALE_REGISTRY["source_tokens"] if s < seq_k]
            tgt = [t for t in _SCALE_REGISTRY["target_tokens"] if t < seq_q]
            if not src or not tgt: return attn_output, attn_weights

            for h in heads_to_edit:
                kv_h = h // _kv_group
                if qkv_ax == 1: q_h, k_h, v_h = query[:, h], key[:, kv_h], value[:, kv_h]
                else: q_h, k_h, v_h = query[:, :, h], key[:, :, kv_h], value[:, :, kv_h]
                scores = torch.matmul(q_h, k_h.transpose(-2, -1)) * scaling
                if _softcap is not None:
                    scores = _softcap * torch.tanh(scores / _softcap)
                if attention_mask is not None and attention_mask.dim() == 4:
                    m = attention_mask[:, 0, :seq_q, :seq_k] if attention_mask.shape[1] == 1 else attention_mask[:, h, :seq_q, :seq_k]
                    scores = scores + m
                for ti in tgt:
                    for si in src:
                        scores[:, ti, si] = torch.finfo(scores.dtype).min
                ko_out = torch.matmul(F.softmax(scores, dim=-1), v_h)
                normal_h = attn_output_normal[:, h] if out_ax == 1 else attn_output_normal[:, :, h]
                blended = ko_out + _alpha * (normal_h - ko_out)
                if out_ax == 1: attn_output[:, h] = blended
                else: attn_output[:, :, h] = blended
            return attn_output, attn_weights
    else:
        # Mistral / Llama / Nemo
        def wrapped_eager(module, query, key, value, attention_mask, scaling,
                          dropout=0.0, **kwargs):
            layer_idx = getattr(module, "layer_idx", None)
            heads_to_edit = _SCALE_REGISTRY["heads"].get(layer_idx, None)
            if not heads_to_edit:
                return original_eager(module, query, key, value, attention_mask,
                                      scaling, dropout=dropout, **kwargs)

            _alpha = _SCALE_REGISTRY["alpha"]
            attn_output, attn_weights = original_eager(
                module, query, key, value, attention_mask, scaling,
                dropout=dropout, **kwargs)
            attn_output_normal = attn_output.clone()

            if query.shape[1] == _n_heads: qkv_ax = 1
            elif query.shape[2] == _n_heads: qkv_ax = 2
            else: return attn_output, attn_weights
            out_ax = 1 if attn_output.shape[1] == _n_heads else 2
            seq_q = query.shape[2] if qkv_ax == 1 else query.shape[1]
            seq_k = key.shape[2] if qkv_ax == 1 else key.shape[1]
            src = [s for s in _SCALE_REGISTRY["source_tokens"] if s < seq_k]
            tgt = [t for t in _SCALE_REGISTRY["target_tokens"] if t < seq_q]
            if not src or not tgt: return attn_output, attn_weights

            for h in heads_to_edit:
                kv_h = h // _kv_group
                if qkv_ax == 1: q_h, k_h, v_h = query[:, h], key[:, kv_h], value[:, kv_h]
                else: q_h, k_h, v_h = query[:, :, h], key[:, :, kv_h], value[:, :, kv_h]
                scores = torch.matmul(q_h, k_h.transpose(-2, -1)) * scaling
                if attention_mask is not None and attention_mask.dim() == 4:
                    m = attention_mask[:, 0, :seq_q, :seq_k] if attention_mask.shape[1] == 1 else attention_mask[:, h, :seq_q, :seq_k]
                    scores = scores + m
                for ti in tgt:
                    for si in src:
                        scores[:, ti, si] = torch.finfo(scores.dtype).min
                ko_out = torch.matmul(F.softmax(scores, dim=-1), v_h)
                normal_h = attn_output_normal[:, h] if out_ax == 1 else attn_output_normal[:, :, h]
                blended = ko_out + _alpha * (normal_h - ko_out)
                if out_ax == 1: attn_output[:, h] = blended
                else: attn_output[:, :, h] = blended
            return attn_output, attn_weights

    attn_mod.eager_attention_forward = wrapped_eager
    orig_reg = ALL_ATTENTION_FUNCTIONS.get("eager", None)
    ALL_ATTENTION_FUNCTIONS["eager"] = wrapped_eager
    try:
        yield
    finally:
        attn_mod.eager_attention_forward = original_eager
        if orig_reg is not None:
            ALL_ATTENTION_FUNCTIONS["eager"] = orig_reg
        _SCALE_REGISTRY = {}
