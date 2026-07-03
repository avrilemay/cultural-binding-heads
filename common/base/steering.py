"""Generation-time steering for the base pipeline (DiffAware).

Copied verbatim from pipeline_base.ipynb ("DiffAware generation-time
steering" cell). ANSWER_START was rewritten to common.config.ANSWER_START.
"""


import torch

from common import config
from .prompts import clean_response, map_to_abc
from .hooks import edge_scale


def generate_steered(pipe, data_list, positions_list, heads, alpha,
                     edge_mode="B_to_item", max_new_tokens=5):
    """Generate answers with edge scaling at a given alpha."""
    results = []
    for i, d in enumerate(data_list):
        if i % 100 == 0 and i > 0:
            print(f"      {i}/{len(data_list)}")
            torch.cuda.empty_cache()

        prompt_text = d["prompt"] + "\n" + config.ANSWER_START
        pos = positions_list[i] if positions_list else None

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens, max_length=None,
            do_sample=False, num_return_sequences=1,
            pad_token_id=pipe.tokenizer.eos_token_id,
        )

        if alpha == 1.0 or pos is None or not heads:
            outputs = pipe(prompt_text, **gen_kwargs)
        else:
            src_toks = pos["item_tokens"]
            tgt_toks = pos["B_tokens"] if edge_mode == "B_to_item" else pos["A_tokens"]
            with edge_scale(pipe.model, heads, src_toks, tgt_toks, alpha=alpha):
                outputs = pipe(prompt_text, **gen_kwargs)

        generated = outputs[0]["generated_text"][len(prompt_text):]
        generated = clean_response(generated)
        parsed = map_to_abc(generated)
        results.append((parsed, generated))

    return results
