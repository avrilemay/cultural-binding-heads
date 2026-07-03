"""Chat formatting, option-token lookup, span detection, and Wang response parsing.

Extracted verbatim from pipeline_instruct.ipynb, 'Shared helpers (data loading,
conditions, span detection)' cell. Config globals (CFG, ANSWER_SUFFIX) are
read from common.config.
"""


import re
import torch
from common import config


def format_for_chat(texts, tokenizer):
    """Format prompts using model-appropriate chat template."""
    if config.CFG["chat_format"] == "gemma":
        return [tokenizer.apply_chat_template(
            [{"role": "user", "content": " " + t + config.ANSWER_SUFFIX}],
            tokenize=False, add_generation_prompt=True) for t in texts]
    else:  # standard: mistral, llama, nemo
        return [tokenizer.apply_chat_template(
            [{"role": "user", "content": t + config.ANSWER_SUFFIX}],
            tokenize=False, add_generation_prompt=True) for t in texts]


def format_for_generation(prompt_text, tokenizer):
    """Format a single prompt for pipeline generation (Wang style)."""
    if config.CFG["chat_format"] == "gemma":
        return [{"role": "user", "content": " " + prompt_text}]
    else:
        return [{"role": "system", "content": ""},
                {"role": "user", "content": prompt_text}]


def find_option_token_ids(tokenizer):
    result = {}
    for opt in ['a', 'b', 'c']:
        ids = set()
        for s in [f'({opt})', opt, opt.upper(), f'({opt.upper()})',
                  f' ({opt})', f' {opt}', f' {opt.upper()}', f' ({opt.upper()})']:
            toks = tokenizer.encode(s, add_special_tokens=False)
            if len(toks) == 1: ids.add(toks[0])
        for p in ['', ' ', '(']:
            for sfx in ['', ')']:
                for c in [opt, opt.upper()]:
                    tok = tokenizer.encode(f"{p}{c}{sfx}", add_special_tokens=False)
                    if len(tok) == 1: ids.add(tok[0])
        result[opt] = sorted(ids)
    return result


def find_subseq(haystack, needle):
    n = len(needle)
    if n == 0: return None
    for i in range(len(haystack) - n + 1):
        if haystack[i:i+n] == needle:
            return list(range(i, i+n))
    return None


def find_span_in_ids(input_ids, fragment, tokenizer):
    variants = [fragment]
    if not fragment.startswith(" "):
        variants.append(" " + fragment)
    stripped = re.sub(r'^(the |an |a )', '', fragment, flags=re.I)
    if stripped != fragment:
        variants.extend([stripped, " " + stripped])
    for v in variants:
        frag_ids = tokenizer(v, add_special_tokens=False).input_ids
        if not frag_ids: continue
        indices = find_subseq(input_ids, frag_ids)
        if indices is not None: return indices
    return None


def detect_spans(input_ids, question_text, opt_a, opt_b, item_text, tokenizer,
                 item_required=True):
    ids_list = input_ids.tolist() if torch.is_tensor(input_ids) else input_ids
    spans = {}
    spans['item'] = find_span_in_ids(ids_list, item_text, tokenizer) if item_text else None
    spans['opt_a'] = find_span_in_ids(ids_list, opt_a, tokenizer)
    spans['opt_b'] = find_span_in_ids(ids_list, opt_b, tokenizer)
    if spans['opt_a'] is None or spans['opt_b'] is None: return None
    if item_required and spans['item'] is None: return None
    if spans['item'] is not None:
        item_set = set(spans['item'])
        if item_set & set(spans['opt_a']) or item_set & set(spans['opt_b']):
            spans['item'] = None
            if item_required: return None
    return spans


def validate_spans(input_ids, spans, tokenizer, label=""):
    ids_list = input_ids.tolist() if torch.is_tensor(input_ids) else input_ids
    print(f"\n  [{label}]")
    for name, indices in spans.items():
        if indices is None:
            print(f"    {name}: NOT FOUND"); continue
        decoded = tokenizer.decode([ids_list[i] for i in indices])
        print(f"    {name}: '{decoded}' (tokens {indices[0]}-{indices[-1]})")


WANG_TEMPLATES = [
    "the answer is ({})", "the correct answer is ({})", "**{}.**",
    "**{}**", "'{}',", "'{}'", "option {}. ", "( {} )",
    "\nanswer {}", "option {}",
]


WANG_REFUSALS = [
    "i cannot", "i can't", "i'm not able", "as an ai",
    "i don't think it's appropriate", "i must respectfully",
    "it would not be appropriate", "i'm unable",
    "cannot provide", "not appropriate to", "would not be fair",
]


def map_to_abc(word):
    """Parse model output to 0/1/2 (a/b/c), -1 (refusal), -2 (unknown)."""
    bina = -2
    word_stripped = word.strip()
    word_lower = word_stripped.lower()
    a = ['a)', '(a)', 'a -', '(a', "'a'", '"a"', "a.", "a,", "a;", "a:", "a "]
    b = ['b)', '(b)', 'b -', '(b', "'b'", '"b"', "b.", "b,", "b;", "b:", "b "]
    c = ['c)', '(c)', 'c -', '(c', "'c'", '"c"', "c.", "c,", "c;", "c:", "c "]
    first3 = word_lower[:3] if len(word_lower) >= 3 else word_lower
    first2 = word_lower[:2] if len(word_lower) >= 2 else word_lower
    if first3 in a or first2 in a or word_lower == 'a': bina = 0
    elif first3 in b or first2 in b or word_lower == 'b': bina = 1
    elif first3 in c or first2 in c or word_lower == 'c': bina = 2
    for temp in WANG_TEMPLATES:
        for c_ind, choice in enumerate(['a', 'b', 'c']):
            if temp.format(choice).lower() in word_lower[:40]: bina = c_ind
        for choice in ['d', 'e']:
            if temp.format(choice).lower() in word_lower[:40]: bina = -1
    if word_stripped == "": bina = -1
    for refusal in WANG_REFUSALS:
        if refusal in word_lower: bina = -1
    if bina == -2:
        if '(a)' in word: bina = 0
        elif '(b)' in word: bina = 1
        elif '(c)' in word: bina = 2
    return bina
