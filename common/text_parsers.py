"""Text parsers shared verbatim by the base and instruct pipelines.

Extracted from the instruct pipeline, 'Shared helpers' cell; the base
pipeline defines byte-identical copies (verified at curation time).
"""


import re
import ast


def norm_identity(s):
    return re.sub(r"^(a |an |the )", "", s.lower().strip())


def extract_options(q):
    ma = re.search(r"\(a\)\s*(.+?)(?:\s*\(b\))", q, re.I)
    mb = re.search(r"\(b\)\s*(.+?)(?:\s*\(c\))", q, re.I)
    if ma and mb:
        return ma.group(1).strip(), mb.group(1).strip()
    return None, None


def extract_scenario(meta_str):
    try:
        return ast.literal_eval(meta_str[meta_str.index("-") + 1:])[0]
    except Exception:
        return meta_str


def parse_meta(meta_str):
    dash_idx = meta_str.index("-")
    qtype = int(meta_str[:dash_idx])
    payload = ast.literal_eval(meta_str[dash_idx + 1:])
    return qtype, payload[0], payload[1], payload[2]


def replace_options(q, new_a=None, new_b=None):
    if new_a is not None:
        q = re.sub(r'(\(a\)\s*)(.+?)(\s*\(b\))',
                   lambda m: m.group(1) + new_a + m.group(3), q, flags=re.I)
    if new_b is not None:
        q = re.sub(r'(\(b\)\s*)(.+?)(\s*\(c\))',
                   lambda m: m.group(1) + new_b + m.group(3), q, flags=re.I)
    return q
