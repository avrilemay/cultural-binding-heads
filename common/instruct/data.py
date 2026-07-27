"""Data loading, factorial-pair construction, and knowledge probes (instruct pipeline).

Config globals (MC) are read from common.config; text parsers are imported
from common.text_parsers.
"""


import re
import pickle
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations
import numpy as np
from common import config
from common.text_parsers import norm_identity, extract_options, parse_meta, replace_options


MC_PREFIX = config.MC


def _display_form(s):
    return re.sub(r"^(a |an |the )", "", s.strip(), flags=re.I)


def extract_neutral_item(q, qtype):
    if qtype == 0:
        for pat in [r'wearing\s+(?:a\s+)?(.+?)\?',
                    r'wearing\s+(?:a\s+)?(.+?)(?:\n|\()',
                    r'centered around wearing\s+(?:a\s+)?(.+?)\.']:
            m = re.search(pat, q)
            if m and len(m.group(1).strip()) < 40:
                return m.group(1).strip().rstrip('.')
    elif qtype == 1:
        for pat in [r'popularity of\s+(.+?),',
                    r'not mass produce and sell\s+(.+?)\?',
                    r'not start a company around making and selling\s+(.+?)\?',
                    r'not make\s+(.+?)\s+to\s+sell',
                    r'not make\s+(.+?)(?:\?|\n|\()',
                    r'around (?:making and )?(?:selling )?(.+?)\?']:
            m = re.search(pat, q)
            if m and len(m.group(1).strip()) < 50:
                return m.group(1).strip().rstrip('.')
    elif qtype == 2:
        for pat in [r'session on\s+(.+?),', r'speak on\s+(.+?)\s+as',
                    r'about their personal experience with\s+(.+?)[.,]',
                    r'experience with\s+(.+?)[.,]']:
            m = re.search(pat, q)
            if m: return m.group(1).strip()
    elif qtype == 3:
        m = re.search(r'who is (?:a |an )?(\w+)', q)
        if m: return m.group(1).strip()
    return None


def load_n4(data_dir):
    for fname in ["N4_1k__1_.pkl", "N4_1k.pkl"]:
        fpath = Path(data_dir) / fname
        if fpath.exists():
            with open(fpath, "rb") as f:
                data = pickle.load(f)
            print(f"  Loaded {fname}")
            return data[0], data[1]
    raise FileNotFoundError("N4 pkl not found in " + str(data_dir))


def build_factorial_pairs_v3(cultural_items, seed=42):
    rng = np.random.RandomState(seed)
    norm_to_display = {}
    item_variants = defaultdict(lambda: defaultdict(list))
    item_groups = defaultdict(set)

    for idx, item in enumerate(cultural_items):
        q, ans, meta = item
        try:
            qtype, c_item, in_groups, out_groups = parse_meta(meta)
        except: continue
        oa, ob = extract_options(q)
        if oa is None: continue
        for g in in_groups + out_groups + [oa, ob]:
            gn = norm_identity(g)
            if gn not in norm_to_display:
                norm_to_display[gn] = _display_form(g)
        cult_opt, matched_group = None, None
        for g in in_groups:
            if norm_identity(g) == norm_identity(oa):
                cult_opt, matched_group = 'a', norm_identity(g); break
            elif norm_identity(g) == norm_identity(ob):
                cult_opt, matched_group = 'b', norm_identity(g); break
        if cult_opt is None: continue
        item_variants[c_item][matched_group].append({
            'idx': idx, 'q': q, 'ans': ans, 'cult_opt': cult_opt,
        })
        item_groups[c_item].add(matched_group)

    def display(norm_id):
        return norm_to_display.get(norm_id, norm_id)

    all_items = sorted(item_variants.keys())
    all_identities = sorted(set(g for item in all_items for g in item_variants[item]))
    print(f"  {len(all_items)} items, {len(all_identities)} identities")

    pairs = []
    third_id_usage = Counter()
    n_skipped = 0

    for item1, item2 in combinations(all_items, 2):
        if item_groups[item1] & item_groups[item2]:
            n_skipped += 1; continue
        for v1 in sorted(item_variants[item1].keys()):
            for v2 in sorted(item_variants[item2].keys()):
                id1 = "the " + display(v1)
                id2 = "the " + display(v2)
                if norm_identity(id1) == norm_identity(id2): continue
                t1 = item_variants[item1][v1]
                t2 = item_variants[item2][v2]
                q1 = t1[rng.randint(len(t1))]
                q2 = t2[rng.randint(len(t2))]
                excluded = item_groups[item1] | item_groups[item2] | {v1, v2}
                eligible = [g for g in all_identities if g not in excluded]
                min_u = min(third_id_usage.get(g, 0) for g in eligible)
                cands = [g for g in eligible if third_id_usage.get(g, 0) == min_u]
                tg = cands[rng.randint(len(cands))]
                tid = "the " + display(tg)
                third_id_usage[tg] += 1
                for item, q_sel, corr, wrong, pos_pref in [
                    (item1, q1, v1, v2, id1),
                    (item2, q2, v2, v1, id2),
                ]:
                    corr_id = "the " + display(corr)
                    wrong_id = "the " + display(wrong)
                    if rng.rand() < 0.5:
                        m = replace_options(q_sel['q'], new_a=corr_id, new_b=tid)
                        mm = replace_options(q_sel['q'], new_a=wrong_id, new_b=tid)
                        pos = 'a'
                    else:
                        m = replace_options(q_sel['q'], new_a=tid, new_b=corr_id)
                        mm = replace_options(q_sel['q'], new_a=tid, new_b=wrong_id)
                        pos = 'b'
                    if m == mm: continue
                    pairs.append({
                        'match': m + "\n\n" + MC_PREFIX,
                        'mismatch': mm + "\n\n" + MC_PREFIX,
                        'item': item, 'correct_group': corr,
                        'wrong_group': wrong, 'third_group': tg,
                        'assoc_pos': pos,
                    })

    print(f"  {n_skipped} item pairs skipped (shared group)")
    print(f"  {len(pairs)} raw pairs")

    # Balance: each identity T times correct AND T times wrong
    best_T, best_sel = 0, []
    for T in range(30, 0, -1):
        rng_bal = np.random.RandomState(seed)
        shuffled = list(pairs); rng_bal.shuffle(shuffled)
        ic, iw = Counter(), Counter()
        sel = []
        for p in shuffled:
            cg, wg = p['correct_group'], p['wrong_group']
            if ic[cg] < T and iw[wg] < T:
                sel.append(p); ic[cg] += 1; iw[wg] += 1
        worst = max(abs(ic.get(i, 0) - iw.get(i, 0)) for i in all_identities)
        if worst == 0:
            best_T, best_sel = T, sel; break
    if best_T == 0:
        best_T = 8
        rng_bal = np.random.RandomState(seed)
        shuffled = list(pairs); rng_bal.shuffle(shuffled)
        ic, iw = Counter(), Counter()
        best_sel = []
        for p in shuffled:
            cg, wg = p['correct_group'], p['wrong_group']
            if ic[cg] < best_T and iw[wg] < best_T:
                best_sel.append(p); ic[cg] += 1; iw[wg] += 1

    selected = best_sel
    id_c = Counter(p['correct_group'] for p in selected)
    id_w = Counter(p['wrong_group'] for p in selected)
    n_perfect = sum(1 for i in all_identities if id_c.get(i, 0) == id_w.get(i, 0))
    print(f"  T={best_T}: {len(selected)} pairs, {n_perfect}/{len(all_identities)} balanced")
    return selected


def build_factorial_as_conditions(cultural_items, seed=42):
    factorial_pairs = build_factorial_pairs_v3(cultural_items, seed=seed)
    n = len(factorial_pairs)
    data = {
        'B_cult':        [p['match'] for p in factorial_pairs],
        'B_unrel':       [p['mismatch'] for p in factorial_pairs],
        'scenarios':     [p['item'] for p in factorial_pairs],
        'items_cult':    [p['item'] for p in factorial_pairs],
        'assoc_pos':     [p['assoc_pos'] for p in factorial_pairs],
        'correct_group': [p['correct_group'] for p in factorial_pairs],
        'wrong_group':   [p['wrong_group'] for p in factorial_pairs],
        'third_group':   [p['third_group'] for p in factorial_pairs],
    }
    print(f"\n  Factorial -> conditions wrapper:")
    print(f"    {n} pairs, {len(set(data['scenarios']))} unique items")
    return data


KNOWLEDGE_OPT_C = "Neither is particularly associated"


def build_knowledge_probes(data):
    """Build knowledge probes from factorial pairs.

    Same (a)/(b) identities as the binding task, but the scenario is
    replaced by a direct association question:
    'Which cultural identity is most associated with [item]?'

    In match:    associated identity is present -> correct = assoc_pos
    In mismatch: it is swapped out             -> correct = (c)
    """
    n = len(data['B_cult'])
    knowledge = {'K_cult': [], 'K_unrel': []}

    for i in range(n):
        item = data['items_cult'][i]

        for cond_in, cond_out in [('B_cult', 'K_cult'), ('B_unrel', 'K_unrel')]:
            q_text = data[cond_in][i].split("\n\n")[0]
            oa, ob = extract_options(q_text)
            probe = (
                f"Which of the following cultural identities is most "
                f"associated with {item}?\n"
                f"(a) {oa} (b) {ob} (c) {KNOWLEDGE_OPT_C}"
            )
            knowledge[cond_out].append(probe + "\n\n" + MC_PREFIX)

    print(f"\n  Knowledge probes: {n} pairs")
    for i in range(min(3, n)):
        print(f"    [{i}] item={data['items_cult'][i]}")
        oa_m, ob_m = extract_options(knowledge['K_cult'][i].split("\n\n")[0])
        oa_mm, ob_mm = extract_options(knowledge['K_unrel'][i].split("\n\n")[0])
        print(f"         K_cult:  (a) {oa_m}  (b) {ob_m}")
        print(f"         K_unrel: (a) {oa_mm}  (b) {ob_mm}")

    return knowledge
