"""Statistical helpers.

ttest_clustered: extracted verbatim from pipeline_instruct.ipynb
('Knowledge evaluation (standalone)' cell). cond_p_R: identical in both
pipelines; extracted from the same cell.
"""


import numpy as np
from scipy.stats import ttest_rel


def ttest_clustered(diffs_a, diffs_b, items):
    """Paired t-test on item-level means (n=66) to account for
    within-item dependence. Returns t, p."""
    unique = np.unique(items)
    mean_a = np.array([diffs_a[items == it].mean() for it in unique])
    mean_b = np.array([diffs_b[items == it].mean() for it in unique])
    return ttest_rel(mean_a, mean_b)


def cond_p_R(logprobs_list, assoc_pos):
    out = []
    for lps, ap in zip(logprobs_list, assoc_pos):
        lp_R = lps[ap]
        lp_U = lps['b' if ap == 'a' else 'a']
        m = max(lp_R, lp_U)
        p_R = np.exp(lp_R - m) / (np.exp(lp_R - m) + np.exp(lp_U - m))
        out.append(p_R)
    return np.array(out)
