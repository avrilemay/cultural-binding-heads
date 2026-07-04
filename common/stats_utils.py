"""Statistical helpers.

ttest_clustered: extracted verbatim from pipeline_instruct.ipynb
('Knowledge evaluation (standalone)' cell). cond_p_R: identical in both
pipelines; extracted from the same cell. benjamini_hochberg: added at
curation for the Table 2 significance markers.
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


def benjamini_hochberg(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR correction. Returns a boolean array (input
    order) marking the hypotheses rejected at level alpha."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    passed = p[order] <= (np.arange(1, m + 1) / m) * alpha
    kmax = np.nonzero(passed)[0].max() if passed.any() else -1
    reject = np.zeros(m, dtype=bool)
    reject[order[:kmax + 1]] = True
    return reject


def cond_p_R(logprobs_list, assoc_pos):
    out = []
    for lps, ap in zip(logprobs_list, assoc_pos):
        lp_R = lps[ap]
        lp_U = lps['b' if ap == 'a' else 'a']
        m = max(lp_R, lp_U)
        p_R = np.exp(lp_R - m) / (np.exp(lp_R - m) + np.exp(lp_U - m))
        out.append(p_R)
    return np.array(out)
