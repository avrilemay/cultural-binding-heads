"""Attention extraction and cross-validated L1 head discovery (Stage 2).

Extracted verbatim from pipeline_instruct.ipynb, 'Stage 2: attention binding
CV (head discovery)' cell. SEED and first_device are read from common.config.

run_feature_type_comparison and run_loo (Stage 3 cell) reference notebook
execution-state globals (cv_results, comparison_results), so they stay
verbatim in the notebook and are deliberately NOT extracted here.
"""


from collections import Counter
import numpy as np
import torch
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from common import config


def extract_attention_scores(model, tokenizer, prompt_text, spans):
    """Extract binding scores for all (layer, head) pairs."""
    enc = tokenizer(prompt_text, return_tensors="pt")
    enc = {k: v.to(config.first_device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc, output_attentions=True, use_cache=False)
    n_layers = len(out.attentions)
    n_heads_model = out.attentions[0].shape[1]
    seq_len = out.attentions[0].shape[2]
    has_item = spans['item'] is not None
    item_idx = np.array(spans['item']) if has_item else None
    a_idx = np.array(spans['opt_a'])
    b_idx = np.array(spans['opt_b'])
    scores = {
        'bind_a_to_item': np.full((n_layers, n_heads_model), np.nan),
        'bind_b_to_item': np.full((n_layers, n_heads_model), np.nan),
        'bind_avg':       np.full((n_layers, n_heads_model), np.nan),
    }
    for l in range(n_layers):
        attn = out.attentions[l][0].float().cpu().numpy()
        if has_item:
            scores['bind_a_to_item'][l] = attn[:, a_idx][:, :, item_idx].sum(axis=2).mean(axis=1)
            scores['bind_b_to_item'][l] = attn[:, b_idx][:, :, item_idx].sum(axis=2).mean(axis=1)
            scores['bind_avg'][l] = (scores['bind_a_to_item'][l] +
                                     scores['bind_b_to_item'][l]) / 2.0
    del out, enc
    torch.cuda.empty_cache()
    return scores


N_OUTER_FOLDS = 5


N_INNER_FOLDS = 5


L1_Cs = [0.001, 0.01, 0.1, 1.0, 10.0]


TOP_K_HEADS = 10


def build_Xy(features_match, features_unrel, scenarios, feature_name):
    n = len(features_match)
    if n == 0: return None, None, None, None
    n_layers, n_heads_model = features_match[0][feature_name].shape
    X = np.zeros((2 * n, n_layers * n_heads_model))
    for i in range(n):
        X[i] = features_match[i][feature_name].flatten()
        X[n + i] = features_unrel[i][feature_name].flatten()
    if np.any(np.isnan(X)):
        nan_mask = np.isnan(X).any(axis=1)
        n_half = n
        valid_examples = ~(nan_mask[:n_half] | nan_mask[n_half:])
        n_keep = valid_examples.sum()
        if n_keep < 10: return None, None, None, None
        keep_idx = np.where(valid_examples)[0]
        keep_all = np.concatenate([keep_idx, keep_idx + n_half])
        X = X[keep_all]
        filtered_scenarios = [scenarios[i] for i in keep_idx]
        n = n_keep
    else:
        filtered_scenarios = list(scenarios)
    y = np.concatenate([np.ones(n), np.zeros(n)])
    groups = np.array(filtered_scenarios + filtered_scenarios)
    return X, y, groups, (n_layers, n_heads_model)


def cv_select_C(X, y, groups, n_splits=5, Cs=None):
    if Cs is None: Cs = L1_Cs
    actual_splits = min(n_splits, len(set(groups)))
    if actual_splits < 2: return 0.1, 0.5
    gkf = GroupKFold(n_splits=actual_splits)
    best_auc, best_C = -1, Cs[len(Cs) // 2]
    for C in Cs:
        pipe = Pipeline([("scaler", StandardScaler()),
            ("lr", LogisticRegression(penalty="l1", C=C, solver="liblinear",
                                      max_iter=1000, random_state=config.SEED))])
        try:
            aucs = cross_val_score(pipe, X, y, cv=gkf, groups=groups, scoring="roc_auc")
            if aucs.mean() > best_auc:
                best_auc, best_C = aucs.mean(), C
        except: pass
    return best_C, best_auc


def topk_heads(coef_2d, k=10):
    flat = np.abs(coef_2d).ravel()
    idx = np.argsort(flat)[::-1][:k]
    n_h = coef_2d.shape[1]
    return [(int(i // n_h), int(i % n_h)) for i in idx if coef_2d.ravel()[i] != 0]


def run_outer_cv(all_features, scenarios_valid, feature_name="bind_avg",
                 K=N_OUTER_FOLDS, k_heads=TOP_K_HEADS):
    X, y, groups, shape = build_Xy(
        all_features["B_cult"], all_features["B_unrel"],
        scenarios_valid, feature_name)
    if X is None:
        print(f"  ⚠ Cannot build X for {feature_name}"); return [], {}
    n_unique = len(set(groups))
    actual_K = min(K, n_unique)
    if actual_K < 2: return [], {}

    gkf = GroupKFold(n_splits=actual_K)
    fold_results = []
    print(f"\n  Outer {actual_K}-fold CV on {feature_name} "
          f"({X.shape[0]} rows, {n_unique} scenarios)")
    print(f"  {'Fold':<6s}  {'C':>6s}  {'innerAUC':>9s}  {'testAUC':>8s}  "
          f"{'#nz':>4s}  {'Top heads'}")
    print(f"  {'-' * 70}")

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_tr, y_tr, g_tr = X[tr_idx], y[tr_idx], groups[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]
        best_C, inner_auc = cv_select_C(
            X_tr, y_tr, g_tr, n_splits=min(N_INNER_FOLDS, len(set(g_tr))))
        pipe = Pipeline([("scaler", StandardScaler()),
            ("lr", LogisticRegression(penalty="l1", C=best_C, solver="liblinear",
                                      max_iter=1000, random_state=config.SEED))])
        pipe.fit(X_tr, y_tr)
        coef = pipe.named_steps["lr"].coef_[0].reshape(shape)
        n_nonzero = int((coef != 0).sum())
        try:
            test_auc = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
        except: test_auc = 0.5
        heads_fold = topk_heads(coef, k=k_heads)
        heads_str = ", ".join(f"L{l:02d}H{h:02d}" for l, h in heads_fold[:5])
        print(f"  {fold:<6d}  {best_C:6.3f}  {inner_auc:9.3f}  {test_auc:8.3f}  "
              f"{n_nonzero:4d}  {heads_str}")
        fold_results.append({
            "fold": fold, "best_C": best_C, "inner_auc": inner_auc,
            "test_auc": test_auc, "n_nonzero": n_nonzero,
            "coef": coef, "heads": heads_fold,
            "train_idx": tr_idx, "test_idx": te_idx,
        })

    # ── Aggregate ──
    head_counter = Counter()
    for f in fold_results:
        for lh in f["heads"]:
            head_counter[tuple(lh)] += 1
    stable_heads = [(lh, cnt) for lh, cnt in head_counter.most_common()
                    if cnt >= max(2, len(fold_results) // 2)]
    mean_coef = np.mean([f["coef"] for f in fold_results], axis=0)

    summary = {
        "test_auc_mean": np.mean([f["test_auc"] for f in fold_results]),
        "test_auc_std":  np.std([f["test_auc"] for f in fold_results]),
        "stable_heads":  stable_heads,
        "head_counter":  dict(head_counter),
        "mean_coef":     mean_coef,
        "n_folds":       len(fold_results),
    }

    print(f"\n  Test AUC: {summary['test_auc_mean']:.3f} ± {summary['test_auc_std']:.3f}")
    if stable_heads:
        print(f"  Stable heads (≥ {len(fold_results)//2+1}/{len(fold_results)} folds):")
        for (l, h), cnt in stable_heads:
            sign = "↑ match" if mean_coef[l, h] > 0 else "↓ match"
            print(f"    L{l:02d}H{h:02d}: {cnt}/{len(fold_results)} folds, "
                  f"mean coef={mean_coef[l,h]:+.4f} ({sign})")

    return fold_results, summary
