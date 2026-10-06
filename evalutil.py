"""Shared evaluation helpers (used by 05_train, 08_window_gbm, 09_ensemble)."""
import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             average_precision_score)
import config as C


def eval_indices(n, cap=None):
    """Deterministic subsample so EVERY model is scored on the same windows."""
    cap = cap or C.MAX_EVAL_WINDOWS
    idx = np.arange(n)
    if n > cap:
        idx = np.sort(np.random.default_rng(0).choice(n, cap, replace=False))
    return idx


def tune_class_scales(prob, y, min_gain=0.003):
    """Scales for classes 1/2 chosen on VAL. A change is only accepted if it
    beats the untuned val macro-F1 by min_gain, which stops the tuner from
    chasing noise when val holds few Pre-Attack blocks."""
    grid = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
    base = f1_score(y, prob.argmax(1), average="macro")
    best, best_f = np.ones(3), base
    for a in grid:
        for b in grid:
            sc = np.array([1.0, a, b])
            f = f1_score(y, (prob * sc).argmax(1), average="macro")
            if f > best_f + 1e-9:
                best, best_f = sc, f
    if best_f < base + min_gain:
        return np.ones(3), base
    return best, best_f


def subtype_recall(y, pred, sub):
    out = {}
    for name, code in (("infiltration", 1), ("portscan", 2)):
        m = (y == 1) & (sub == code)
        out[name] = {"n": int(m.sum()),
                     "recall": float((pred[m] == 1).mean()) if m.any() else None}
    return out


def make_report(yt, pred, name, out, proba=None, ds_ids=None, sub=None, ds_names=None):
    rep = classification_report(yt, pred, labels=[0, 1, 2], target_names=C.CLASS_NAMES,
                                output_dict=True, zero_division=0)
    print(f"\n=== {name} ({len(yt):,} windows) ===")
    print(classification_report(yt, pred, labels=[0, 1, 2], target_names=C.CLASS_NAMES,
                                zero_division=0, digits=4))
    print("confusion matrix (rows=true):\n", confusion_matrix(yt, pred, labels=[0, 1, 2]))
    if proba is not None and (yt == 1).any():
        rep["pre_attack_ap"] = float(average_precision_score(yt == 1, proba[:, 1]))
        print(f"Pre-Attack average precision (threshold-free): {rep['pre_attack_ap']:.4f}")
    if sub is not None:
        rep["pre_attack_by_subtype"] = subtype_recall(yt, pred, sub)
        print("Pre-Attack recall by subtype:", rep["pre_attack_by_subtype"])
    if ds_ids is not None:
        names = ds_names or list(C.DATASET_DIRS)
        per = {}
        for k in np.unique(ds_ids):
            m = ds_ids == k
            per[names[k] if 0 <= k < len(names) else str(k)] = float(
                f1_score(yt[m], pred[m], average="macro", labels=[0, 1, 2], zero_division=0))
        rep["macro_f1_by_dataset"] = per
        print("macro-F1 by dataset:", {k: round(v, 4) for k, v in per.items()})
    out[name] = rep
    return rep
