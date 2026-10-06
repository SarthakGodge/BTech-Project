"""
Stage 8 - LightGBM on WINDOW statistics (no neural network).

For every window of T flows it computes, per selected feature:
mean, std, min, max, last-flow value and (last - first). Trees are very good
at the tabular part of this problem, and the window statistics give them the
sequence context. Scores are written in the same format as 05_train.py, and
val/test probabilities are saved so 09_ensemble.py can blend them with the
CNN-LSTM on exactly the same windows.

  python 08_window_gbm.py --tag gbm_s1 --seed 1
  python 08_window_gbm.py --seq holdout_cic2017_psattack --tag gbm_2018to2017_s1
"""
import argparse
import json
import numpy as np
import lightgbm as lgb

import config as C
from evalutil import eval_indices, tune_class_scales, make_report

GBM_MAX_TRAIN = C.GBM_MAX_TRAIN


def window_stats(X, idx, chunk=50_000):
    """X: memmap (N, T, F). Returns (len(idx), 6*F) float32."""
    out = []
    for i in range(0, len(idx), chunk):
        w = np.asarray(X[idx[i:i + chunk]], dtype=np.float32)         # (b, T, F)
        out.append(np.concatenate([w.mean(1), w.std(1), w.min(1), w.max(1),
                                   w[:, -1], w[:, -1] - w[:, 0]], axis=1))
    return np.concatenate(out) if out else np.zeros((0, 6 * X.shape[2]), np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="")
    ap.add_argument("--tag", default="gbm")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--keep-ratio", type=float, default=0.5)
    ap.add_argument("--no-tune", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    tag = f"_{args.tag}"
    seq = C.SEQ_DIR / args.seq if args.seq else C.SEQ_DIR

    Xtr, ytr = np.load(seq / "X_train.npy", mmap_mode="r"), np.load(seq / "y_train.npy")
    Xva, yva = np.load(seq / "X_val.npy", mmap_mode="r"), np.load(seq / "y_val.npy")
    Xte, yte = np.load(seq / "X_test.npy", mmap_mode="r"), np.load(seq / "y_test.npy")

    benign = np.flatnonzero(ytr == 0)
    benign = rng.choice(benign, int(len(benign) * args.keep_ratio), replace=False)
    idx = np.concatenate([benign, np.flatnonzero(ytr != 0)])
    if len(idx) > GBM_MAX_TRAIN:
        idx = rng.choice(idx, GBM_MAX_TRAIN, replace=False)
    idx = np.sort(idx)
    cnt = np.bincount(ytr[idx], minlength=3).astype(float)
    print("GBM train windows per class:", cnt.astype(int).tolist())
    w_cls = np.sqrt(cnt.sum() / (3 * np.maximum(cnt, 1)))

    va_i, te_i = eval_indices(len(yva)), eval_indices(len(yte))
    print("computing window statistics ...")
    A, Av, At = window_stats(Xtr, idx), window_stats(Xva, va_i), window_stats(Xte, te_i)
    y, yv, yt = ytr[idx], yva[va_i], yte[te_i]

    present = np.unique(y)
    if len(present) < 3:
        print(f"WARNING: training split has only classes {present.tolist()} - the missing "
              f"class gets probability 0 (expected for some cross-dataset runs).")
    keep_v = np.isin(yv, present)          # early stopping can only score classes it trained on
    model = lgb.LGBMClassifier(n_estimators=800,       # objective auto: binary / multiclass
                               learning_rate=0.05, num_leaves=63, subsample=0.8,
                               subsample_freq=1, colsample_bytree=0.6, min_child_samples=30,
                               reg_lambda=1.0, random_state=args.seed, n_jobs=-1, verbose=-1)
    model.fit(A, y, sample_weight=w_cls[y], eval_set=[(Av[keep_v], yv[keep_v])],
              callbacks=[lgb.early_stopping(40, verbose=False)])
    pv, pt = model.predict_proba(Av), model.predict_proba(At)
    # classes missing from training come back with fewer columns
    def full(p):
        o = np.zeros((len(p), 3), np.float32); o[:, model.classes_] = p; return o
    pv, pt = full(pv), full(pt)

    dte = np.load(seq / "d_test.npy")[te_i] if (seq / "d_test.npy").exists() else None
    sub = np.load(seq / "s_test.npy")[te_i] if (seq / "s_test.npy").exists() else None
    report = {}
    make_report(yv, pv.argmax(1), "val", report, proba=pv)
    make_report(yt, pt.argmax(1), "test", report, proba=pt, ds_ids=dte, sub=sub)
    if not args.no_tune and len(yv):
        sc, f = tune_class_scales(pv, yv)
        print(f"\nclass scales tuned on val: {sc.tolist()} (val macro-F1 {f:.4f})")
        make_report(yt, (pt * sc).argmax(1), "test_tuned", report, proba=pt * sc,
                    ds_ids=dte, sub=sub)
        json.dump({"class_scale": sc.tolist()}, open(C.ARTIFACT_DIR / f"thresholds{tag}.json", "w"))
    np.save(C.ARTIFACT_DIR / f"val_probs{tag}.npy", pv); np.save(C.ARTIFACT_DIR / f"val_idx{tag}.npy", va_i)
    np.save(C.ARTIFACT_DIR / f"test_probs{tag}.npy", pt); np.save(C.ARTIFACT_DIR / f"test_idx{tag}.npy", te_i)
    json.dump({"args": vars(args), "reports": report},
              open(C.ARTIFACT_DIR / f"report{tag}.json", "w"), indent=2)
    print(f"\nsaved artifacts with tag '{tag}'")


if __name__ == "__main__":
    main()
