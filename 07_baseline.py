"""
Stage 7 - LightGBM single-flow baseline on the SAME block split.

  python 07_baseline.py                          # pooled
  python 07_baseline.py --holdout cic2017 --tag lgbm_xds
  python 07_baseline.py --seed 2 --tag lgbm_s2

Every deep-learning IDS paper should beat (or honestly lose to) this.
Writes artifacts/report{tag}.json in the same format as 05_train.py, so
collect_results.py puts both in one table.
"""
import argparse
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report, f1_score

import config as C


def effective_split(split, ds, holdout):
    if holdout is None:
        return split
    if ds == holdout:
        return np.full_like(split, 2)
    return np.where(split == 2, 0, split)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default=None)
    ap.add_argument("--portscan-as", choices=["pre", "attack"], default="pre")
    ap.add_argument("--tag", default="lgbm")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--keep-ratio", type=float, default=0.5)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    feats = json.load(open(C.ARTIFACT_DIR / "selected_features.json"))
    Xs, ys, ss, ds_ids = [], [], [], []
    names = list(C.DATASET_DIRS)
    for f in sorted(C.PARQUET_DIR.glob("*.parquet")):
        ds = f.name.split("__")[0]
        d = pd.read_parquet(f, columns=feats + ["__y", "__sub", "__split"])
        y = d["__y"].to_numpy(np.int8)
        if args.portscan_as == "attack":
            y = np.where(d["__sub"].to_numpy() == 2, 2, y).astype(np.int8)
        Xs.append(d[feats].to_numpy(np.float32)); ys.append(y)
        ss.append(effective_split(d["__split"].to_numpy(), ds, args.holdout))
        ds_ids.append(np.full(len(d), names.index(ds) if ds in names else -1, np.int8))
    X, y, s, dsid = map(np.concatenate, (Xs, ys, ss, ds_ids))
    print({n: int((s == i).sum()) for i, n in enumerate(("train", "val", "test"))})

    tr = np.flatnonzero(s == 0)
    benign = tr[y[tr] == 0]
    keep = np.concatenate([rng.choice(benign, int(len(benign) * args.keep_ratio), replace=False),
                           tr[y[tr] != 0]])
    va, te = np.flatnonzero(s == 1), np.flatnonzero(s == 2)
    if len(va) > C.MAX_EVAL_WINDOWS:
        va = rng.choice(va, C.MAX_EVAL_WINDOWS, replace=False)
    cnt = np.bincount(y[keep], minlength=3).astype(float)
    w_cls = np.sqrt(cnt.sum() / (3 * np.maximum(cnt, 1)))               # softened balanced
    model = lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=600,
                               learning_rate=0.05, num_leaves=63, subsample=0.8,
                               subsample_freq=1, colsample_bytree=0.8,
                               random_state=args.seed, n_jobs=-1, verbose=-1)
    model.fit(X[keep], y[keep], sample_weight=w_cls[y[keep]],
              eval_set=[(X[va], y[va])],
              callbacks=[lgb.early_stopping(30, verbose=False)])

    report = {}
    for name, idx in (("val", va), ("test", te)):
        pred = model.predict(X[idx])
        report[name] = classification_report(y[idx], pred, labels=[0, 1, 2],
                                             target_names=C.CLASS_NAMES, output_dict=True,
                                             zero_division=0)
        print(f"\n=== {name} ({len(idx):,} flows) ===")
        print(classification_report(y[idx], pred, labels=[0, 1, 2], target_names=C.CLASS_NAMES,
                                    digits=4, zero_division=0))
        if name == "test":
            per = {names[k]: float(f1_score(y[idx][dsid[idx] == k], pred[dsid[idx] == k],
                                            average="macro", labels=[0, 1, 2], zero_division=0))
                   for k in np.unique(dsid[idx]) if 0 <= k < len(names)}
            report[name]["macro_f1_by_dataset"] = per
            print("macro-F1 by dataset:", per)
    imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
    imp.to_csv(C.ARTIFACT_DIR / f"lgbm_importance_{args.tag}.csv")
    print("\ntop features:\n", imp.head(10).to_string())
    json.dump({"args": vars(args), "reports": report},
              open(C.ARTIFACT_DIR / f"report_{args.tag}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
