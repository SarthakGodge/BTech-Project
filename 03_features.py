"""
Stage 3 - Feature selection and scaling.

Fitted on a sample of the TRAIN DAYS ONLY. Fitting on the full dataset
would leak test-day statistics into the scaler and the selector, and
your reported metrics would be optimistic by several points.

Outputs to artifacts/:
  selected_features.json  - the chosen column names, in order
  scaler.pkl              - StandardScaler fitted on train days
  mi_scores.csv           - mutual information per feature (report figure)

Run:  python 03_features.py
"""
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.preprocessing import StandardScaler

from config import (PARQUET_DIR, ARTIFACT_DIR, DAY_SPLITS, LEAK_COLUMNS,
                    VARIANCE_THRESHOLD, K_FEATURES, SEED)

# Mutual information is O(n log n) per feature - sample rather than use 13M rows.
MI_SAMPLE = 400_000
SCALER_SAMPLE = 2_000_000


def train_files():
    return [PARQUET_DIR / f"{d}.parquet" for d, s in DAY_SPLITS.items()
            if s == "train" and (PARQUET_DIR / f"{d}.parquet").exists()]


def stratified_sample(files, n_total, rng):
    """Sample rows across train days, over-representing rare classes."""
    per_file = max(1, n_total // len(files))
    parts = []
    for f in files:
        df = pd.read_parquet(f)
        y = df["__y"].to_numpy()
        idx = []
        for cls in (0, 1, 2):
            cls_idx = np.flatnonzero(y == cls)
            if len(cls_idx) == 0:
                continue
            take = min(len(cls_idx), per_file // 3)
            idx.append(rng.choice(cls_idx, size=take, replace=False))
        if idx:
            parts.append(df.iloc[np.concatenate(idx)])
        del df
    return pd.concat(parts, ignore_index=True)


def main():
    rng = np.random.default_rng(SEED)
    files = train_files()
    if not files:
        raise SystemExit("No train-day parquet files found.")
    print(f"Fitting on {len(files)} train days: {[f.stem for f in files]}")

    sample = stratified_sample(files, MI_SAMPLE * 3, rng)
    y = sample["__y"].to_numpy()

    feature_cols = [c for c in sample.columns if c not in LEAK_COLUMNS]
    X = sample[feature_cols].to_numpy(dtype=np.float32)
    print(f"Sample for selection: {X.shape}")

    # --- 1. Variance threshold: drop near-constant columns -------------
    vt = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
    vt.fit(X)
    kept = np.array(feature_cols)[vt.get_support()]
    dropped = sorted(set(feature_cols) - set(kept))
    print(f"\nVarianceThreshold: {len(feature_cols)} -> {len(kept)} features")
    print(f"  dropped (near-constant): {dropped}")

    Xv = X[:, vt.get_support()]

    # --- 2. Mutual information ranking ---------------------------------
    # MI over f_classif: f_classif measures LINEAR separability only, and
    # flow features separate the classes non-linearly (a SYN count of 1 is
    # suspicious, 0 and 50 are not). MI catches that; ANOVA F does not.
    sub = rng.choice(len(Xv), size=min(MI_SAMPLE, len(Xv)), replace=False)
    print(f"\nComputing mutual information on {len(sub):,} rows "
          f"(this takes a few minutes)...")
    mi = mutual_info_classif(Xv[sub], y[sub], random_state=SEED,
                             discrete_features=False)

    order = np.argsort(mi)[::-1]
    selected = list(kept[order][:K_FEATURES])

    mi_df = pd.DataFrame({"feature": kept[order], "mutual_info": mi[order]})
    mi_df["selected"] = mi_df["feature"].isin(selected)
    mi_df.to_csv(ARTIFACT_DIR / "mi_scores.csv", index=False)

    print(f"\nTop {K_FEATURES} features by mutual information:")
    for i, (f, s) in enumerate(zip(mi_df.feature[:K_FEATURES],
                                   mi_df.mutual_info[:K_FEATURES]), 1):
        print(f"  {i:>2}. {f:<32} MI = {s:.4f}")

    with open(ARTIFACT_DIR / "selected_features.json", "w") as fh:
        json.dump(selected, fh, indent=2)

    # --- 3. Scaler, fitted on train days only --------------------------
    print(f"\nFitting StandardScaler on up to {SCALER_SAMPLE:,} train rows...")
    scaler = StandardScaler()
    seen = 0
    for f in files:
        df = pd.read_parquet(f, columns=selected)
        arr = df.to_numpy(dtype=np.float32)
        # Clip extreme outliers before fitting: a single 1e9 byte-rate
        # otherwise sets the std and squashes every real value to ~0.
        arr = np.clip(arr, np.percentile(arr, 0.1, axis=0),
                      np.percentile(arr, 99.9, axis=0))
        scaler.partial_fit(arr)
        seen += len(arr)
        del df, arr
        if seen >= SCALER_SAMPLE:
            break
    joblib.dump(scaler, ARTIFACT_DIR / "scaler.pkl")

    print(f"\nSaved:")
    print(f"  {ARTIFACT_DIR/'selected_features.json'}  ({len(selected)} features)")
    print(f"  {ARTIFACT_DIR/'scaler.pkl'}")
    print(f"  {ARTIFACT_DIR/'mi_scores.csv'}")


if __name__ == "__main__":
    main()
