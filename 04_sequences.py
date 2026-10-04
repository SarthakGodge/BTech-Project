"""
Stage 4 - Sliding-window sequence creation.

Converts each day's (N, 35) scaled feature matrix into (W, T, 35) windows
using np.lib.stride_tricks.sliding_window_view, which is a VIEW - it costs
no memory until the copy is written to disk. A naive Python loop over 13M
rows takes hours; this takes minutes.

Window labels use attack-priority voting: label = max(labels in window).
If any flow in the window is an Attack, the window is an Attack. This is
the security-conservative choice - missing an attack costs far more than
a false alarm.

Windows never cross a day boundary, so no sequence contains a discontinuity.

Output: one memmapped .npy per split in SEQ_DIR.

Run:  python 04_sequences.py
"""
import json
import numpy as np
import pandas as pd
import joblib

from config import (PARQUET_DIR, SEQ_DIR, ARTIFACT_DIR, DAY_SPLITS,
                    WINDOW, STRIDE, WINDOW_DTYPE, CLASS_NAMES)


def load_day(path, features, scaler):
    df = pd.read_parquet(path, columns=features + ["__y"])
    X = df[features].to_numpy(dtype=np.float32)
    y = df["__y"].to_numpy(dtype=np.int8)
    del df
    X = scaler.transform(X).astype(np.float32)
    # Guard against any residual non-finite value reaching the model.
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y


def window_day(X, y):
    n = len(X)
    if n < WINDOW:
        return None, None
    views = np.lib.stride_tricks.sliding_window_view(X, WINDOW, axis=0)
    # sliding_window_view gives (n-T+1, F, T); we want (n-T+1, T, F)
    views = views.transpose(0, 2, 1)[::STRIDE]

    ylab = np.lib.stride_tricks.sliding_window_view(y, WINDOW)[::STRIDE]
    ywin = ylab.max(axis=1).astype(np.int8)   # attack-priority voting
    return views, ywin


def build_split(split_name, features, scaler):
    days = [d for d, s in DAY_SPLITS.items()
            if s == split_name and (PARQUET_DIR / f"{d}.parquet").exists()]
    if not days:
        print(f"  no days for split '{split_name}', skipping")
        return

    # Pass 1: count windows so we can allocate the memmap exactly.
    counts = []
    for d in days:
        n = len(pd.read_parquet(PARQUET_DIR / f"{d}.parquet", columns=["__y"]))
        counts.append(max(0, (n - WINDOW) // STRIDE + 1))
    total = sum(counts)
    F = len(features)

    gb = total * WINDOW * F * 4 / 1e9
    print(f"\n[{split_name}] {len(days)} days -> {total:,} windows "
          f"({WINDOW}x{F}) = {gb:.2f} GB on disk")

    Xpath = SEQ_DIR / f"X_{split_name}.npy"
    ypath = SEQ_DIR / f"y_{split_name}.npy"

    Xmm = np.lib.format.open_memmap(Xpath, mode="w+", dtype=WINDOW_DTYPE,
                                    shape=(total, WINDOW, F))
    ymm = np.lib.format.open_memmap(ypath, mode="w+", dtype="int8",
                                    shape=(total,))

    # Pass 2: fill.
    off = 0
    for d, expected in zip(days, counts):
        X, y = load_day(PARQUET_DIR / f"{d}.parquet", features, scaler)
        Xw, yw = window_day(X, y)
        del X, y
        if Xw is None:
            continue
        k = len(Xw)
        Xmm[off:off + k] = Xw
        ymm[off:off + k] = yw
        off += k
        print(f"    {d}: +{k:,} windows "
              f"(labels {dict(zip(*np.unique(yw, return_counts=True)))})")
        del Xw, yw

    Xmm.flush(); ymm.flush()

    dist = np.bincount(np.asarray(ymm), minlength=3)
    print(f"  {split_name} window distribution: " + " | ".join(
        f"{CLASS_NAMES[i]}: {dist[i]:,} ({100*dist[i]/max(dist.sum(),1):.2f}%)"
        for i in range(3)))


def main():
    with open(ARTIFACT_DIR / "selected_features.json") as fh:
        features = json.load(fh)
    scaler = joblib.load(ARTIFACT_DIR / "scaler.pkl")

    print(f"T={WINDOW}, stride={STRIDE}, features={len(features)}")
    for split in ("train", "val", "test"):
        build_split(split, features, scaler)

    print(f"\nSequences written to {SEQ_DIR}")


if __name__ == "__main__":
    main()
