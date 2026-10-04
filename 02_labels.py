"""
Stage 2 - Label engineering: 15 raw labels -> {0 No Attack, 1 Pre-Attack, 2 Attack}

Two mechanisms produce the Pre-Attack class:

  (a) Direct mapping. "Infilteration" -> 1. Those scenarios are lateral
      movement and probing, not payload delivery.

  (b) Temporal derivation. Within each day (already time-sorted in stage 1),
      find each contiguous ATTACK EPISODE - a run of consecutive flows
      sharing the same attack label. Walk BACKWARDS from the start of that
      episode and relabel the preceding benign flows as Pre-Attack.

Why episodes and not "group by Src IP" as the design doc says: nine of the
ten CIC-IDS2018 CSVs have no Src IP column at all. Only the 20-02 file
carries it. Episode detection on the time axis achieves the same thing -
it captures the ramp-up window immediately before each attack execution -
and works on every file.

Run:  python 02_labels.py
"""
import numpy as np
import pandas as pd

from config import (PARQUET_DIR, LABEL_MAP, PRE_ATTACK_FRACTION,
                    PRE_ATTACK_MAX_FLOWS, CLASS_NAMES)


def find_episodes(y: np.ndarray):
    """Yield (start, end) index pairs for each contiguous run of class 2."""
    is_attack = (y == 2).astype(np.int8)
    if is_attack.sum() == 0:
        return
    edges = np.diff(np.concatenate([[0], is_attack, [0]]))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    yield from zip(starts, ends)


def derive_pre_attack(y: np.ndarray) -> np.ndarray:
    """Relabel benign flows immediately preceding each attack episode."""
    y = y.copy()
    n_relabelled = 0

    for start, end in find_episodes(y):
        episode_len = end - start
        lookback = min(int(episode_len * PRE_ATTACK_FRACTION),
                       PRE_ATTACK_MAX_FLOWS)
        if lookback < 1:
            continue
        lo = max(0, start - lookback)
        window = y[lo:start]
        # Only convert benign flows - never overwrite a real attack label.
        mask = window == 0
        window[mask] = 1
        n_relabelled += int(mask.sum())

    return y, n_relabelled


def process_day(path):
    df = pd.read_parquet(path, columns=["Timestamp", "Label"])

    y = df["Label"].map(LABEL_MAP)
    unmapped = df.loc[y.isna(), "Label"].unique()
    if len(unmapped):
        print(f"    unmapped labels dropped: {list(unmapped)}")

    keep = y.notna().to_numpy()
    y = y.to_numpy()[keep].astype(np.int8)

    y, n_pre = derive_pre_attack(y)

    # Write labels back alongside the full frame.
    full = pd.read_parquet(path)
    full = full.loc[keep].reset_index(drop=True)
    full["__y"] = y
    full.to_parquet(path, index=False, compression="snappy")

    counts = np.bincount(y, minlength=3)
    total = counts.sum()
    print(f"    {total:,} rows | " + " | ".join(
        f"{CLASS_NAMES[i]}: {counts[i]:,} ({100*counts[i]/total:.2f}%)"
        for i in range(3)))
    print(f"    temporally derived Pre-Attack flows: {n_pre:,}")
    return counts


def main():
    files = sorted(PARQUET_DIR.glob("*.parquet"))
    if not files:
        raise SystemExit("No parquet files. Run 01_ingest.py first.")

    grand = np.zeros(3, dtype=np.int64)
    for path in files:
        print(f"\n[{path.stem}]")
        grand += process_day(path)

    total = grand.sum()
    print("\n" + "=" * 60)
    print("OVERALL CLASS DISTRIBUTION  (put this table in your report)")
    print("=" * 60)
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i} {name:<12} {grand[i]:>12,}  {100*grand[i]/total:6.2f}%")
    print(f"  {'TOTAL':<14} {total:>12,}")
    print(f"\n  Imbalance ratio (majority:minority) = "
          f"{grand.max()/max(grand.min(),1):.1f} : 1")


if __name__ == "__main__":
    main()
