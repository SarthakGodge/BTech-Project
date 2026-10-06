"""
Stage 4 - sliding windows from the block-split parquet files.

* Windows are built INSIDE a block, so none straddles train/val/test.
* Window label = last flow (config.WINDOW_LABEL_MODE).
* Writes X_{split}.npy, y_{split}.npy, d_{split}.npy (dataset id per window)
  and s_{split}.npy (sub-type of the last flow: 1 infiltration, 2 portscan,
  3 other attack, 0 benign) so Pre-Attack can be broken down.

Cross-dataset experiment:
    python 04_sequences.py --holdout cic2017
trains on every other dataset (all their blocks) and tests ONLY on cic2017.
Output goes to SEQ_DIR/holdout_cic2017 (use: 05_train.py --seq holdout_cic2017).

PortScan exists only in 2017, so for fair cross-dataset runs add --portscan-as attack
(folder name gets a '_psattack' suffix).

Leakage experiment (for the paper):
    python 04_sequences.py --split random
builds the SAME windows but assigns each window to train/val/test AT RANDOM, which is
what many published IDS papers do. Overlapping/adjacent windows then land on both sides
of the split, so scores are inflated. Output goes to SEQ_DIR/random_split. Comparing a
model trained there with the block-split model quantifies the inflation.
"""
import argparse
import json
import zlib
import numpy as np
import pandas as pd
import joblib

import config as C

SPLITS = ("train", "val", "test")


def window_block(Xb, yb, sb):
    v = np.lib.stride_tricks.sliding_window_view(Xb, C.WINDOW, axis=0)
    v = v.transpose(0, 2, 1)[::C.STRIDE]                      # (W, T, F)
    yl = np.lib.stride_tricks.sliding_window_view(yb, C.WINDOW)[::C.STRIDE]
    yw = yl[:, -1] if C.WINDOW_LABEL_MODE == "last" else yl.max(axis=1)
    sw = np.lib.stride_tricks.sliding_window_view(sb, C.WINDOW)[::C.STRIDE][:, -1]
    return v, yw.astype(np.int8), sw.astype(np.int8)


def block_runs(blk):
    starts = np.r_[0, np.flatnonzero(np.diff(blk)) + 1]
    ends = np.r_[starts[1:], len(blk)]
    return starts, ends


def effective_split(split, ds, holdout):
    if holdout is None:
        return split
    if ds == holdout:
        return np.full_like(split, 2)
    return np.where(split == 2, 0, split)          # other datasets: test -> train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default=None, help="dataset id used only for testing")
    ap.add_argument("--portscan-as", choices=["pre", "attack"], default="pre",
                    help="'attack' removes PortScan from Pre-Attack (fair cross-dataset runs)")
    ap.add_argument("--split", choices=["block", "random"], default="block",
                    help="'random' = leaky per-window random split (inflation experiment)")
    args = ap.parse_args()
    random_mode = args.split == "random"
    if random_mode and args.holdout:
        raise SystemExit("--split random cannot be combined with --holdout")

    name = f"holdout_{args.holdout}" if args.holdout else ("random_split" if random_mode else "")
    if args.portscan_as == "attack":
        name = (name + "_" if name else "") + "psattack"
    out_dir = C.SEQ_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(C.ARTIFACT_DIR / "selected_features.json") as fh:
        feats = json.load(fh)
    scaler = joblib.load(C.ARTIFACT_DIR / "scaler.pkl")
    F = len(feats)
    files = sorted(C.PARQUET_DIR.glob("*.parquet"))
    ds_names = list(C.DATASET_DIRS)

    counts = {s: 0 for s in SPLITS}
    plans = {}
    for f in files:                                           # pass 1: count
        ds = f.name.split("__")[0]
        m = pd.read_parquet(f, columns=["__blk", "__split"])
        blk = m["__blk"].to_numpy()
        if random_mode:                                       # whole file = one run, random windows
            n_win = max(0, (len(blk) - C.WINDOW) // C.STRIDE + 1)
            rs = np.random.default_rng(C.SEED + zlib.crc32(f.name.encode()) % 10_000)
            assign = rs.choice(3, size=n_win, p=list(C.SPLIT_FRACS)).astype(np.int8)
            plans[f] = (ds, assign, None)
            for k, s in enumerate(SPLITS):
                counts[s] += int((assign == k).sum())
            continue
        sp = effective_split(m["__split"].to_numpy(), ds, args.holdout)
        st, en = block_runs(blk)
        plans[f] = (ds, st, en)
        for s, e in zip(st, en):
            if e - s >= C.WINDOW:
                counts[SPLITS[sp[s]]] += int((e - s - C.WINDOW) // C.STRIDE + 1)
    print("windows per split:", counts)

    mm = {}
    for s in SPLITS:
        mm[s] = (np.lib.format.open_memmap(out_dir / f"X_{s}.npy", "w+", C.WINDOW_DTYPE,
                                           (counts[s], C.WINDOW, F)),
                 np.lib.format.open_memmap(out_dir / f"y_{s}.npy", "w+", "int8", (counts[s],)),
                 np.lib.format.open_memmap(out_dir / f"d_{s}.npy", "w+", "int8", (counts[s],)),
                 np.lib.format.open_memmap(out_dir / f"s_{s}.npy", "w+", "int8", (counts[s],)))
    off = {s: 0 for s in SPLITS}

    for f, (ds, st, en) in plans.items():                     # pass 2: fill
        d = pd.read_parquet(f, columns=feats + ["__y", "__sub", "__split"])
        X = scaler.transform(d[feats].to_numpy(dtype=np.float32)).astype(np.float32)
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        y = d["__y"].to_numpy(dtype=np.int8)
        sub = d["__sub"].to_numpy(dtype=np.int8)
        if args.portscan_as == "attack":
            y = np.where(d["__sub"].to_numpy() == 2, 2, y).astype(np.int8)
        did = ds_names.index(ds) if ds in ds_names else -1
        if random_mode:
            assign = st                                       # per-window split codes
            Xv, yw_all, sw_all = window_block(X, y, sub)      # views/arrays over the whole file
            if C.WINDOW_LABEL_MODE != "last":
                pass                                          # window_block already honours it
            CH = 100_000
            for a in range(0, len(assign), CH):
                asg = assign[a:a + CH]
                Xc = Xv[a:a + CH]
                for k, name in enumerate(SPLITS):
                    sel = np.flatnonzero(asg == k)
                    if len(sel) == 0:
                        continue
                    Xm, ym, dm, sm = mm[name]
                    o, n = off[name], len(sel)
                    Xm[o:o + n] = Xc[sel]
                    ym[o:o + n] = yw_all[a:a + CH][sel]
                    dm[o:o + n] = did
                    sm[o:o + n] = sw_all[a:a + CH][sel]
                    off[name] += n
            print(f"  {f.name} done (random windows)")
            continue
        sp = effective_split(d["__split"].to_numpy(), ds, args.holdout)
        for s, e in zip(st, en):
            if e - s < C.WINDOW:
                continue
            Xw, yw, sw = window_block(X[s:e], y[s:e], sub[s:e])
            name = SPLITS[sp[s]]
            k = len(Xw)
            Xm, ym, dm, sm = mm[name]
            Xm[off[name]:off[name] + k] = Xw
            ym[off[name]:off[name] + k] = yw
            dm[off[name]:off[name] + k] = did
            sm[off[name]:off[name] + k] = sw
            off[name] += k
        print(f"  {f.name} done")

    for s in SPLITS:
        for a in mm[s]:
            a.flush()
        dist = np.bincount(np.asarray(mm[s][1]), minlength=3)
        print(f"[{s}] " + " | ".join(
            f"{C.CLASS_NAMES[i]}: {dist[i]:,} ({100 * dist[i] / max(dist.sum(), 1):.2f}%)"
            for i in range(3)))
    print("dataset ids:", dict(enumerate(ds_names)))


if __name__ == "__main__":
    main()
