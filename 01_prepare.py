"""
Stage 1 (replaces 01_ingest / 02_labels / 03_features).

Phase A  per CSV: harmonise columns -> clean -> 3-class labels -> time-sort
         -> cut into blocks -> stratified block->split assignment -> parquet.
Phase B  using TRAIN rows only: variance + correlation filter + mutual
         information -> selected_features.json, then fit
         signed-log1p + StandardScaler -> scaler.pkl.

Parquet columns: <canonical features>, __y (0/1/2), __blk (block id),
                 __split (0 train / 1 val / 2 test)

Run: python 01_prepare.py
"""
import json
import zlib
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_selection import mutual_info_classif
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

import config as C
from common import CANON, LOOKUP, norm, signed_log1p

NA = ["Infinity", "-Infinity", "inf", "-inf", "NaN", "nan", ""]


# ---------------------------------------------------------------- Phase A
def read_csv(path):
    df = pd.read_csv(path, low_memory=False, na_values=NA, encoding="latin1",
                     usecols=lambda c: norm(c) in LOOKUP)
    new, seen = [], set()
    for c in df.columns:                     # rename + drop duplicate targets
        canon = LOOKUP[norm(c)]
        new.append(canon if canon not in seen else f"__dup_{c}")
        seen.add(canon)
    df.columns = new
    return df.drop(columns=[c for c in df.columns if c.startswith("__dup_")])


def map_labels(s):
    t = s.astype(str).str.strip().str.lower()
    y = np.full(len(t), C.LABEL_DEFAULT, dtype=np.int8)
    sub = np.full(len(t), 3, dtype=np.int8)
    assigned = np.zeros(len(t), dtype=bool)
    for pat, cls in C.LABEL_RULES:
        hit = t.str.contains(pat, regex=False).to_numpy() & ~assigned
        y[hit] = cls
        sub[hit] = C.SUBCODES.get(pat, 3)
        assigned |= hit
    return y, sub


def assign_splits(y, blk, rng):
    """Stratify blocks by their most severe label, then split 70/10/20."""
    nb = int(blk.max()) + 1
    key = np.zeros(nb, dtype=np.int8)
    np.maximum.at(key, blk, y)
    code = np.zeros(nb, dtype=np.int8)
    for k in (0, 1, 2):
        ids = np.flatnonzero(key == k)
        rng.shuffle(ids)
        n_tr = int(round(C.SPLIT_FRACS[0] * len(ids)))
        n_va = int(round(C.SPLIT_FRACS[1] * len(ids)))
        if len(ids) >= 3:                      # every severity group reaches val AND test
            n_va = max(n_va, 1)
            n_tr = min(n_tr, len(ids) - n_va - 1)
        code[ids[:n_tr]] = 0
        code[ids[n_tr:n_tr + n_va]] = 1
        code[ids[n_tr + n_va:]] = 2
    return code[blk]


def ingest(path, ds):
    df = read_csv(path)
    if "Label" not in df.columns:
        print(f"  !! {path.name}: no Label column, skipped")
        return None
    df = df[df["Label"].astype(str).str.strip() != "Label"]   # repeated headers
    df = df.dropna(subset=["Label"])
    feats = [c for c in CANON if c in df.columns and c not in C.EXCLUDE_FEATURES]
    missing = [c for c in CANON if c not in df.columns]
    if "Timestamp" in df.columns:
        ts = pd.to_datetime(df["Timestamp"], errors="coerce", dayfirst=True)
        df = df.assign(__ts=ts).dropna(subset=["__ts"]).sort_values(
            "__ts", kind="mergesort")
        has_ts = True
    else:
        has_ts = False
    y, sub = map_labels(df["Label"])
    X = df[feats].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    ok = np.isfinite(X.to_numpy()).all(axis=1)
    X, y, sub = X[ok].reset_index(drop=True), y[ok], sub[ok]
    n_dup = 0
    if C.DEDUP_FLOWS:                       # exact duplicates (features + label)
        h = pd.util.hash_pandas_object(X, index=False).to_numpy()
        h = h ^ (sub.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15))
        keep = ~pd.Series(h).duplicated().to_numpy()
        n_dup = int((~keep).sum())
        X, y, sub = X[keep].reset_index(drop=True), y[keep], sub[keep]
    blk = (np.arange(len(X)) // C.BLOCK_SIZE).astype(np.int32)
    rng = np.random.default_rng(C.SEED + zlib.crc32(path.name.encode()) % 10_000)
    split = assign_splits(y, blk, rng)
    X["__y"], X["__sub"], X["__blk"], X["__split"] = y, sub, blk, split.astype(np.int8)
    out = C.PARQUET_DIR / f"{ds}__{path.stem}.parquet"
    X.to_parquet(out, index=False)
    cnt = np.bincount(y, minlength=3)
    print(f"  {out.name}: {len(X):,} flows | classes {cnt.tolist()} | "
          f"duplicates removed: {n_dup:,} | "
          f"timestamp={'yes' if has_ts else 'NO (row order used!)'} | "
          f"missing features: {len(missing)}")
    return out


def phase_a():
    files = []
    for ds, folder in C.DATASET_DIRS.items():
        csvs = sorted(folder.glob("*.csv")) if folder.exists() else []
        if not csvs:
            print(f"[{ds}] no CSVs in {folder} - skipped")
            continue
        print(f"[{ds}] {len(csvs)} CSV files")
        for p in csvs:
            if not C.INCLUDE_0220 and ds == "cic2018" and "20-02" in p.name:
                print(f"  skip {p.name} (INCLUDE_0220=False)")
                continue
            out = ingest(p, ds)
            if out is not None:
                files.append(out)
    return files


# ---------------------------------------------------------------- Phase B
def phase_b(files):
    import pyarrow.parquet as pq
    cols_per_file = []
    for f in files:
        cols_per_file.append(set(pq.ParquetFile(f).schema.names))
    common = [c for c in CANON
              if all(c in cs for cs in cols_per_file) and c not in C.EXCLUDE_FEATURES]
    print(f"\n{len(common)} features present in every file")

    per_file = max(1, C.MI_SAMPLE_PER_CLASS * 4 // len(files))
    parts = []
    rng = np.random.default_rng(C.SEED)
    for f in files:
        d = pd.read_parquet(f, columns=common + ["__y", "__split"])
        d = d[d["__split"] == 0]
        take = []
        for k in (0, 1, 2):
            ids = np.flatnonzero(d["__y"].to_numpy() == k)
            if len(ids) > per_file:
                ids = rng.choice(ids, per_file, replace=False)
            take.append(ids)
        parts.append(d.iloc[np.concatenate(take)])
    S = pd.concat(parts, ignore_index=True)
    y = S["__y"].to_numpy()
    print("selection sample (train rows only):", np.bincount(y, minlength=3).tolist())

    Xl = signed_log1p(S[common].to_numpy())
    keep = np.flatnonzero(Xl.var(axis=0) > C.VARIANCE_THRESHOLD)
    print(f"variance filter: {len(common)} -> {len(keep)}")

    corr = np.abs(np.corrcoef(Xl[:, keep], rowvar=False))
    corr = np.nan_to_num(corr)
    drop = set()
    for i in range(len(keep)):
        if i in drop:
            continue
        for j in range(i + 1, len(keep)):
            if corr[i, j] > C.CORR_THRESHOLD:
                drop.add(j)
    keep = np.array([k for n, k in enumerate(keep) if n not in drop])
    print(f"correlation filter (>{C.CORR_THRESHOLD}): -> {len(keep)}")

    mi = mutual_info_classif(Xl[:, keep], y, discrete_features=False,
                             n_neighbors=3, random_state=C.SEED)
    order = np.argsort(mi)[::-1]
    names = [common[k] for k in keep]
    pd.DataFrame({"feature": [names[i] for i in order],
                  "mi": mi[order]}).to_csv(C.ARTIFACT_DIR / "mi_scores.csv", index=False)
    selected = [names[i] for i in order[:C.K_FEATURES]]
    # keep canonical column order for reproducibility
    selected = [c for c in CANON if c in selected]
    with open(C.ARTIFACT_DIR / "selected_features.json", "w") as fh:
        json.dump(selected, fh, indent=2)
    print(f"selected {len(selected)} features -> selected_features.json")

    pipe = make_pipeline(FunctionTransformer(signed_log1p, validate=False),
                         StandardScaler())
    pipe.fit(S[selected].to_numpy(dtype=np.float32))
    joblib.dump(pipe, C.ARTIFACT_DIR / "scaler.pkl")
    print("scaler (signed-log1p + StandardScaler, fit on train only) -> scaler.pkl")


if __name__ == "__main__":
    fs = phase_a()
    if not fs:
        raise SystemExit("No data ingested - check C.DATASET_DIRS")
    phase_b(fs)
