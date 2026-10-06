"""
Checks that run_all.sh performs so a 5-hour job cannot fail late for a reason
that was knowable at the start.

  python preflight.py env          before anything: packages, GPU, disk, BOTH datasets
  python preflight.py prepared     after 01_prepare.py: both datasets present, classes sane
  python preflight.py sequences    after 04_sequences.py: every class in train/val/test

Exit code 0 = fine (warnings allowed), 1 = stop now. Messages say how to fix.
"""
import importlib
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ALLOW_CPU = os.environ.get("ALLOW_CPU") == "1"
ALLOW_SINGLE = os.environ.get("ALLOW_SINGLE_DATASET") == "1"
MIN_FREE_GB = float(os.environ.get("MIN_FREE_GB", 25))
errors, warns = [], []


def err(m): errors.append(m); print(f"  [FAIL] {m}")
def warn(m): warns.append(m); print(f"  [warn] {m}")
def ok(m): print(f"  [ ok ] {m}")


def finish():
    print()
    if errors:
        print(f"PREFLIGHT FAILED ({len(errors)} problem(s)). Nothing was started.")
        sys.exit(1)
    print(f"preflight passed ({len(warns)} warning(s)).")
    sys.exit(0)


def check_env():
    print("== packages")
    for m in ["numpy", "pandas", "sklearn", "pyarrow", "joblib", "lightgbm", "imblearn",
              "matplotlib", "seaborn", "tabulate", "tensorflow"]:
        try:
            importlib.import_module(m); ok(m)
        except Exception as e:
            err(f"cannot import {m} ({type(e).__name__}). Fix: pip install -r requirements.txt")

    print("== GPU")
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            ok(f"{len(gpus)} GPU visible to TensorFlow")
            try:   # really run work on it (catches broken CUDA/cuDNN installs now, not hours later)
                import numpy as _np
                with tf.device("/GPU:0"):
                    a = tf.random.normal((512, 512))
                    _ = float(tf.reduce_sum(tf.matmul(a, a)))
                    lstm = tf.keras.layers.LSTM(8)
                    _ = lstm(tf.random.normal((4, 20, 6))).numpy()
                    conv = tf.keras.layers.Conv1D(4, 3)
                    _ = conv(tf.random.normal((4, 20, 6))).numpy()
                ok("GPU self-test passed (matmul, Conv1D, LSTM ran on the GPU)")
            except Exception as e:
                err(f"GPU is visible but a test computation failed ({type(e).__name__}: {str(e)[:200]}). "
                    "Usually a CUDA/cuDNN mismatch: recreate the venv with 'pip install tensorflow[and-cuda]'.")
        elif ALLOW_CPU:
            warn("no GPU, continuing because ALLOW_CPU=1 (training will be slow)")
        else:
            err("TensorFlow sees NO GPU, so training would take days. Fix: reinstall "
                "'tensorflow[and-cuda]' in a fresh venv. (Override with ALLOW_CPU=1.)")
    except Exception:
        pass  # already reported above

    import config as C
    print("== disk / memory")
    root = Path(C.DATA_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free / 2**30
    (ok if free >= MIN_FREE_GB * 2 else warn if free >= MIN_FREE_GB else err)(
        f"{free:.0f} GB free under {root} (need >= {MIN_FREE_GB:.0f}, comfortable >= {2*MIN_FREE_GB:.0f})")
    try:
        import psutil  # optional
        ok(f"{psutil.virtual_memory().total / 2**30:.0f} GB RAM")
    except Exception:
        try:
            kb = int(next(l for l in open("/proc/meminfo") if l.startswith("MemTotal")).split()[1])
            ok(f"{kb / 2**20:.0f} GB RAM")
        except Exception:
            pass

    print("== datasets (BOTH are required)")
    from common import LOOKUP, CANON, norm
    import pandas as pd
    have = {}
    for ds, folder in C.DATASET_DIRS.items():
        csvs = sorted(Path(folder).glob("*.csv")) if Path(folder).exists() else []
        if not csvs:
            msg = (f"{ds}: no CSV files in {folder}")
            if ds in os.environ.get("IDS_WILL_DOWNLOAD", "").split(","):
                warn(msg + " (will be downloaded by the runner before training starts)")
                have[ds] = 0
            elif ALLOW_SINGLE:
                warn(msg + " (continuing: ALLOW_SINGLE_DATASET=1; cross-dataset runs will be skipped)")
            else:
                err(msg + ". Put the CSVs there (2018: bash 00_download.sh <dir>; 2017: copy with scp), "
                          "or set ALLOW_SINGLE_DATASET=1 to run on one dataset.")
            continue
        used = [p for p in csvs if not (ds == "cic2018" and "20-02" in p.name and not C.INCLUDE_0220)]
        gb = sum(p.stat().st_size for p in used) / 2**30
        no_ts, bad = 0, 0
        for p in used:
            try:
                cols = pd.read_csv(p, nrows=0, encoding="latin1").columns
            except Exception as e:
                err(f"{ds}: cannot read {p.name} ({e})"); bad += 1; continue
            canon = {LOOKUP[norm(c)] for c in cols if norm(c) in LOOKUP}
            if "Label" not in canon:
                err(f"{ds}: {p.name} has no Label column"); bad += 1
            n_feat = len([c for c in CANON if c in canon])
            if n_feat < 60:
                err(f"{ds}: {p.name} maps only {n_feat}/{len(CANON)} features - column names differ "
                    f"(add them to _PAIRS in common.py)")
            if "Timestamp" not in canon:
                no_ts += 1
        if not bad:
            ok(f"{ds}: {len(used)} CSV files, {gb:.1f} GB")
            have[ds] = len(used)
        if no_ts:
            warn(f"{ds}: {no_ts}/{len(used)} CSVs have no Timestamp column - windows will follow FILE order "
                 f"(only fine if the file is already in time order)")
    if len(have) < 2 and not ALLOW_SINGLE and not errors:
        err("only one dataset usable")
    finish()


def load_meta():
    import pandas as pd
    import config as C
    files = sorted(C.PARQUET_DIR.glob("*.parquet"))
    rows = []
    for f in files:
        d = pd.read_parquet(f, columns=["__y", "__sub", "__split"])
        rows.append((f.name.split("__")[0], f.name, d))
    return rows


def check_prepared():
    import config as C
    rows = load_meta()
    print("== prepared data")
    if not rows:
        err("no parquet files were produced"); finish()
    per = {}
    for ds, name, d in rows:
        y = d["__y"].to_numpy()
        cnt = np.bincount(y, minlength=3)
        per.setdefault(ds, np.zeros(3, dtype=np.int64))
        per[ds] += cnt
        ok(f"{name}: {len(d):,} flows | No Attack {cnt[0]:,} | Pre-Attack {cnt[1]:,} | Attack {cnt[2]:,}")
    for ds in C.DATASET_DIRS:
        if ds not in per:
            (warn if ALLOW_SINGLE else err)(f"dataset {ds} produced no usable flows")
    tot = sum(per.values())
    for i, n in enumerate(C.CLASS_NAMES):
        if tot[i] == 0:
            err(f"class '{n}' has ZERO flows overall - check LABEL_RULES in config.py")
    for ds, c in per.items():
        if c[1] < 500:
            warn(f"{ds}: only {c[1]} Pre-Attack flows - Pre-Attack scores for this dataset will be unreliable")
    import json
    if not (C.ARTIFACT_DIR / "selected_features.json").exists() or not (C.ARTIFACT_DIR / "scaler.pkl").exists():
        err("selected_features.json / scaler.pkl missing")
    finish()


def check_sequences():
    import config as C
    print("== window files")
    for s in ("train", "val", "test"):
        for pre in ("X", "y", "d", "s"):
            if not (C.SEQ_DIR / f"{pre}_{s}.npy").exists():
                err(f"missing {pre}_{s}.npy")
    if errors:
        finish()
    for s in ("train", "val", "test"):
        y = np.load(C.SEQ_DIR / f"y_{s}.npy")
        cnt = np.bincount(y, minlength=3)
        ok(f"{s}: {len(y):,} windows | " + " | ".join(f"{C.CLASS_NAMES[i]} {cnt[i]:,}" for i in range(3)))
        for i in range(3):
            if cnt[i] == 0:
                err(f"{s} has no '{C.CLASS_NAMES[i]}' windows")
            elif cnt[i] < 200 and s != "train":
                warn(f"{s} has only {cnt[i]} '{C.CLASS_NAMES[i]}' windows - its scores will be noisy")
    finish()


if __name__ == "__main__":
    {"env": check_env, "prepared": check_prepared, "sequences": check_sequences}[sys.argv[1]]()
