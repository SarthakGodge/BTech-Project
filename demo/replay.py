#!/usr/bin/env python3
"""
Live-traffic demo: replay held-out flows through the detector, one flow at a time.

This is what the system would do on a real interface, with CICFlowMeter replaced by
a replay of flows the model has never seen. Each flow is appended to a rolling buffer
of the last 20; once the buffer is full, every new flow produces a decision.

Two modes:
  --local            load the model directly (no server needed)   <- easiest for a demo
  --api URL          POST each flow to a running serve/app.py     <- shows the real path

Examples
--------
  # simplest: replay a contiguous stretch of test traffic that contains an attack
  python demo/replay.py --local --scenario escalation

  # the real serving path (start the server first):
  #   uvicorn serve.app:app --port 8000
  python demo/replay.py --api http://localhost:8000 --scenario escalation

  # which test blocks contain what
  python demo/replay.py --list-blocks

  # replay one real block exactly as it happened, at 5 flows per second
  python demo/replay.py --local --scenario contiguous --block 137 --rate 5
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config as C  # noqa: E402

CLASS_NAMES = ["No Attack", "Pre-Attack", "Attack"]
SUB_NAMES = {0: "benign", 1: "infiltration", 2: "portscan", 3: "other attack"}
ACTION = {0: "log", 1: "ALERT", 2: "BLOCK"}

# ----------------------------------------------------------------- terminal colour
def _supports_colour():
    return sys.stdout.isatty()


class Col:
    def __init__(self, on):
        self.on = on

    def __call__(self, s, code):
        return f"\033[{code}m{s}\033[0m" if self.on else s

    def green(self, s):  return self(s, "32")
    def yellow(self, s): return self(s, "33")
    def red(self, s):    return self(s, "31")
    def dim(self, s):    return self(s, "2")
    def bold(self, s):   return self(s, "1")


# ----------------------------------------------------------------- data loading
def load_test_rows(parquet_dir, feats):
    """Every TEST-split row from every parquet file, in file + time order."""
    files = sorted(Path(parquet_dir).glob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet files in {parquet_dir} - run 01_prepare.py first")
    keep = list(feats) + ["__y", "__sub", "__blk", "__split"]
    parts = []
    for f in files:
        df = pd.read_parquet(f, columns=keep)
        df = df[df["__split"] == 2]
        if len(df):
            df = df.assign(__file=f.stem)
            parts.append(df)
    if not parts:
        raise SystemExit("no test rows found")
    return pd.concat(parts, ignore_index=True)


def pick_block(df, want_sub=None):
    """A test block that contains the requested sub-type, preferring a mixed one."""
    g = df.groupby(["__file", "__blk"])
    best, best_score = None, -1
    for key, blk in g:
        y = blk["__y"].to_numpy()
        sub = blk["__sub"].to_numpy()
        if want_sub is not None and not (sub == want_sub).any():
            continue
        n_pre, n_atk = int((y == 1).sum()), int((y == 2).sum())
        if n_pre == 0 and want_sub is None:
            continue
        score = min(n_pre, 200) + min(n_atk, 100)      # prefer blocks with both
        if score > best_score:
            best, best_score = key, score
    return best


def build_scenario(df, args):
    """Return (frame, title). The frame is replayed in order."""
    if args.scenario == "contiguous":
        if args.block is not None:
            sel = df[df["__blk"] == args.block]
            if sel.empty:
                raise SystemExit(f"block {args.block} has no test rows")
            key = (sel["__file"].iloc[0], args.block)
            sel = sel[sel["__file"] == key[0]]
        else:
            key = pick_block(df)
            if key is None:
                raise SystemExit("no test block contains Pre-Attack traffic")
            sel = df[(df["__file"] == key[0]) & (df["__blk"] == key[1])]
        return sel.head(args.n), f"block {key[1]} of {key[0]} (real order, untouched)"

    if args.scenario == "escalation":
        benign = df[df["__sub"] == 0]
        recon = df[df["__sub"].isin([1, 2])]
        attack = df[df["__sub"] == 3]
        if recon.empty or attack.empty:
            raise SystemExit("test split lacks reconnaissance or attack rows")
        n = args.n
        a, b, c = int(n * 0.45), int(n * 0.30), n - int(n * 0.45) - int(n * 0.30)
        sel = pd.concat([benign.head(a), recon.head(b), attack.head(c)], ignore_index=True)
        return sel, f"constructed: {a} benign -> {b} reconnaissance -> {c} attack"

    sub = {"benign": 0, "infiltration": 1, "portscan": 2, "attack": 3}[args.scenario]
    sel = df[df["__sub"] == sub]
    if sel.empty:
        raise SystemExit(f"no '{args.scenario}' rows in the test split")
    head = df[df["__sub"] == 0].head(25)               # warm the buffer with benign first
    return pd.concat([head, sel.head(args.n)], ignore_index=True), \
        f"25 benign, then {args.scenario} only"


# ----------------------------------------------------------------- detectors
class LocalDetector:
    """Loads the model straight from artifacts/ - no server required."""

    def __init__(self, tag, window):
        import tensorflow as tf
        import joblib
        sys.path.insert(0, str(ROOT))
        import common  # noqa: F401  (scaler.pkl needs signed_log1p to unpickle)
        self.tf = tf
        art = C.ARTIFACT_DIR
        sm = art / f"saved_model_{tag}" if (art / f"saved_model_{tag}").exists() else art / "saved_model"
        if not sm.exists():
            raise SystemExit(f"{sm} not found - train a model first, or pass --tag")
        self.model = tf.saved_model.load(str(sm))
        self.infer = self.model.signatures["serving_default"]
        self.out_key = list(self.infer.structured_outputs.keys())[0]
        self.scaler = joblib.load(art / "scaler.pkl")
        th = art / f"thresholds_{tag}.json"
        self.scale = (np.array(json.loads(th.read_text())["class_scale"], np.float32)
                      if th.exists() else np.ones(3, np.float32))
        self.window = window
        self.name = f"local model '{sm.name}'"

    def predict(self, buf):
        t0 = time.perf_counter()
        x = self.scaler.transform(np.asarray(buf, dtype=np.float32))
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)[None, ...].astype(np.float32)
        p = self.infer(self.tf.constant(x))[self.out_key].numpy()[0]
        cls = int((p * self.scale).argmax())
        return p, cls, (time.perf_counter() - t0) * 1000


class ApiDetector:
    """Posts each flow to serve/app.py, which keeps the buffer itself."""

    def __init__(self, url):
        import requests
        self.requests = requests
        self.url = url.rstrip("/")
        h = requests.get(f"{self.url}/health", timeout=10).json()
        self.window = int(h["window"])
        self.feature_order = requests.get(f"{self.url}/features", timeout=10).json()["order"]
        try:                                   # start from an empty buffer every run
            requests.delete(f"{self.url}/flow/demo", timeout=10)
        except Exception:
            pass
        self.name = f"server {self.url} (window {self.window})"

    def send(self, row):
        r = self.requests.post(f"{self.url}/flow",
                               json={"source": "demo", "features": [float(v) for v in row]},
                               timeout=30)
        r.raise_for_status()
        return r.json()


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local", action="store_true", help="run the model in this process")
    ap.add_argument("--api", default=None, help="base URL of a running serve/app.py")
    ap.add_argument("--tag", default="weights_s1", help="which trained model (local mode)")
    ap.add_argument("--scenario", default="escalation",
                    choices=["escalation", "contiguous", "benign", "infiltration",
                             "portscan", "attack"])
    ap.add_argument("--block", type=int, default=None, help="block id for --scenario contiguous")
    ap.add_argument("--n", type=int, default=120, help="flows to replay")
    ap.add_argument("--rate", type=float, default=0.0, help="flows per second (0 = as fast as possible)")
    ap.add_argument("--quiet", action="store_true", help="only print alerts and the summary")
    ap.add_argument("--no-scale", action="store_true",
                    help="decide by raw argmax, ignoring the validation-tuned class scales")
    ap.add_argument("--list-blocks", action="store_true", help="show test blocks and their content")
    args = ap.parse_args()

    c = Col(_supports_colour())
    feats = json.loads((C.ARTIFACT_DIR / "selected_features.json").read_text())
    df = load_test_rows(C.PARQUET_DIR, feats)

    if args.list_blocks:
        g = df.groupby(["__file", "__blk"]).agg(
            flows=("__y", "size"),
            benign=("__y", lambda s: int((s == 0).sum())),
            pre=("__y", lambda s: int((s == 1).sum())),
            attack=("__y", lambda s: int((s == 2).sum())))
        g = g[(g["pre"] > 0) | (g["attack"] > 0)].sort_values("pre", ascending=False)
        print(g.head(30).to_string())
        print(f"\n{len(g)} test blocks contain Pre-Attack or Attack traffic.")
        return 0

    if not args.local and not args.api:
        args.local = True
    det = ApiDetector(args.api) if args.api else LocalDetector(args.tag, C.WINDOW)
    if args.no_scale and not args.api:
        det.scale = np.ones(3, np.float32)

    sel, title = build_scenario(df, args)
    X = sel[feats].to_numpy(dtype=np.float32)
    y = sel["__y"].to_numpy()
    sub = sel["__sub"].to_numpy()
    window = det.window

    print(c.bold("\n  Live replay of held-out network flows"))
    print(f"  detector : {det.name}")
    print(f"  scenario : {title}")
    print(f"  flows    : {len(X)}   window: {window} flows   "
          f"(no decision until {window} flows have arrived)")
    sc = getattr(det, "scale", None)
    if sc is not None:
        print(f"  scales   : {np.round(sc, 3).tolist()}  "
              f"(tuned on validation; the decision is argmax of probability x scale,")
        print(f"             so a class can lead on probability and still not be chosen)")
    print()
    print(c.dim("  flow   true label     P(No Attack) P(Pre-Attack) P(Attack)   decision      action"))
    print(c.dim("  " + "-" * 86))

    buf, lat, n_pred, n_right = [], [], 0, 0
    alerts = {0: 0, 1: 0, 2: 0}
    first_alert_at = {}          # sub-type -> flows after it first appeared
    first_seen_at = {}
    t_start = time.time()

    for i in range(len(X)):
        if sub[i] in (1, 2, 3) and sub[i] not in first_seen_at:
            first_seen_at[int(sub[i])] = i

        if args.api:
            res = det.send(X[i])
            if res.get("status") == "buffering":
                if not args.quiet:
                    print(c.dim(f"  {i:4d}   {SUB_NAMES[int(sub[i])]:<13}  buffering "
                                f"({res['have']}/{res['need']})"))
                continue
            p = np.array([res["probabilities"][n] for n in CLASS_NAMES], dtype=float)
            cls, ms = int(res["class_id"]), float(res["latency_ms"])
        else:
            buf.append(X[i])
            if len(buf) > window:
                buf.pop(0)
            if len(buf) < window:
                if not args.quiet:
                    print(c.dim(f"  {i:4d}   {SUB_NAMES[int(sub[i])]:<13}  buffering "
                                f"({len(buf)}/{window})"))
                continue
            p, cls, ms = det.predict(buf)

        lat.append(ms)
        n_pred += 1
        true = int(y[i])
        ok = cls == true
        n_right += ok
        alerts[cls] += 1
        if cls in (1, 2) and int(sub[i]) in first_seen_at and int(sub[i]) not in first_alert_at:
            if cls == true:
                first_alert_at[int(sub[i])] = i - first_seen_at[int(sub[i])]

        mark = c.green("correct") if ok else c.red("wrong  ")
        act = ACTION[cls]
        act_s = c.green(act) if cls == 0 else (c.yellow(act) if cls == 1 else c.red(act))
        line = (f"  {i:4d}   {SUB_NAMES[int(sub[i])]:<13} "
                f"   {p[0]:6.3f}       {p[1]:6.3f}      {p[2]:6.3f}    "
                f"{CLASS_NAMES[cls]:<12} {act_s:<16} {mark}")
        if not args.quiet or cls != 0:
            print(line)

        if args.rate > 0:
            time.sleep(1.0 / args.rate)

    # ------------------------------------------------------------- summary
    el = time.time() - t_start
    print(c.dim("  " + "-" * 86))
    print(c.bold("\n  Summary"))
    if n_pred == 0:
        print("  no decisions were made - replay fewer than the window size of flows")
        return 0
    print(f"  decisions            : {n_pred}  ({n_right} correct, {100*n_right/n_pred:.1f}%)")
    print(f"  raised as Pre-Attack : {alerts[1]}")
    print(f"  raised as Attack     : {alerts[2]}")
    print(f"  logged as normal     : {alerts[0]}")
    warm = lat[1:] if len(lat) > 1 else lat          # first call includes graph warm-up
    print(f"  model latency        : {np.median(warm):.1f} ms median, "
          f"{np.percentile(warm, 95):.1f} ms p95, {lat[0]:.0f} ms first call (warm-up)")
    print(f"  wall clock           : {el:.1f} s for {len(X)} flows")
    for code, name in SUB_NAMES.items():
        if code in first_seen_at:
            d = first_alert_at.get(code)
            if d is None:
                print(f"  {name:<13} : never correctly flagged")
            else:
                print(f"  {name:<13} : first correct detection {d} flows after it began")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
