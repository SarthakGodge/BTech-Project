"""
Stage 9 - blend saved models (probability averaging) on the SAME windows.

  python 09_ensemble.py --members weights_s1 gbm_s1 --tag ens_s1

Needs val_probs_<tag>.npy / test_probs_<tag>.npy from 05_train.py and
08_window_gbm.py. The blend weight and the class scales are chosen on VAL only.
With 2 members the weight is searched on a grid; with more, plain averaging.
"""
import argparse
import itertools
import json
import numpy as np
from sklearn.metrics import f1_score

import config as C
from evalutil import tune_class_scales, make_report


def load(tag):
    d = C.ARTIFACT_DIR
    return (np.load(d / f"val_probs_{tag}.npy"), np.load(d / f"val_idx_{tag}.npy"),
            np.load(d / f"test_probs_{tag}.npy"), np.load(d / f"test_idx_{tag}.npy"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seq", default="")
    args = ap.parse_args()
    seq = C.SEQ_DIR / args.seq if args.seq else C.SEQ_DIR

    mem = [load(t) for t in args.members]
    for m in mem[1:]:
        assert np.array_equal(m[1], mem[0][1]) and np.array_equal(m[3], mem[0][3]), \
            "members were scored on different windows - rerun them with the same data"
    va_i, te_i = mem[0][1], mem[0][3]
    yv = np.load(seq / "y_val.npy")[va_i]
    yt = np.load(seq / "y_test.npy")[te_i]
    dte = np.load(seq / "d_test.npy")[te_i] if (seq / "d_test.npy").exists() else None
    sub = np.load(seq / "s_test.npy")[te_i] if (seq / "s_test.npy").exists() else None

    best_w, best_f = np.ones(len(mem)) / len(mem), -1
    if len(mem) == 2:
        for a in np.linspace(0, 1, 11):
            f = f1_score(yv, (a * mem[0][0] + (1 - a) * mem[1][0]).argmax(1), average="macro")
            if f > best_f + 1e-9:
                best_f, best_w = f, np.array([a, 1 - a])
    print("blend weights (chosen on val):", dict(zip(args.members, best_w.round(2).tolist())))
    pv = sum(w * m[0] for w, m in zip(best_w, mem))
    pt = sum(w * m[2] for w, m in zip(best_w, mem))

    report = {}
    make_report(yv, pv.argmax(1), "val", report, proba=pv)
    make_report(yt, pt.argmax(1), "test", report, proba=pt, ds_ids=dte, sub=sub)
    sc, f = tune_class_scales(pv, yv)
    print(f"\nclass scales tuned on val: {sc.tolist()} (val macro-F1 {f:.4f})")
    make_report(yt, (pt * sc).argmax(1), "test_tuned", report, proba=pt * sc, ds_ids=dte, sub=sub)
    json.dump({"args": vars(args), "weights": best_w.tolist(), "reports": report},
              open(C.ARTIFACT_DIR / f"report_{args.tag}.json", "w"), indent=2)
    print(f"saved report_{args.tag}.json")


if __name__ == "__main__":
    main()
