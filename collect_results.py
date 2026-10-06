"""
Collect artifacts/report*.json into one table (mean +/- std over seeds).

Tags ending in _s<digit(s)> are grouped, e.g. weights_s1, weights_s2 -> "weights".
  python collect_results.py            # prints markdown, writes artifacts/results/summary.csv
"""
import json
import re
import numpy as np
import pandas as pd
import config as C

rows = []
for p in sorted(C.ARTIFACT_DIR.glob("report*.json")):
    r = json.load(open(p))
    tag = p.stem.replace("report_", "").replace("report", "") or "default"
    group = re.sub(r"_s\d+$", "", tag)
    for split in ("test", "test_tuned"):
        rep = r["reports"].get(split)
        if rep is None:
            continue
        rows.append({
            "run": group, "split": split, "seed_tag": tag,
            "acc": rep["accuracy"], "macro_f1": rep["macro avg"]["f1-score"],
            "NoAtk_rec": rep["No Attack"]["recall"],
            "Pre_prec": rep["Pre-Attack"]["precision"], "Pre_rec": rep["Pre-Attack"]["recall"],
            "Pre_f1": rep["Pre-Attack"]["f1-score"],
            "Atk_rec": rep["Attack"]["recall"], "Atk_f1": rep["Attack"]["f1-score"],
        })
if not rows:
    raise SystemExit("no report*.json found in artifacts/")
df = pd.DataFrame(rows)
metrics = ["acc", "macro_f1", "NoAtk_rec", "Pre_prec", "Pre_rec", "Pre_f1", "Atk_rec", "Atk_f1"]
g = df.groupby(["run", "split"])
out = g[metrics].mean().round(4).astype(str)
std = g[metrics].std(ddof=1).round(4)
n = g.size()
for m in metrics:
    out[m] = [f"{a}" + (f" ± {s:.4f}" if n[i] > 1 and not np.isnan(s) else "")
              for (i, a), s in zip(out[m].items(), std[m])]
out.insert(0, "n_seeds", n)
(C.ARTIFACT_DIR / "results").mkdir(exist_ok=True)
out.to_csv(C.ARTIFACT_DIR / "results" / "summary.csv")
print(out.reset_index().to_markdown(index=False) if hasattr(out, "to_markdown") else out)
