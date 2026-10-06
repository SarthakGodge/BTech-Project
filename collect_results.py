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
            "Pre_AP": rep.get("pre_attack_ap", np.nan),
            "Infil_rec": (rep.get("pre_attack_by_subtype", {}).get("infiltration", {}) or {}).get("recall") or np.nan,
            "PortScan_rec": (rep.get("pre_attack_by_subtype", {}).get("portscan", {}) or {}).get("recall") or np.nan,
        })
if not rows:
    raise SystemExit("no report*.json found in artifacts/")
df = pd.DataFrame(rows)
metrics = ["acc", "macro_f1", "NoAtk_rec", "Pre_prec", "Pre_rec", "Pre_f1", "Pre_AP",
           "Infil_rec", "PortScan_rec", "Atk_rec", "Atk_f1"]
g = df.groupby(["run", "split"])
out = g[metrics].mean().round(4).astype(str).replace("nan", "-")
std = g[metrics].std(ddof=1).round(4)
n = g.size()
for m in metrics:
    out[m] = [f"{a}" + (f" ± {s:.4f}" if n[i] > 1 and not np.isnan(s) and a != "-" else "")
              for (i, a), s in zip(out[m].items(), std[m])]
out.insert(0, "n_seeds", n)
(C.ARTIFACT_DIR / "results").mkdir(exist_ok=True)
out.to_csv(C.ARTIFACT_DIR / "results" / "summary.csv")
print(out.reset_index().to_markdown(index=False) if hasattr(out, "to_markdown") else out)


# ---------------------------------------------------------------- paper tables
def md(frame):
    t = frame.to_markdown(index=False) if hasattr(frame, "to_markdown") else frame.to_string(index=False)
    return re.sub(r"\bnan\b", "-", t)


flat = out.reset_index()
means = g[metrics].mean()
cols = ["run", "split", "n_seeds", "macro_f1", "Pre_f1", "Pre_AP", "Infil_rec", "PortScan_rec", "Atk_f1", "acc"]
all_runs = list(flat["run"].unique())
sections = [
    ("Table 1 - pooled 2017+2018, test blocks (block split)",
     [r for r in ["lgbm", "gbm", "none", "weights", "sampler", "smote", "attn", "ens", "ensattn"] if r in all_runs]),
    ("Table 2 - cross-dataset (PortScan counted as Attack; Pre-Attack = Infiltration only)",
     [r for r in all_runs if r.startswith("train20") or "to20" in r]),
    ("Table 3 - leakage experiment: random window split vs block split",
     [r for r in ["rand_weights", "weights", "rand_gbm", "gbm"] if r in all_runs]),
]
lines = ["# Results tables (mean +/- std over seeds; n_seeds = 1 means no spread is available)", "",
         "Compare models on Pre_AP and macro_f1 first. `test` = argmax; `test_tuned` = class scales tuned on "
         "validation only. A gap smaller than the +/- spread is not evidence of a better model.", ""]
for title, runs in sections:
    sub = flat[flat["run"].isin(runs)]
    if len(sub):
        lines += [f"## {title}", "", md(sub[cols]), ""]
infl = []
for leaky, honest in (("rand_weights", "weights"), ("rand_gbm", "gbm")):
    if (leaky, "test") in means.index and (honest, "test") in means.index:
        for m in ("macro_f1", "Pre_f1", "Pre_AP"):
            a, b = means.loc[(leaky, "test"), m], means.loc[(honest, "test"), m]
            if not (np.isnan(a) or np.isnan(b)):
                infl.append({"model": honest, "metric": m, "random_split": round(a, 4),
                             "block_split": round(b, 4), "inflation": round(a - b, 4)})
if infl:
    lines += ["## Inflation caused by a random window split (random minus block)", "", md(pd.DataFrame(infl)), ""]
(C.ARTIFACT_DIR / "results" / "summary.md").write_text("\n".join(lines))
print("\nwrote artifacts/results/summary.md")
