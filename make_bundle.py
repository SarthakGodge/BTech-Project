"""
Builds the hand-over package after run_all.sh finishes (or runs out of time):

  results_bundle/RESULTS.md      what ran, what failed, the main tables, settings used
  results_bundle/summary.csv     mean +/- std per run (from collect_results.py)
  results_bundle/figures/        confusion matrices, ROC, PR, training curves
  results_bundle/reports/        per-run report json files
  results_bundle/logs/           one log per step
  results_bundle.zip             all of the above (copy THIS to your laptop)

Only reads files, never trains anything, and never fails the pipeline.
"""
import csv
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

import config as C

ART = C.ARTIFACT_DIR
OUT = Path("results_bundle")
PROG = ART / ".progress"


def read_status():
    p = PROG / "status.tsv"
    rows = []
    if p.exists():
        for line in p.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) >= 3:
                rows.append(parts[:3] + (parts[3:4] or [""]))
    return rows


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    for sub, pat, dest in (("figures", None, "figures"), ("results", None, "results_extra")):
        src = ART / sub
        if src.exists():
            shutil.copytree(src, OUT / dest)
    (OUT / "reports").mkdir()
    for f in ART.glob("report*.json"):
        shutil.copy(f, OUT / "reports" / f.name)
    for f in ART.glob("thresholds*.json"):
        shutil.copy(f, OUT / "reports" / f.name)
    for f in ART.glob("lgbm_importance_*.csv"):
        shutil.copy(f, OUT / "reports" / f.name)
    if (ART / "mi_scores.csv").exists():
        shutil.copy(ART / "mi_scores.csv", OUT / "reports" / "mi_scores.csv")
    if (ART / "selected_features.json").exists():
        shutil.copy(ART / "selected_features.json", OUT / "reports" / "selected_features.json")
    if Path("logs").exists():
        shutil.copytree("logs", OUT / "logs")
    summ = ART / "results" / "summary.csv"
    if summ.exists():
        shutil.copy(summ, OUT / "summary.csv")

    status = read_status()
    done = [r for r in status if r[1] == "done"]
    failed = [r for r in status if r[1] == "FAILED"]
    skipped = [r for r in status if r[1].startswith("skipped")]
    L = [f"# Results package ({dt.datetime.now():%Y-%m-%d %H:%M})", ""]
    L += [f"- steps completed: **{len(done)}**, failed: **{len(failed)}**, skipped: **{len(skipped)}**", ""]
    if failed:
        L += ["## FAILED steps (see logs/<step>.log)", ""]
        L += [f"- `{r[0]}`" for r in failed] + [""]
    if skipped:
        L += ["## Skipped steps", ""] + [f"- `{r[0]}`: {r[1]}" for r in skipped] + [""]

    L += ["## Settings used", ""]
    L += [md_table(["setting", "value"], [
        ("window / stride", f"{C.WINDOW} / {C.STRIDE}"),
        ("block size (flows)", C.BLOCK_SIZE),
        ("split (train/val/test)", C.SPLIT_FRACS),
        ("window label", C.WINDOW_LABEL_MODE),
        ("features selected", len(json.loads((ART / 'selected_features.json').read_text()))
         if (ART / "selected_features.json").exists() else "?"),
        ("max train windows (CNN)", f"{C.MAX_TRAIN_WINDOWS:,}"),
        ("max train windows (GBM)", f"{C.GBM_MAX_TRAIN:,}"),
        ("max epochs / patience", f"{C.EPOCHS} / {C.PATIENCE}"),
        ("max eval windows", f"{C.MAX_EVAL_WINDOWS:,}"),
        ("de-duplication", C.DEDUP_FLOWS),
        ("Pre-Attack definition", "Infiltration + PortScan (LABEL_RULES in config.py)"),
    ]), ""]

    if summ.exists():
        rows = list(csv.DictReader(open(summ)))
        keep = ["run", "split", "n_seeds", "macro_f1", "Pre_f1", "Pre_AP", "Infil_rec", "PortScan_rec", "Atk_f1", "acc"]
        L += ["## Main results (test blocks; mean ± std over the seeds that finished)", "",
              "`test` = argmax of the model; `test_tuned` = class scales tuned on validation only. "
              "Compare models on `Pre_AP` and `macro_f1` first. A difference smaller than the ± spread "
              "(or with n_seeds = 1) is not evidence of a better model.", "",
              md_table(keep, [[r.get(k, "") for k in keep] for r in rows]), ""]
    else:
        L += ["## No summary table was produced (collect_results.py found no reports)", ""]

    L += ["## Step log", "", md_table(["step", "status", "seconds", "note"], status), ""]
    (OUT / "RESULTS.md").write_text("\n".join(L))
    shutil.make_archive("results_bundle", "zip", ".", "results_bundle")
    print(f"bundle written: results_bundle.zip ({Path('results_bundle.zip').stat().st_size/2**20:.1f} MB)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:                       # never break the pipeline's final step
        print("bundle failed:", repr(e))
        sys.exit(0)
