"""
Stage 6 - Evaluation on the held-out TEST blocks (v2).

  python 06_eval.py                          # default model, pooled test set
  python 06_eval.py --tag weights_s1         # a tagged run
  python 06_eval.py --seq holdout_cic2017 --tag xds

Produces (under artifacts/figures and artifacts/results):
  confusion_matrix{tag}.png, roc_curves{tag}.png, pr_pre_attack{tag}.png,
  training_curves{tag}.png, classification_report{tag}.txt/.csv,
  per_dataset{tag}.csv
If artifacts/thresholds{tag}.json exists (tuned on VAL by 05_train.py) the
class scales are applied and both raw and tuned numbers are printed.
"""
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, auc,
                             precision_recall_curve, average_precision_score, f1_score)
from sklearn.preprocessing import label_binarize

import config as C

FIG = C.ARTIFACT_DIR / "figures"; FIG.mkdir(exist_ok=True)
RES = C.ARTIFACT_DIR / "results"; RES.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")


def predict_in_batches(model, X, batch=C.BATCH_SIZE * 4):
    out = []
    for i in range(0, len(X), batch):
        out.append(model.predict(np.asarray(X[i:i + batch], dtype=np.float32), verbose=0))
        if (i // batch) % 50 == 0:
            print(f"  {i:,}/{len(X):,}", end="\r")
    return np.concatenate(out) if out else np.zeros((0, 3), np.float32)


def plot_confusion(y_true, y_pred, tag):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=axes[0],
                xticklabels=C.CLASS_NAMES, yticklabels=C.CLASS_NAMES, cbar=False)
    axes[0].set_title("Confusion matrix (counts)")
    sns.heatmap(cmn, annot=True, fmt=".2%", cmap="Blues", ax=axes[1],
                xticklabels=C.CLASS_NAMES, yticklabels=C.CLASS_NAMES, vmin=0, vmax=1)
    axes[1].set_title("Row-normalised (diagonal = recall)")
    for ax in axes:
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout(); plt.savefig(FIG / f"confusion_matrix{tag}.png", dpi=200); plt.close()
    return cm


def plot_roc(y_true, proba, tag):
    yb = label_binarize(y_true, classes=[0, 1, 2])
    plt.figure(figsize=(7, 6)); aucs = {}
    for i, name in enumerate(C.CLASS_NAMES):
        if yb[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(yb[:, i], proba[:, i])
        aucs[name] = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {aucs[name]:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("One-vs-rest ROC curves"); plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(FIG / f"roc_curves{tag}.png", dpi=200); plt.close()
    return aucs


def plot_pr_pre_attack(y_true, proba, tag):
    yb = (y_true == 1).astype(int)
    if yb.sum() == 0:
        return float("nan")
    prec, rec, _ = precision_recall_curve(yb, proba[:, 1])
    ap = average_precision_score(yb, proba[:, 1])
    plt.figure(figsize=(7, 5.5))
    plt.plot(rec, prec, lw=2, label=f"Pre-Attack (AP = {ap:.4f})")
    plt.axhline(yb.mean(), ls="--", c="grey", lw=1, label=f"Base rate = {yb.mean():.4f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall - Pre-Attack")
    plt.legend(); plt.tight_layout()
    plt.savefig(FIG / f"pr_pre_attack{tag}.png", dpi=200); plt.close()
    return ap


def plot_history(tag):
    p = C.ARTIFACT_DIR / f"history{tag}.json"
    if not p.exists():
        return
    h = json.load(open(p))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(h["loss"], label="train"); axes[0].plot(h["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    if "macro_f1" in h:
        axes[1].plot(h["macro_f1"], label="train"); axes[1].plot(h["val_macro_f1"], label="val")
        axes[1].set_title("Macro F1"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    plt.tight_layout(); plt.savefig(FIG / f"training_curves{tag}.png", dpi=200); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--seq", default="")
    args = ap.parse_args()
    tag = f"_{args.tag}" if args.tag else ""
    seq_dir = C.SEQ_DIR / args.seq if args.seq else C.SEQ_DIR

    X = np.load(seq_dir / "X_test.npy", mmap_mode="r")
    y = np.load(seq_dir / "y_test.npy")
    print(f"Test windows: {X.shape}")
    model = tf.keras.models.load_model(C.ARTIFACT_DIR / f"best_model{tag}.keras", compile=False)
    proba = predict_in_batches(model, X)

    th = C.ARTIFACT_DIR / f"thresholds{tag}.json"
    scales = np.array(json.load(open(th))["class_scale"]) if th.exists() else np.ones(3)
    for name, pr in (("raw", proba), ("tuned", proba * scales)):
        if name == "tuned" and not th.exists():
            continue
        pred = pr.argmax(1)
        txt = classification_report(y, pred, labels=[0, 1, 2], target_names=C.CLASS_NAMES,
                                    digits=4, zero_division=0)
        print(f"\n--- {name} ---\n{txt}")
        suffix = tag if name == "raw" else f"{tag}_tuned"
        (RES / f"classification_report{suffix}.txt").write_text(txt)
        d = classification_report(y, pred, labels=[0, 1, 2], target_names=C.CLASS_NAMES,
                                  output_dict=True, zero_division=0)
        pd.DataFrame(d).T.to_csv(RES / f"classification_report{suffix}.csv")

    pred = (proba * scales).argmax(1)
    cm = plot_confusion(y, pred, tag)
    aucs = plot_roc(y, proba, tag)
    ap_ = plot_pr_pre_attack(y, proba, tag)
    plot_history(tag)

    if (seq_dir / "d_test.npy").exists():                      # per-dataset breakdown
        dsid = np.load(seq_dir / "d_test.npy")
        names = list(C.DATASET_DIRS)
        rows = []
        for k in np.unique(dsid):
            m = dsid == k
            rows.append({"dataset": names[k] if 0 <= k < len(names) else k, "windows": int(m.sum()),
                         "macro_f1": f1_score(y[m], pred[m], average="macro", labels=[0, 1, 2],
                                              zero_division=0),
                         "accuracy": float((y[m] == pred[m]).mean())})
        df = pd.DataFrame(rows); print("\nPer dataset:\n", df.to_string(index=False))
        df.to_csv(RES / f"per_dataset{tag}.csv", index=False)

    missed = cm[2].sum() - cm[2, 2]
    fpr_benign = (cm[0].sum() - cm[0, 0]) / max(cm[0].sum(), 1)
    print(f"\nMissed attacks: {missed:,} of {cm[2].sum():,} ({100*missed/max(cm[2].sum(),1):.2f}%)")
    print(f"False alarm rate on benign: {100*fpr_benign:.3f}%")
    print(f"Pre-Attack average precision: {ap_:.4f}")
    print("AUCs:", {k: round(v, 4) for k, v in aucs.items()})
    print(f"\nFigures -> {FIG}")


if __name__ == "__main__":
    main()
