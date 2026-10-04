"""
Stage 6 - Evaluation on the held-out TEST DAYS.

Produces everything Chapter 6 of your report needs:
  figures/confusion_matrix.png   - row-normalised, percentages
  figures/roc_curves.png         - one-vs-rest, per class, with AUC
  figures/training_curves.png    - loss and macro-F1 vs epoch
  figures/pr_pre_attack.png      - precision-recall for the rare class
  results/classification_report.txt / .csv

Run:  python 06_eval.py
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc, precision_recall_curve,
                             average_precision_score)
from sklearn.preprocessing import label_binarize

from config import SEQ_DIR, ARTIFACT_DIR, CLASS_NAMES, BATCH_SIZE

FIG = ARTIFACT_DIR / "figures"; FIG.mkdir(exist_ok=True)
RES = ARTIFACT_DIR / "results"; RES.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")


def predict_in_batches(model, X, batch=BATCH_SIZE * 4):
    out = []
    for i in range(0, len(X), batch):
        out.append(model.predict(np.asarray(X[i:i + batch], dtype=np.float32),
                                 verbose=0))
        if i % (batch * 50) == 0:
            print(f"  {i:,}/{len(X):,}", end="\r")
    return np.concatenate(out)


def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=axes[0],
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar=False)
    axes[0].set_title("Confusion matrix (counts)")
    sns.heatmap(cmn, annot=True, fmt=".2%", cmap="Blues", ax=axes[1],
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, vmin=0, vmax=1)
    axes[1].set_title("Row-normalised (diagonal = recall)")
    for ax in axes:
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(FIG / "confusion_matrix.png", dpi=200)
    plt.close()
    return cm


def plot_roc(y_true, proba):
    yb = label_binarize(y_true, classes=[0, 1, 2])
    plt.figure(figsize=(7, 6))
    aucs = {}
    for i, name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(yb[:, i], proba[:, i])
        a = auc(fpr, tpr); aucs[name] = a
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {a:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("One-vs-rest ROC curves")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(FIG / "roc_curves.png", dpi=200); plt.close()
    return aucs


def plot_pr_pre_attack(y_true, proba):
    yb = (y_true == 1).astype(int)
    prec, rec, _ = precision_recall_curve(yb, proba[:, 1])
    ap = average_precision_score(yb, proba[:, 1])
    plt.figure(figsize=(7, 5.5))
    plt.plot(rec, prec, lw=2, label=f"Pre-Attack (AP = {ap:.4f})")
    plt.axhline(yb.mean(), ls="--", c="grey", lw=1,
                label=f"Base rate = {yb.mean():.4f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall - Pre-Attack class")
    plt.legend(); plt.tight_layout()
    plt.savefig(FIG / "pr_pre_attack.png", dpi=200); plt.close()
    return ap


def plot_history():
    p = ARTIFACT_DIR / "history.json"
    if not p.exists():
        return
    h = json.load(open(p))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(h["loss"], label="train"); axes[0].plot(h["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    if "macro_f1" in h:
        axes[1].plot(h["macro_f1"], label="train")
        axes[1].plot(h["val_macro_f1"], label="val")
        axes[1].set_title("Macro F1"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    plt.tight_layout(); plt.savefig(FIG / "training_curves.png", dpi=200)
    plt.close()


def main():
    X = np.load(SEQ_DIR / "X_test.npy", mmap_mode="r")
    y = np.load(SEQ_DIR / "y_test.npy")
    print(f"Test windows: {X.shape}")

    model = tf.keras.models.load_model(ARTIFACT_DIR / "best_model.keras",
                                       compile=False)
    proba = predict_in_batches(model, X)
    pred = proba.argmax(axis=1)

    report = classification_report(y, pred, labels=[0, 1, 2],
                                   target_names=CLASS_NAMES, digits=4,
                                   zero_division=0)
    print("\n" + report)
    (RES / "classification_report.txt").write_text(report)

    rep_dict = classification_report(y, pred, labels=[0, 1, 2],
                                     target_names=CLASS_NAMES,
                                     output_dict=True, zero_division=0)
    pd.DataFrame(rep_dict).T.to_csv(RES / "classification_report.csv")

    cm = plot_confusion(y, pred)
    aucs = plot_roc(y, proba)
    ap = plot_pr_pre_attack(y, proba)
    plot_history()

    # The two numbers that actually matter operationally.
    missed = cm[2].sum() - cm[2, 2]
    fpr_benign = (cm[0].sum() - cm[0, 0]) / max(cm[0].sum(), 1)
    print(f"\nMissed attacks (false negatives): {missed:,} "
          f"of {cm[2].sum():,} ({100*missed/max(cm[2].sum(),1):.2f}%)")
    print(f"False alarm rate on benign traffic: {100*fpr_benign:.3f}%")
    print(f"Pre-Attack average precision: {ap:.4f}")
    print("AUCs:", {k: round(v, 4) for k, v in aucs.items()})
    print(f"\nFigures -> {FIG}")


if __name__ == "__main__":
    main()
