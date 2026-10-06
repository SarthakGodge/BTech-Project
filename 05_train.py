"""
Stage 5 - hybrid CNN + LSTM with selectable imbalance handling.

  python 05_train.py --balance weights            # default
  python 05_train.py --balance none  --tag none   # baseline, plain CE
  python 05_train.py --balance sampler --tag sampler
  python 05_train.py --balance smote --tag smote
  python 05_train.py --seq holdout_cic2017 --tag xds   # cross-dataset run

--balance
  none     all windows (capped), plain cross-entropy, no weights
  weights  benign undersampled, sqrt-balanced class weights, focal loss
  sampler  every batch holds ~equal numbers of each class (minority is
           re-drawn), focal loss, no weights
  smote    window-level SMOTE on a capped subset (flattened T*F vectors),
           focal loss, no weights. Needs imbalanced-learn.

Everything is evaluated with per-class precision/recall/F1 on val and test;
accuracy alone is misleading on this data.
"""
import argparse
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight

import config as C

SEED = C.SEED


# ------------------------------------------------------------------ model
def build_model(timesteps, n_features, n_classes=3):
    L = tf.keras.layers
    inp = L.Input(shape=(timesteps, n_features), name="flow_window")
    x = L.Conv1D(64, 3, padding="same", use_bias=False)(inp)
    x = L.BatchNormalization()(x)
    x = L.Activation("relu")(x)
    x = L.Conv1D(128, 3, padding="same", use_bias=False)(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("relu")(x)
    x = L.MaxPooling1D(2)(x)
    x = L.Dropout(0.3)(x)
    x = L.LSTM(128, return_sequences=True)(x)
    x = L.Dropout(0.3)(x)
    x = L.LSTM(64)(x)
    x = L.Dense(64, activation="relu")(x)
    x = L.Dropout(0.4)(x)
    out = L.Dense(n_classes, activation="softmax", name="predictions")(x)
    return tf.keras.Model(inp, out, name="cnn_lstm_ids")


def focal_loss(gamma=2.0, class_weights=None):
    cw = None if class_weights is None else tf.constant(
        [float(class_weights.get(i, 1.0)) for i in range(3)], tf.float32)   # missing class -> 1.0

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        p_t = tf.gather(y_pred, y_true, batch_dims=1)
        loss = tf.pow(1.0 - p_t, gamma) * -tf.math.log(p_t)
        if cw is not None:
            loss *= tf.gather(cw, y_true)
        return tf.reduce_mean(loss)
    return loss_fn


class MacroF1(tf.keras.metrics.Metric):
    def __init__(self, n_classes=3, name="macro_f1", **kw):
        super().__init__(name=name, **kw)
        self.n = n_classes
        self.cm = self.add_weight(name="cm", shape=(n_classes, n_classes),
                                  initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        yt = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        yp = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
        self.cm.assign_add(tf.math.confusion_matrix(yt, yp, self.n, dtype=tf.float32))

    def result(self):
        tp = tf.linalg.diag_part(self.cm)
        fp = tf.reduce_sum(self.cm, 0) - tp
        fn = tf.reduce_sum(self.cm, 1) - tp
        return tf.reduce_mean(2 * tp / (2 * tp + fp + fn + 1e-9))

    def reset_state(self):
        self.cm.assign(tf.zeros_like(self.cm))


# ------------------------------------------------------------------- data
def load_split(seq_dir, name):
    return (np.load(seq_dir / f"X_{name}.npy", mmap_mode="r"),
            np.load(seq_dir / f"y_{name}.npy"))


def select_train_indices(y, rng, keep_ratio):
    parts = [np.flatnonzero(y == c) for c in (0, 1, 2)]
    parts[0] = rng.choice(parts[0], int(len(parts[0]) * keep_ratio), replace=False)
    minority = np.concatenate(parts[1:])
    if C.MAX_TRAIN_WINDOWS and len(parts[0]) + len(minority) > C.MAX_TRAIN_WINDOWS:
        budget = max(0, C.MAX_TRAIN_WINDOWS - len(minority))
        parts[0] = rng.choice(parts[0], budget, replace=False)
    return parts                                  # list of per-class index arrays


def _attach_fetch(idx_ds, X, y):
    def fetch(bidx):
        bi = np.sort(bidx.numpy())
        return X[bi].astype(np.float32), y[bi].astype(np.int32)

    def wrap(bidx):
        xb, yb = tf.py_function(fetch, [bidx], [tf.float32, tf.int32])
        xb.set_shape((None, X.shape[1], X.shape[2]))
        yb.set_shape((None,))
        return xb, yb
    return idx_ds.map(wrap, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


def index_dataset(X, y, indices, shuffle):
    ds = tf.data.Dataset.from_tensor_slices(indices)
    if shuffle:
        ds = ds.shuffle(min(len(indices), 200_000), seed=SEED,
                        reshuffle_each_iteration=True)
    return _attach_fetch(ds.batch(C.BATCH_SIZE), X, y)


def balanced_dataset(X, y, parts):
    dss = [tf.data.Dataset.from_tensor_slices(p)
           .shuffle(min(len(p), 100_000), seed=SEED).repeat() for p in parts if len(p)]
    ds = tf.data.Dataset.sample_from_datasets(dss, [1 / len(dss)] * len(dss), seed=SEED)
    return _attach_fetch(ds.batch(C.BATCH_SIZE), X, y)


def smote_arrays(X, y, parts, rng):
    from imblearn.over_sampling import SMOTE
    major = parts[0]
    if len(major) > C.SMOTE_MAJOR_CAP:
        major = rng.choice(major, C.SMOTE_MAJOR_CAP, replace=False)
    sel = [major]
    for p in parts[1:]:
        if len(p) > C.SMOTE_SOURCE_CAP:
            p = rng.choice(p, C.SMOTE_SOURCE_CAP, replace=False)
        sel.append(p)
    idx = np.sort(np.concatenate(sel))
    Xs, ys = np.asarray(X[idx], dtype=np.float32), np.asarray(y[idx])
    n, T, F = Xs.shape
    target = int(C.SMOTE_TARGET_FRAC * (ys == 0).sum())
    strat = {c: max(target, int((ys == c).sum()))
             for c in (1, 2) if 6 <= (ys == c).sum()}
    k = min(5, min(int((ys == c).sum()) for c in strat) - 1) if strat else 1
    print(f"SMOTE: {np.bincount(ys, minlength=3).tolist()} -> targets {strat}, k={k}")
    Xr, yr = SMOTE(sampling_strategy=strat, k_neighbors=k,
                   random_state=SEED).fit_resample(Xs.reshape(n, -1), ys)
    return Xr.reshape(-1, T, F).astype(np.float32), yr.astype(np.int32)


# ------------------------------------------------------------------- eval
def predict_probs(model, X, y, rng):
    idx = np.arange(len(y))
    if len(idx) > C.MAX_EVAL_WINDOWS:
        idx = np.sort(rng.choice(idx, C.MAX_EVAL_WINDOWS, replace=False))
    if len(idx) == 0:
        return idx, np.zeros((0, 3), dtype=np.float32)
    return idx, model.predict(index_dataset(X, y, idx, False), verbose=0)


def tune_class_scales(prob, y):
    """Multiply class-1/2 probabilities by factors chosen on VAL to maximise macro-F1."""
    from sklearn.metrics import f1_score
    grid = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
    best, best_f = np.ones(3), f1_score(y, prob.argmax(1), average="macro")
    for a in grid:
        for b in grid:
            sc = np.array([1.0, a, b])
            f = f1_score(y, (prob * sc).argmax(1), average="macro")
            if f > best_f + 1e-9:
                best, best_f = sc, f
    return best, best_f


def make_report(yt, pred, name, out, ds_ids=None):
    rep = classification_report(yt, pred, labels=[0, 1, 2], target_names=C.CLASS_NAMES,
                                output_dict=True, zero_division=0)
    print(f"\n=== {name} ({len(yt):,} windows) ===")
    print(classification_report(yt, pred, labels=[0, 1, 2], target_names=C.CLASS_NAMES,
                                zero_division=0, digits=4))
    print("confusion matrix (rows=true):\n", confusion_matrix(yt, pred, labels=[0, 1, 2]))
    if ds_ids is not None:
        names = list(C.DATASET_DIRS)
        per = {}
        for k in np.unique(ds_ids):
            m = ds_ids == k
            per[names[k] if 0 <= k < len(names) else str(k)] = float(
                f1_score(yt[m], pred[m], average="macro", labels=[0, 1, 2], zero_division=0))
        rep["macro_f1_by_dataset"] = per
        print("macro-F1 by dataset:", {k: round(v, 4) for k, v in per.items()})
    out[name] = rep


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--balance", choices=["none", "weights", "sampler", "smote"],
                    default="weights")
    ap.add_argument("--loss", choices=["focal", "ce"], default=None)
    ap.add_argument("--keep-ratio", type=float, default=0.5,
                    help="share of benign training windows kept")
    ap.add_argument("--seq", default="", help="sub-folder of SEQ_DIR")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--no-tune", action="store_true", help="skip class-scale tuning on val")
    args = ap.parse_args()
    global SEED
    SEED = args.seed
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    tag = f"_{args.tag}" if args.tag else ""
    seq_dir = C.SEQ_DIR / args.seq if args.seq else C.SEQ_DIR
    loss_kind = args.loss or ("ce" if args.balance == "none" else "focal")
    keep = 1.0 if args.balance == "none" else args.keep_ratio
    rng = np.random.default_rng(SEED)

    Xtr, ytr = load_split(seq_dir, "train")
    Xva, yva = load_split(seq_dir, "val")
    Xte, yte = load_split(seq_dir, "test")
    print(f"train {Xtr.shape} | val {Xva.shape} | test {Xte.shape} | "
          f"balance={args.balance} loss={loss_kind}")

    parts = select_train_indices(ytr, rng, keep)
    print("train windows per class:", [len(p) for p in parts])
    for c, p in enumerate(parts):
        if len(p) == 0:
            print(f"WARNING: no '{C.CLASS_NAMES[c]}' windows in the training split - "
                  f"the model cannot learn that class (expected for some cross-dataset runs).")
    ylist = np.concatenate([np.full(len(p), c) for c, p in enumerate(parts)])

    cw = None
    if args.balance in ("none", "weights"):
        if args.balance == "weights":
            classes = np.unique(ylist)
            w = compute_class_weight("balanced", classes=classes, y=ylist)
            cw = {int(c): float(np.sqrt(v)) for c, v in zip(classes, w)}   # softened
            print("class weights (sqrt-balanced):", cw)

    if args.balance == "smote":
        Xs, ys = smote_arrays(Xtr, ytr, parts, rng)
        order = rng.permutation(len(ys))
        train_ds = (tf.data.Dataset.from_tensor_slices((Xs[order], ys[order]))
                    .shuffle(100_000, seed=SEED).batch(C.BATCH_SIZE)
                    .prefetch(tf.data.AUTOTUNE))
        steps = None
    elif args.balance == "sampler":
        train_ds = balanced_dataset(Xtr, ytr, parts)
        steps = max(1, sum(len(p) for p in parts) // C.BATCH_SIZE)
    else:
        idx = np.concatenate(parts)
        rng.shuffle(idx)
        train_ds = index_dataset(Xtr, ytr, idx, shuffle=True)
        steps = None

    va_idx = np.arange(len(yva))
    if len(va_idx) > C.MAX_EVAL_WINDOWS:
        va_idx = np.sort(rng.choice(va_idx, C.MAX_EVAL_WINDOWS, replace=False))
    val_ds = index_dataset(Xva, yva, va_idx, shuffle=False)

    model = build_model(Xtr.shape[1], Xtr.shape[2])
    loss = (focal_loss(C.FOCAL_GAMMA, cw) if loss_kind == "focal"
            else tf.keras.losses.SparseCategoricalCrossentropy())
    model.compile(tf.keras.optimizers.Adam(C.LEARNING_RATE), loss=loss,
                  metrics=["accuracy", MacroF1()])

    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_macro_f1", mode="max", patience=6,
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_macro_f1", mode="max",
                                             factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(str(C.ARTIFACT_DIR / f"best_model{tag}.keras"),
                                           monitor="val_macro_f1", mode="max",
                                           save_best_only=True, verbose=1),
        tf.keras.callbacks.CSVLogger(str(C.ARTIFACT_DIR / f"training_log{tag}.csv")),
    ]
    fit_kw = dict(validation_data=val_ds, epochs=C.EPOCHS, callbacks=cbs, verbose=1)
    if steps:
        fit_kw["steps_per_epoch"] = steps
    if loss_kind == "ce" and cw:
        fit_kw["class_weight"] = cw
    history = model.fit(train_ds, **fit_kw)

    model.export(str(C.ARTIFACT_DIR / f"saved_model{tag}"))

    report = {}
    va_i, va_p = predict_probs(model, Xva, yva, rng)
    te_i, te_p = predict_probs(model, Xte, yte, rng)
    yv, yt = yva[va_i], yte[te_i]
    dte = None
    if (seq_dir / "d_test.npy").exists():
        dte = np.load(seq_dir / "d_test.npy")[te_i]
    make_report(yv, va_p.argmax(1), "val", report)
    make_report(yt, te_p.argmax(1), "test", report, dte)
    if not args.no_tune and len(yv):
        scales, f = tune_class_scales(va_p, yv)          # tuned on VAL only
        print(f"\nclass scales tuned on val: {scales.tolist()} (val macro-F1 {f:.4f})")
        make_report(yt, (te_p * scales).argmax(1), "test_tuned", report, dte)
        with open(C.ARTIFACT_DIR / f"thresholds{tag}.json", "w") as fh:
            json.dump({"class_scale": scales.tolist()}, fh)
    np.save(C.ARTIFACT_DIR / f"test_probs{tag}.npy", te_p.astype(np.float32))
    np.save(C.ARTIFACT_DIR / f"test_idx{tag}.npy", te_i)
    with open(C.ARTIFACT_DIR / f"history{tag}.json", "w") as fh:
        json.dump({k: [float(v) for v in vs] for k, vs in history.history.items()}, fh, indent=2)
    with open(C.ARTIFACT_DIR / f"report{tag}.json", "w") as fh:
        json.dump({"args": vars(args), "reports": report}, fh, indent=2)
    print(f"\nsaved artifacts with tag '{tag or 'default'}'")


if __name__ == "__main__":
    main()
