"""
Stage 5 - Build and train the hybrid CNN + LSTM classifier.

Architecture (matches the design document):
    Input (T, F)
    Conv1D 64  k=3 same + BN + ReLU
    Conv1D 128 k=3 same + BN + ReLU
    MaxPooling1D 2
    Dropout 0.3
    LSTM 128 return_sequences=True
    Dropout 0.3
    LSTM 64  return_sequences=False
    Dense 64 ReLU
    Dropout 0.4
    Dense 3 softmax

Class imbalance is handled at the WINDOW level, not the row level:
  1. Majority (No Attack) windows are randomly undersampled.
  2. SMOTE optionally oversamples Pre-Attack windows in flattened space.
  3. Class weights + focal loss handle whatever imbalance remains.

Note on SMOTE: the design doc applies it to raw rows before windowing.
That is wrong - a synthetic row has no position in time, so inserting it
into a sequence corrupts the temporal structure the LSTM exists to learn.
Applying it to whole windows preserves the sequence semantics. Set
USE_SMOTE=False to rely on class weights + focal loss alone, which is the
cleaner result to report.

Run:  python 05_train.py
"""
import json
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from config import (SEQ_DIR, ARTIFACT_DIR, WINDOW, BATCH_SIZE, EPOCHS,
                    LEARNING_RATE, SEED, CLASS_NAMES, FOCAL_GAMMA,
                    USE_FOCAL_LOSS, MAX_TRAIN_WINDOWS)

USE_SMOTE = False          # window-level SMOTE; see note above
MAJORITY_KEEP_RATIO = 0.25  # keep this fraction of No-Attack training windows

tf.random.set_seed(SEED)
np.random.seed(SEED)


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
def build_model(timesteps, n_features, n_classes=3):
    L = tf.keras.layers
    inp = L.Input(shape=(timesteps, n_features), name="flow_window")

    x = L.Conv1D(64, 3, padding="same", use_bias=False, name="conv1")(inp)
    x = L.BatchNormalization(name="bn1")(x)
    x = L.Activation("relu")(x)

    x = L.Conv1D(128, 3, padding="same", use_bias=False, name="conv2")(x)
    x = L.BatchNormalization(name="bn2")(x)
    x = L.Activation("relu")(x)

    x = L.MaxPooling1D(2, name="pool")(x)
    x = L.Dropout(0.3)(x)

    x = L.LSTM(128, return_sequences=True, name="lstm1")(x)
    x = L.Dropout(0.3)(x)
    x = L.LSTM(64, return_sequences=False, name="lstm2")(x)

    x = L.Dense(64, activation="relu", name="dense1")(x)
    x = L.Dropout(0.4)(x)
    out = L.Dense(n_classes, activation="softmax", name="predictions")(x)

    return tf.keras.Model(inp, out, name="cnn_lstm_ids")


# BatchNormalization right after Conv1D makes the conv bias redundant
# (BN's beta absorbs it), hence use_bias=False - fewer params, same result.


def focal_loss(gamma=2.0, class_weights=None):
    """Sparse categorical focal loss.

    Down-weights easy examples so the gradient is dominated by the hard,
    rare Pre-Attack windows rather than the 13M trivially-benign ones.
    """
    cw = None if class_weights is None else tf.constant(
        [class_weights[i] for i in range(len(class_weights))], dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        p_t = tf.gather(y_pred, y_true, batch_dims=1)
        ce = -tf.math.log(p_t)
        modulator = tf.pow(1.0 - p_t, gamma)
        loss = modulator * ce
        if cw is not None:
            loss *= tf.gather(cw, y_true)
        return tf.reduce_mean(loss)

    return loss_fn


class MacroF1(tf.keras.metrics.Metric):
    """Macro-averaged F1 across the three classes, computed per epoch."""

    def __init__(self, n_classes=3, name="macro_f1", **kw):
        super().__init__(name=name, **kw)
        self.n = n_classes
        self.cm = self.add_weight(name="cm", shape=(n_classes, n_classes),
                                  initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        yt = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        yp = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
        cm = tf.math.confusion_matrix(yt, yp, num_classes=self.n,
                                      dtype=tf.float32)
        self.cm.assign_add(cm)

    def result(self):
        tp = tf.linalg.diag_part(self.cm)
        fp = tf.reduce_sum(self.cm, axis=0) - tp
        fn = tf.reduce_sum(self.cm, axis=1) - tp
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)
        return tf.reduce_mean(f1)

    def reset_state(self):
        self.cm.assign(tf.zeros_like(self.cm))


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def load_split(name, mmap=True):
    X = np.load(SEQ_DIR / f"X_{name}.npy", mmap_mode="r" if mmap else None)
    y = np.load(SEQ_DIR / f"y_{name}.npy")
    return X, y


def balance_indices(y, rng):
    """Undersample the majority class; return shuffled index array."""
    idx_parts = []
    for cls in (0, 1, 2):
        idx = np.flatnonzero(y == cls)
        if cls == 0:
            take = int(len(idx) * MAJORITY_KEEP_RATIO)
            idx = rng.choice(idx, size=take, replace=False)
        idx_parts.append(idx)
    out = np.concatenate(idx_parts)

    if MAX_TRAIN_WINDOWS and len(out) > MAX_TRAIN_WINDOWS:
        # Keep every minority window; trim only the majority.
        minority = np.concatenate(idx_parts[1:])
        budget = max(0, MAX_TRAIN_WINDOWS - len(minority))
        out = np.concatenate([rng.choice(idx_parts[0], size=budget,
                                         replace=False), minority])
    rng.shuffle(out)
    return out


def make_dataset(X, y, indices, shuffle, batch=BATCH_SIZE):
    """tf.data pipeline that reads windows from the memmap lazily."""
    idx_ds = tf.data.Dataset.from_tensor_slices(indices)
    if shuffle:
        idx_ds = idx_ds.shuffle(min(len(indices), 200_000), seed=SEED,
                                reshuffle_each_iteration=True)
    idx_ds = idx_ds.batch(batch)

    def fetch(batch_idx):
        bi = np.sort(batch_idx.numpy())          # sorted = fewer page faults
        return X[bi].astype(np.float32), y[bi].astype(np.int32)

    def wrap(batch_idx):
        xb, yb = tf.py_function(fetch, [batch_idx], [tf.float32, tf.int32])
        xb.set_shape((None, X.shape[1], X.shape[2]))
        yb.set_shape((None,))
        return xb, yb

    return idx_ds.map(wrap, num_parallel_calls=tf.data.AUTOTUNE) \
                 .prefetch(tf.data.AUTOTUNE)


# ----------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)

    Xtr, ytr = load_split("train")
    Xva, yva = load_split("val")
    print(f"train windows {Xtr.shape} | val windows {Xva.shape}")

    tr_idx = balance_indices(ytr, rng)
    va_idx = np.arange(len(yva))
    print("after balancing:",
          dict(zip(*np.unique(ytr[tr_idx], return_counts=True))))

    if USE_SMOTE:
        raise NotImplementedError(
            "Window-level SMOTE: flatten to (n, T*F), fit SMOTE, reshape back. "
            "Only do this if you have the RAM - it materialises the array.")

    classes = np.unique(ytr[tr_idx])
    weights = compute_class_weight("balanced", classes=classes, y=ytr[tr_idx])
    cw = {int(c): float(w) for c, w in zip(classes, weights)}
    print("class weights:", cw)

    model = build_model(Xtr.shape[1], Xtr.shape[2])
    model.summary()

    loss = (focal_loss(FOCAL_GAMMA, cw) if USE_FOCAL_LOSS
            else tf.keras.losses.SparseCategoricalCrossentropy())

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss=loss,
        metrics=["accuracy", MacroF1()],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_macro_f1", mode="max",
                                         patience=6, restore_best_weights=True,
                                         verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=3, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            str(ARTIFACT_DIR / "best_model.keras"),
            monitor="val_macro_f1", mode="max", save_best_only=True, verbose=1),
        tf.keras.callbacks.CSVLogger(str(ARTIFACT_DIR / "training_log.csv")),
    ]

    history = model.fit(
        make_dataset(Xtr, ytr, tr_idx, shuffle=True),
        validation_data=make_dataset(Xva, yva, va_idx, shuffle=False),
        epochs=EPOCHS,
        class_weight=None if USE_FOCAL_LOSS else cw,  # focal loss already has it
        callbacks=callbacks,
        verbose=1,
    )

    model.export(str(ARTIFACT_DIR / "saved_model"))   # for TF Serving / FastAPI
    with open(ARTIFACT_DIR / "history.json", "w") as fh:
        json.dump({k: [float(v) for v in vals]
                   for k, vals in history.history.items()}, fh, indent=2)

    print(f"\nSaved model -> {ARTIFACT_DIR/'saved_model'}")


if __name__ == "__main__":
    main()
