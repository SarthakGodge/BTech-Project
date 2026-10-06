"""Model definitions. Imported by 05_train.py AND 06_eval.py so the custom
layer is registered before a .keras file is loaded."""
import tensorflow as tf

L = tf.keras.layers


@tf.keras.utils.register_keras_serializable(package="ids")
class AttentionPool(L.Layer):
    """Learned weighted average over time steps."""

    def __init__(self, units=64, **kw):
        super().__init__(**kw)
        self.units = units
        self.proj = L.Dense(units, activation="tanh")
        self.score = L.Dense(1)

    def call(self, x):
        a = tf.nn.softmax(self.score(self.proj(x)), axis=1)      # (B, T, 1)
        return tf.reduce_sum(a * x, axis=1)

    def get_config(self):
        return {**super().get_config(), "units": self.units}


def build_base(timesteps, n_features, n_classes=3):
    inp = L.Input(shape=(timesteps, n_features), name="flow_window")
    x = L.Conv1D(64, 3, padding="same", use_bias=False)(inp)
    x = L.BatchNormalization()(x); x = L.Activation("relu")(x)
    x = L.Conv1D(128, 3, padding="same", use_bias=False)(x)
    x = L.BatchNormalization()(x); x = L.Activation("relu")(x)
    x = L.MaxPooling1D(2)(x); x = L.Dropout(0.3)(x)
    x = L.LSTM(128, return_sequences=True)(x); x = L.Dropout(0.3)(x)
    x = L.LSTM(64)(x)
    x = L.Dense(64, activation="relu")(x); x = L.Dropout(0.4)(x)
    out = L.Dense(n_classes, activation="softmax", name="predictions")(x)
    return tf.keras.Model(inp, out, name="cnn_lstm_ids")


def build_attn(timesteps, n_features, n_classes=3):
    """Multi-scale CNN -> BiGRU -> attention pooling (+ max pooling)."""
    inp = L.Input(shape=(timesteps, n_features), name="flow_window")
    branches = [L.Conv1D(48, k, padding="same", use_bias=False)(inp) for k in (3, 5, 7)]
    x = L.Concatenate()(branches)
    x = L.BatchNormalization()(x); x = L.Activation("relu")(x)
    x = L.Conv1D(128, 3, padding="same", use_bias=False)(x)
    x = L.BatchNormalization()(x); x = L.Activation("relu")(x)
    x = L.Dropout(0.3)(x)
    x = L.Bidirectional(L.GRU(96, return_sequences=True))(x)
    x = L.Dropout(0.3)(x)
    x = L.Concatenate()([AttentionPool(64)(x), L.GlobalMaxPooling1D()(x)])
    x = L.Dense(96, activation="relu")(x); x = L.Dropout(0.4)(x)
    out = L.Dense(n_classes, activation="softmax", name="predictions")(x)
    return tf.keras.Model(inp, out, name="cnn_bigru_attn_ids")


def build_model(timesteps, n_features, arch="base"):
    return {"base": build_base, "attn": build_attn}[arch](timesteps, n_features)
