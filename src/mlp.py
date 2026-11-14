"""MLP for the deep-learning comparison (H1a), written in Keras and trained on the GPU.

Design, following the tracker (T3.2):

* inputs are median-imputed and standardised, both fitted on the training data only;
* two or three hidden layers with ReLU and dropout, one logit as output;
* the loss is binary cross-entropy on logits with ``pos_weight = n_neg / n_pos``,
  the same quantity as ``BCEWithLogitsLoss(pos_weight=...)`` in PyTorch;
* training stops when validation PR-AUC has not improved for ``patience`` epochs
  and the weights of the best epoch are restored.

TensorFlow is imported only when the first model is built, so the scripts that
use LightGBM alone start quickly.
"""
from __future__ import annotations

import os

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # hide C++ info, warnings and graph notices

_TF = None


def tensorflow():
    """Import TensorFlow once: grow GPU memory on demand and switch on deterministic ops."""
    global _TF
    if _TF is None:
        import tensorflow as tf
        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass                                   # already initialised
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
        _TF = tf
    return _TF


def device_name() -> str:
    """Human-readable name of the device the MLP trains on."""
    tf = tensorflow()
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return "CPU"
    details = tf.config.experimental.get_device_details(gpus[0])
    return details.get("device_name", "GPU")


class KerasMLP:
    """Small fully connected network with a scikit-learn style interface."""

    def __init__(self, hidden: int = 128, n_layers: int = 2, dropout: float = 0.2,
                 learning_rate: float = 1e-3, weight_decay: float = 1e-5, batch_size: int = 256,
                 max_epochs: int = 200, patience: int = 10, seed: int = 0):
        self.hidden = hidden
        self.n_layers = n_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed

    # ----------------------------------------------------------------- preprocessing
    def _prepare(self, X) -> np.ndarray:
        X = np.asarray(X, dtype="float64")
        return self.scaler_.transform(self.imputer_.transform(X)).astype("float32")

    # ----------------------------------------------------------------- training
    def fit(self, X_train, y_train, X_val, y_val):
        tf = tensorflow()
        import keras

        keras.utils.set_random_seed(self.seed)
        y_train = np.asarray(y_train).astype("float32")
        y_val = np.asarray(y_val).astype(int)

        self.imputer_ = SimpleImputer(strategy="median").fit(np.asarray(X_train, dtype="float64"))
        self.scaler_ = StandardScaler().fit(self.imputer_.transform(np.asarray(X_train, dtype="float64")))
        Xt, Xv = self._prepare(X_train), tf.constant(self._prepare(X_val))

        pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

        def weighted_bce(y_true, logits):
            return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
                labels=tf.cast(y_true, tf.float32), logits=logits, pos_weight=pos_weight))

        layers = [keras.Input(shape=(Xt.shape[1],))]
        for _ in range(self.n_layers):
            layers += [keras.layers.Dense(self.hidden, activation="relu"), keras.layers.Dropout(self.dropout)]
        layers.append(keras.layers.Dense(1))
        model = keras.Sequential(layers)
        model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=self.learning_rate, weight_decay=self.weight_decay),
            loss=weighted_bce)

        outer = self

        class StopOnValidationPRAUC(keras.callbacks.Callback):
            def on_train_begin(self, logs=None):
                self.best, self.best_epoch, self.wait = -np.inf, 0, 0
                self.best_weights, self.curve = None, []

            def on_epoch_end(self, epoch, logs=None):
                logits = self.model(Xv, training=False).numpy().ravel()
                score = float(average_precision_score(y_val, logits))
                self.curve.append(score)
                if score > self.best:
                    self.best, self.best_epoch, self.wait = score, epoch + 1, 0
                    self.best_weights = self.model.get_weights()
                else:
                    self.wait += 1
                    if self.wait >= outer.patience:
                        self.model.stop_training = True

        stopper = StopOnValidationPRAUC()
        model.fit(Xt, y_train.reshape(-1, 1), batch_size=self.batch_size, epochs=self.max_epochs,
                  shuffle=True, verbose=0, callbacks=[stopper])
        model.set_weights(stopper.best_weights)

        self.model_ = model
        self.best_epoch_ = stopper.best_epoch
        self.best_val_pr_auc_ = stopper.best
        self.val_curve_ = stopper.curve
        return self

    # ----------------------------------------------------------------- prediction
    def decision_function(self, X) -> np.ndarray:
        tf = tensorflow()
        return self.model_(tf.constant(self._prepare(X)), training=False).numpy().ravel()

    def predict_proba(self, X) -> np.ndarray:
        p = 1.0 / (1.0 + np.exp(-self.decision_function(X).astype("float64")))
        return np.column_stack([1.0 - p, p])
