import os

import numpy as np
import pandas as pd

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_ipcw
from sksurv.util import Surv

# Force CPU execution on native Windows environments.
if tf.config.list_physical_devices("GPU"):
    tf.config.set_visible_devices([], "GPU")


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def cox_partial_likelihood_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Negative average Cox partial log-likelihood.

    Parameters
    ----------
    y_true : Tensor of shape (batch, 2)
        Column 0 → OS_YEARS (observed time), Column 1 → OS_STATUS (event flag, float).
    y_pred : Tensor of shape (batch, 1)
        Predicted log-risk scores h_θ(x).

    Returns
    -------
    Scalar loss tensor.
    """
    times = tf.cast(y_true[:, 0], tf.float32)
    events = tf.cast(y_true[:, 1], tf.float32)
    log_risk = tf.squeeze(y_pred, axis=-1)

    sorted_idx = tf.argsort(times, direction="ASCENDING")
    times = tf.gather(times, sorted_idx)
    events = tf.gather(events, sorted_idx)
    log_risk = tf.gather(log_risk, sorted_idx)

    exp_risk = tf.exp(log_risk)
    risk_set_sum = tf.cumsum(exp_risk, reverse=True)
    log_risk_set = tf.math.log(risk_set_sum + 1e-8)

    ll_per_patient = events * (log_risk - log_risk_set)
    n_events = tf.reduce_sum(events) + 1e-8

    return -tf.reduce_sum(ll_per_patient) / n_events


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def _build_network(
    n_features: int,
    hidden_sizes: list[int],
    dropout: float,
    l2_reg: float,
    activation: str = "relu",
) -> keras.Model:
    """
    Build a feed-forward DeepSurv network.
    """
    reg = keras.regularizers.l2(l2_reg)
    activation_fn = activation.lower()

    if activation_fn not in {"relu", "selu"}:
        raise ValueError("activation must be either 'relu' or 'selu'")

    inp = keras.Input(shape=(n_features,), name="features")
    x = inp

    for i, units in enumerate(hidden_sizes):
        x = keras.layers.Dense(
            units,
            use_bias=False,
            kernel_regularizer=reg,
            name=f"fc_{i}",
        )(x)
        if activation_fn == "selu":
            x = keras.layers.BatchNormalization(name=f"bn_{i}")(x)
            x = keras.layers.Activation("selu", name=f"selu_{i}")(x)
        else:
            x = keras.layers.BatchNormalization(name=f"bn_{i}")(x)
            x = keras.layers.Activation("relu", name=f"relu_{i}")(x)
        x = keras.layers.Dropout(dropout, name=f"drop_{i}")(x)

    out = keras.layers.Dense(1, activation="linear", name="log_risk")(x)

    return keras.Model(inputs=inp, outputs=out, name="DeepSurv")


# ---------------------------------------------------------------------------
# Configuration generators
# ---------------------------------------------------------------------------

def get_architecture_configs(activation: str = "relu") -> list[dict]:
    """
    Return 10 neural-network configurations that vary hidden sizes,
    dropout, regularisation strength, learning rate, batch size and epochs.
    Use activation='selu' to switch all configurations to SeLU.
    """
    configs = [
        {
            "name": "baseline",
            "hidden_sizes": [64, 64, 64],
            "dropout": 0.3,
            "l2_reg": 1e-4,
            "lr": 1e-3,
            "batch_size": 128,
            "epochs": 250,
            "patience": 20,
        },
        {
            "name": "narrow_shallow",
            "hidden_sizes": [32, 32],
            "dropout": 0.2,
            "l2_reg": 1e-4,
            "lr": 1e-3,
            "batch_size": 64,
            "epochs": 220,
            "patience": 18,
        },
        {
            "name": "wide_shallow",
            "hidden_sizes": [128, 64],
            "dropout": 0.25,
            "l2_reg": 5e-4,
            "lr": 5e-4,
            "batch_size": 128,
            "epochs": 260,
            "patience": 20,
        },
        {
            "name": "deep_medium",
            "hidden_sizes": [64, 64, 64, 64],
            "dropout": 0.3,
            "l2_reg": 1e-4,
            "lr": 1e-3,
            "batch_size": 128,
            "epochs": 300,
            "patience": 25,
        },
        {
            "name": "bottleneck",
            "hidden_sizes": [128, 64, 32],
            "dropout": 0.25,
            "l2_reg": 1e-4,
            "lr": 1e-3,
            "batch_size": 64,
            "epochs": 250,
            "patience": 20,
        },
        {
            "name": "wider_dropout",
            "hidden_sizes": [96, 96, 96],
            "dropout": 0.45,
            "l2_reg": 1e-4,
            "lr": 8e-4,
            "batch_size": 96,
            "epochs": 240,
            "patience": 18,
        },
        {
            "name": "strong_reg",
            "hidden_sizes": [64, 64, 64],
            "dropout": 0.2,
            "l2_reg": 5e-3,
            "lr": 1e-3,
            "batch_size": 128,
            "epochs": 250,
            "patience": 20,
        },
        {
            "name": "small_lr",
            "hidden_sizes": [80, 40, 20],
            "dropout": 0.15,
            "l2_reg": 1e-4,
            "lr": 5e-4,
            "batch_size": 64,
            "epochs": 280,
            "patience": 22,
        },
        {
            "name": "larger_batches",
            "hidden_sizes": [128, 128, 64],
            "dropout": 0.3,
            "l2_reg": 1e-4,
            "lr": 1e-3,
            "batch_size": 256,
            "epochs": 220,
            "patience": 18,
        },
        {
            "name": "deeper_narrow",
            "hidden_sizes": [64, 32, 16, 16],
            "dropout": 0.2,
            "l2_reg": 5e-4,
            "lr": 7e-4,
            "batch_size": 96,
            "epochs": 300,
            "patience": 25,
        },
    ]
    for cfg in configs:
        cfg["activation"] = activation.lower()
    return configs


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DeepSurv:
    """
    DeepSurv: Cox Proportional Hazards Deep Neural Network.
    """

    def __init__(
        self,
        hidden_sizes: list[int] = (64, 64, 64),
        dropout: float = 0.3,
        l2_reg: float = 1e-4,
        lr: float = 1e-3,
        batch_size: int = 128,
        epochs: int = 300,
        patience: int = 20,
        val_fraction: float = 0.15,
        random_state: int | None = 42,
        verbose: int = 1,
        activation: str = "relu",
    ):
        self.hidden_sizes = list(hidden_sizes)
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.verbose = verbose
        self.activation = activation.lower()

        self.model_: keras.Model | None = None
        self.scaler_: StandardScaler | None = None
        self.history_: dict | None = None
        self._y_train_structured = None

    @staticmethod
    def _to_arrays(X: pd.DataFrame | np.ndarray, y: pd.DataFrame | None = None):
        """Convert inputs to plain numpy arrays."""
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        if y is None:
            return X_arr.astype(np.float32), None, None

        if isinstance(y, pd.DataFrame):
            times = y["OS_YEARS"].values.astype(np.float32)
            events = y["OS_STATUS"].values.astype(np.float32)
        else:
            times = y["time"].astype(np.float32)
            events = y["event"].astype(np.float32)

        y_arr = np.stack([times, events], axis=1)
        return X_arr.astype(np.float32), y_arr, times, events

    @staticmethod
    def _make_structured(times: np.ndarray, events: np.ndarray) -> np.ndarray:
        """Build a sksurv-compatible structured array."""
        return Surv.from_arrays(event=events.astype(bool), time=times)

    def _stratified_split(self, X, y_arr):
        """Hold out a validation subset, stratified by event status."""
        rng = np.random.default_rng(self.random_state)
        n = len(X)
        events = y_arr[:, 1].astype(bool)

        ev_idx = np.where(events)[0]
        cen_idx = np.where(~events)[0]
        rng.shuffle(ev_idx)
        rng.shuffle(cen_idx)

        n_val_ev = max(1, int(len(ev_idx) * self.val_fraction))
        n_val_cen = max(1, int(len(cen_idx) * self.val_fraction))

        val_idx = np.concatenate([ev_idx[:n_val_ev], cen_idx[:n_val_cen]])
        train_idx = np.concatenate([ev_idx[n_val_ev:], cen_idx[n_val_cen:]])

        return X[train_idx], y_arr[train_idx], X[val_idx], y_arr[val_idx]

    def _make_batches(self, X, y_arr):
        """Yield mini-batches for training."""
        n = len(X)
        rng = np.random.default_rng(self.random_state)
        idx = rng.permutation(n)

        for start in range(0, n, self.batch_size):
            batch_idx = idx[start : start + self.batch_size]
            yield X[batch_idx], y_arr[batch_idx]

    def fit(self, X, y):
        """Fit DeepSurv on training data."""
        if self.random_state is not None:
            tf.random.set_seed(self.random_state)
            np.random.seed(self.random_state)

        X_arr, y_arr, times, events = self._to_arrays(X, y)

        self.scaler_ = StandardScaler()
        X_arr = self.scaler_.fit_transform(X_arr)
        X_tr, y_tr, X_val, y_val = self._stratified_split(X_arr, y_arr)

        self._y_train_structured = self._make_structured(y_tr[:, 0], y_tr[:, 1])

        self.model_ = _build_network(
            n_features=X_arr.shape[1],
            hidden_sizes=self.hidden_sizes,
            dropout=self.dropout,
            l2_reg=self.l2_reg,
            activation=getattr(self, "activation", "relu"),
        )
        self.model_.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.lr),
            loss=cox_partial_likelihood_loss,
        )

        if self.verbose:
            self.model_.summary()

        best_cindex = -np.inf
        best_weights = None
        patience_count = 0
        train_losses = []
        val_cindices = []

        y_val_struct = self._make_structured(y_val[:, 0], y_val[:, 1])

        for epoch in range(self.epochs):
            epoch_loss = []
            for X_batch, y_batch in self._make_batches(X_tr, y_tr):
                X_t = tf.constant(X_batch, dtype=tf.float32)
                y_t = tf.constant(y_batch, dtype=tf.float32)
                loss_val = self.model_.train_on_batch(X_t, y_t)
                epoch_loss.append(float(loss_val))

            mean_loss = float(np.mean(epoch_loss))
            train_losses.append(mean_loss)

            val_risk = self.predict(X_val, already_scaled=True)
            try:
                c_idx = concordance_index_ipcw(
                    self._y_train_structured, y_val_struct, estimate=val_risk
                )[0]
            except Exception:
                c_idx = np.nan

            val_cindices.append(c_idx)

            if self.verbose:
                print(f"Epoch {epoch + 1:>4}/{self.epochs}  loss={mean_loss:.4f}  val_cindex={c_idx:.4f}")

            if not np.isnan(c_idx) and c_idx > best_cindex:
                best_cindex = c_idx
                best_weights = self.model_.get_weights()
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= self.patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch + 1}. Best val C-index: {best_cindex:.4f}")
                    break

        if best_weights is not None:
            self.model_.set_weights(best_weights)

        self.history_ = {
            "train_loss": train_losses,
            "val_cindex": val_cindices,
        }

        return self

    def predict(self, X, already_scaled: bool = False) -> np.ndarray:
        """Return predicted log-risk scores."""
        X_arr = X if isinstance(X, np.ndarray) else (
            X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        )
        X_arr = X_arr.astype(np.float32)

        if not already_scaled:
            if self.scaler_ is None:
                raise RuntimeError("Call fit() before predict().")
            X_arr = self.scaler_.transform(X_arr)

        raw = self.model_.predict(X_arr, batch_size=512, verbose=0)
        return raw.squeeze(axis=-1)

    def score(self, X, y) -> float:
        """Compute IPCW concordance index on (X, y)."""
        _, _, times, events = self._to_arrays(X, y)
        risk = self.predict(X)
        y_test_struct = self._make_structured(times, events)

        c_idx = concordance_index_ipcw(
            self._y_train_structured,
            y_test_struct,
            estimate=risk,
        )[0]
        return float(c_idx)


def run_architecture_search(X, y, configs: list[dict] | None = None, verbose: int = 0):
    """
    Train a set of DeepSurv architectures and return a list of summaries.
    """
    if configs is None:
        configs = get_architecture_configs()

    results = []
    for cfg in configs:
        model = DeepSurv(
            hidden_sizes=cfg["hidden_sizes"],
            dropout=cfg["dropout"],
            l2_reg=cfg["l2_reg"],
            lr=cfg["lr"],
            batch_size=cfg["batch_size"],
            epochs=cfg["epochs"],
            patience=cfg["patience"],
            verbose=verbose,
            activation=cfg.get("activation", "relu"),
        )
        model.fit(X, y)
        results.append({"name": cfg["name"], "config": cfg, "model": model})

    return results


def evaluate_architecture_sweep(
    X_train,
    X_test,
    y_train,
    y_test,
    configs: list[dict] | None = None,
    activation: str = "relu",
    verbose: int = 0,
):
    """
    Train all architecture configurations on the training set, evaluate them
    on the test set with the IPCW C-index, and return a ranked list.
    """
    if configs is None:
        configs = get_architecture_configs(activation=activation)

    results = []
    for cfg in configs:
        model = DeepSurv(
            hidden_sizes=cfg["hidden_sizes"],
            dropout=cfg["dropout"],
            l2_reg=cfg["l2_reg"],
            lr=cfg["lr"],
            batch_size=cfg["batch_size"],
            epochs=cfg["epochs"],
            patience=cfg["patience"],
            verbose=verbose,
            activation=cfg.get("activation", activation),
        )
        model.fit(X_train, y_train)
        test_score = model.score(X_test, y_test)
        results.append(
            {
                "name": cfg["name"],
                "activation": cfg.get("activation", activation),
                "hidden_sizes": cfg["hidden_sizes"],
                "dropout": cfg["dropout"],
                "l2_reg": cfg["l2_reg"],
                "lr": cfg["lr"],
                "batch_size": cfg["batch_size"],
                "epochs": cfg["epochs"],
                "test_ipcw_cindex": float(test_score),
                "model": model,
            }
        )

    results.sort(key=lambda item: item["test_ipcw_cindex"], reverse=True)
    return results


def print_architecture_sweep_results(
    X_train,
    X_test,
    y_train,
    y_test,
    configs: list[dict] | None = None,
    activation: str = "relu",
    verbose: int = 0,
):
    """Train and print the IPCW C-index comparison for all architectures."""
    results = evaluate_architecture_sweep(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        configs=configs,
        activation=activation,
        verbose=verbose,
    )

    summary = pd.DataFrame(results)
    summary = summary[[
        "name",
        "activation",
        "hidden_sizes",
        "test_ipcw_cindex",
    ]]
    print(summary.to_string(index=False))
    return results


def main(
    X_train,
    X_test,
    y_train,
    y_test,
    configs: list[dict] | None = None,
    activation: str = "relu",
    verbose: int = 0,
):
    """Notebook-friendly entry point for the full architecture sweep."""
    return print_architecture_sweep_results(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        configs=configs,
        activation=activation,
        verbose=verbose,
    )
