import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_ipcw
from sksurv.util import Surv


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
    times  = tf.cast(y_true[:, 0], tf.float32)
    events = tf.cast(y_true[:, 1], tf.float32)
    log_risk = tf.squeeze(y_pred, axis=-1)          # (batch,)

    # Sort ascending by observed time so the risk set for patient i
    # is exactly patients i, i+1, …, n-1 (indices after sorting).
    sorted_idx = tf.argsort(times, direction="ASCENDING")
    times     = tf.gather(times,    sorted_idx)
    events    = tf.gather(events,   sorted_idx)
    log_risk  = tf.gather(log_risk, sorted_idx)

    # Reverse cumulative sum of exp(log_risk) gives Σ_{j ≥ i} exp(h_j)
    # i.e. the Breslow approximation of the risk-set sum at each time.
    exp_risk       = tf.exp(log_risk)
    risk_set_sum   = tf.cumsum(exp_risk, reverse=True)          # (batch,)
    log_risk_set   = tf.math.log(risk_set_sum + 1e-8)           # numerical safety

    # Partial log-likelihood contribution (only uncensored patients)
    ll_per_patient = events * (log_risk - log_risk_set)
    n_events       = tf.reduce_sum(events) + 1e-8

    return -tf.reduce_sum(ll_per_patient) / n_events


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def _build_network(
    n_features:   int,
    hidden_sizes: list[int],
    dropout:      float,
    l2_reg:       float,
) -> keras.Model:
    """
    Build the DeepSurv feed-forward network.

    Each hidden block:   Dense(units) → BatchNorm → ReLU → Dropout
    Output block:        Dense(1, activation='linear')
    """
    reg = keras.regularizers.l2(l2_reg)

    inp = keras.Input(shape=(n_features,), name="features")
    x   = inp

    for i, units in enumerate(hidden_sizes):
        x = keras.layers.Dense(
            units,
            use_bias=False,          # bias absorbed by BatchNorm
            kernel_regularizer=reg,
            name=f"fc_{i}",
        )(x)
        x = keras.layers.BatchNormalization(name=f"bn_{i}")(x)
        x = keras.layers.Activation("relu", name=f"relu_{i}")(x)
        x = keras.layers.Dropout(dropout, name=f"drop_{i}")(x)

    out = keras.layers.Dense(1, activation="linear", name="log_risk")(x)

    return keras.Model(inputs=inp, outputs=out, name="DeepSurv")


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DeepSurv:
    """
    DeepSurv: Cox Proportional Hazards Deep Neural Network.

    Parameters
    ----------
    hidden_sizes : list of int
        Number of units in each hidden layer.  Default matches the thesis
        specification: three layers of width 64.
    dropout : float
        Dropout probability applied after each hidden ReLU block.
    l2_reg : float
        L2 regularisation strength λ on all weight matrices.
    lr : float
        Initial Adam learning rate.
    batch_size : int
        Mini-batch size.  Batches are drawn with stratified time sampling to
        ensure non-trivial risk sets.
    epochs : int
        Maximum training epochs.
    patience : int
        Early-stopping patience (epochs without improvement in validation
        IPCW C-index).
    val_fraction : float
        Fraction of training data held out for early stopping.
    random_state : int or None
        Seed for reproducibility.
    verbose : int
        0 = silent, 1 = epoch bar, 2 = one line per epoch.
    """

    def __init__(
        self,
        hidden_sizes: list[int] = (64, 64, 64),
        dropout:      float     = 0.3,
        l2_reg:       float     = 1e-4,
        lr:           float     = 1e-3,
        batch_size:   int       = 128,
        epochs:       int       = 300,
        patience:     int       = 20,
        val_fraction: float     = 0.15,
        random_state: int | None = 42,
        verbose:      int       = 1,
    ):
        self.hidden_sizes  = list(hidden_sizes)
        self.dropout       = dropout
        self.l2_reg        = l2_reg
        self.lr            = lr
        self.batch_size    = batch_size
        self.epochs        = epochs
        self.patience      = patience
        self.val_fraction  = val_fraction
        self.random_state  = random_state
        self.verbose       = verbose

        self.model_   : keras.Model | None   = None
        self.scaler_  : StandardScaler | None = None
        self.history_ : dict | None           = None
        self._y_train_structured              = None   # kept for IPCW score()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_arrays(X: pd.DataFrame | np.ndarray,
                   y: pd.DataFrame | None = None):
        """Convert inputs to plain numpy arrays."""
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        if y is None:
            return X_arr.astype(np.float32), None, None

        if isinstance(y, pd.DataFrame):
            times  = y["OS_YEARS"].values.astype(np.float32)
            events = y["OS_STATUS"].values.astype(np.float32)
        else:
            # Accept structured array (sksurv convention)
            times  = y["time"].astype(np.float32)
            events = y["event"].astype(np.float32)

        y_arr = np.stack([times, events], axis=1)   # (n, 2)
        return X_arr.astype(np.float32), y_arr, times, events

    @staticmethod
    def _make_structured(times: np.ndarray, events: np.ndarray) -> np.ndarray:
        """Build a sksurv-compatible structured array."""
        return Surv.from_arrays(event=events.astype(bool), time=times)

    def _stratified_split(self, X, y_arr):
        """
        Hold out val_fraction of data, stratified by event status so that
        the validation set contains enough events for IPCW evaluation.
        """
        rng     = np.random.default_rng(self.random_state)
        n       = len(X)
        events  = y_arr[:, 1].astype(bool)

        # Separate event and censored indices
        ev_idx  = np.where(events)[0]
        cen_idx = np.where(~events)[0]
        rng.shuffle(ev_idx)
        rng.shuffle(cen_idx)

        n_val_ev  = max(1, int(len(ev_idx)  * self.val_fraction))
        n_val_cen = max(1, int(len(cen_idx) * self.val_fraction))

        val_idx   = np.concatenate([ev_idx[:n_val_ev], cen_idx[:n_val_cen]])
        train_idx = np.concatenate([ev_idx[n_val_ev:], cen_idx[n_val_cen:]])

        return (
            X[train_idx], y_arr[train_idx],
            X[val_idx],   y_arr[val_idx],
        )

    def _make_batches(self, X, y_arr):
        """
        Yield mini-batches.  Each batch is sorted by time internally so that
        the loss's cumsum produces correct risk sets within the batch.
        The global sort happens inside the loss itself; this sort makes the
        batches locally coherent and improves gradient signal.
        """
        n   = len(X)
        rng = np.random.default_rng(self.random_state)
        idx = rng.permutation(n)

        for start in range(0, n, self.batch_size):
            batch_idx = idx[start : start + self.batch_size]
            yield X[batch_idx], y_arr[batch_idx]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """
        Fit DeepSurv on training data.

        Parameters
        ----------
        X : DataFrame or ndarray, shape (n_samples, n_features)
        y : DataFrame with columns OS_YEARS, OS_STATUS
            or sksurv structured array with fields 'time', 'event'

        Returns
        -------
        self
        """
        if self.random_state is not None:
            tf.random.set_seed(self.random_state)
            np.random.seed(self.random_state)

        X_arr, y_arr, times, events = self._to_arrays(X, y)

        # Standardise features
        self.scaler_ = StandardScaler()
        X_arr = self.scaler_.fit_transform(X_arr)

        # Train / validation split
        X_tr, y_tr, X_val, y_val = self._stratified_split(X_arr, y_arr)

        # Store structured y for IPCW in score()
        self._y_train_structured = self._make_structured(
            y_tr[:, 0], y_tr[:, 1]
        )

        # Build model
        self.model_ = _build_network(
            n_features   = X_arr.shape[1],
            hidden_sizes = self.hidden_sizes,
            dropout      = self.dropout,
            l2_reg       = self.l2_reg,
        )
        self.model_.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.lr),
            loss=cox_partial_likelihood_loss,
        )

        if self.verbose:
            self.model_.summary()

        # Training loop with manual early stopping on val IPCW C-index
        best_cindex    = -np.inf
        best_weights   = None
        patience_count = 0

        train_losses = []
        val_cindices = []

        y_val_struct = self._make_structured(y_val[:, 0], y_val[:, 1])

        for epoch in range(self.epochs):
            # --- one epoch of mini-batch gradient updates ---
            epoch_loss = []
            for X_batch, y_batch in self._make_batches(X_tr, y_tr):
                X_t = tf.constant(X_batch, dtype=tf.float32)
                y_t = tf.constant(y_batch, dtype=tf.float32)
                loss_val = self.model_.train_on_batch(X_t, y_t)
                epoch_loss.append(float(loss_val))

            mean_loss = float(np.mean(epoch_loss))
            train_losses.append(mean_loss)

            # --- validation IPCW C-index ---
            val_risk  = self.predict(X_val, already_scaled=True)
            try:
                c_idx = concordance_index_ipcw(
                    self._y_train_structured, y_val_struct,
                    estimate=val_risk,
                )[0]
            except Exception:
                c_idx = np.nan

            val_cindices.append(c_idx)

            if self.verbose:
                print(
                    f"Epoch {epoch+1:>4}/{self.epochs}  "
                    f"loss={mean_loss:.4f}  val_cindex={c_idx:.4f}"
                )

            # --- early stopping ---
            if not np.isnan(c_idx) and c_idx > best_cindex:
                best_cindex  = c_idx
                best_weights = self.model_.get_weights()
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= self.patience:
                    if self.verbose:
                        print(
                            f"Early stopping at epoch {epoch+1}. "
                            f"Best val C-index: {best_cindex:.4f}"
                        )
                    break

        # Restore best weights
        if best_weights is not None:
            self.model_.set_weights(best_weights)

        self.history_ = {
            "train_loss": train_losses,
            "val_cindex": val_cindices,
        }

        return self

    def predict(self, X, already_scaled: bool = False) -> np.ndarray:
        """
        Return predicted log-risk scores.

        Higher score → higher predicted hazard → shorter expected survival.

        Parameters
        ----------
        X : DataFrame or ndarray, shape (n_samples, n_features)
        already_scaled : bool
            Internal flag; leave as False when calling from outside.

        Returns
        -------
        risk_scores : ndarray of shape (n_samples,)
        """
        X_arr = X if isinstance(X, np.ndarray) else (
            X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        )
        X_arr = X_arr.astype(np.float32)

        if not already_scaled:
            if self.scaler_ is None:
                raise RuntimeError("Call fit() before predict().")
            X_arr = self.scaler_.transform(X_arr)

        raw = self.model_.predict(X_arr, batch_size=512, verbose=0)  # (n, 1)
        return raw.squeeze(axis=-1)                                    # (n,)

    def score(self, X, y) -> float:
        """
        Compute IPCW concordance index on (X, y).

        Parameters
        ----------
        X : DataFrame or ndarray
        y : DataFrame with OS_YEARS, OS_STATUS  or sksurv structured array

        Returns
        -------
        float  IPCW C-index  (higher is better; 0.5 = random)
        """
        _, _, times, events = self._to_arrays(X, y)
        risk    = self.predict(X)
        y_test_struct = self._make_structured(times, events)

        c_idx = concordance_index_ipcw(
            self._y_train_structured,
            y_test_struct,
            estimate=risk,
        )[0]
        return float(c_idx)

    def plot_history(self):
        """Plot training loss and validation C-index curves."""
        import matplotlib.pyplot as plt

        if self.history_ is None:
            raise RuntimeError("Call fit() first.")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(self.history_["train_loss"])
        axes[0].set_title("Training loss (neg. partial log-likelihood)")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(self.history_["val_cindex"])
        axes[1].set_title("Validation IPCW C-index")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("C-index")
        axes[1].axhline(0.5, color="gray", linestyle="--", label="random")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
