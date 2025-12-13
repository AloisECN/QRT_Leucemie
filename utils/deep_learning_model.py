"""DeepSurv-style neural network for tabular survival data.

This module provides a small PyTorch implementation that trains a
feed-forward network using the Cox partial log-likelihood (DeepSurv).

Usage (high-level):
  from utils.deep_learning_model import DeepSurvTrainer
  trainer = DeepSurvTrainer()
  model = trainer.fit(X_train, y_train_df, X_val, y_val_df)
  risk_scores = trainer.predict_risk(model, X_test)

The implementation performs full-batch updates (suitable for small-to-medium datasets).
"""


from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from sksurv.metrics import concordance_index_censored


class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, times: np.ndarray, events: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.times = torch.from_numpy(times.astype(np.float32))
        self.events = torch.from_numpy(events.astype(np.int64))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.times[idx], self.events[idx]


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.2,
        use_layer_norm: bool = False,
        use_residuals: bool = False,
    ):
        super().__init__()
        self.use_residuals = use_residuals
        self.layers = nn.ModuleList()
        in_dim = input_dim
        for h in hidden_dims:
            block = nn.ModuleList()
            block.append(nn.Linear(in_dim, h))
            if use_layer_norm:
                block.append(nn.LayerNorm(h))
            else:
                block.append(nn.BatchNorm1d(h))
            block.append(nn.ReLU())
            block.append(nn.Dropout(dropout))
            self.layers.append(block)
            in_dim = h
        # final linear to scalar output
        self.final = nn.Linear(in_dim, 1)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        out = x
        for i, block in enumerate(self.layers):
            lin = block[0]
            norm = block[1]
            act = block[2]
            drop = block[3]

            y = lin(out)
            # batchnorm/LayerNorm expect 2D -> keep as-is
            y = norm(y)
            y = act(y)
            y = drop(y)

            # apply residual connection when dims match and enabled
            if self.use_residuals and y.shape[-1] == out.shape[-1]:
                out = out + y
            else:
                out = y
        return self.final(out).squeeze(-1)


def cox_ph_loss(preds: torch.Tensor, times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    """Negative Cox partial log-likelihood (vectorized).

    preds: hazard/risk score (higher -> larger hazard)
    times: follow-up time
    events: 0/1 event indicator
    """
    # ensure float tensors
    preds = preds.reshape(-1)
    times = times.reshape(-1)
    events = events.reshape(-1).float()

    # sort by descending time so risk set is cumulative
    order = torch.argsort(times, descending=True)
    preds_ord = preds[order]
    events_ord = events[order]

    exp_preds = torch.exp(preds_ord)
    # cumulative sum of exp_preds over descending times gives denominator for each item
    denom = torch.cumsum(exp_preds, dim=0)

    # for each observed event, contribution is (pred - log(sum_{j in R_i} exp(pred_j)))
    # select only event rows
    observed_preds = preds_ord * events_ord
    observed_log_denom = torch.log(denom) * events_ord
    # sum over all events
    partial_lik = torch.sum(observed_preds - observed_log_denom)
    # negative average
    n_events = torch.sum(events_ord)
    if n_events <= 0:
        return torch.tensor(0.0, device=preds.device)
    loss = -partial_lik / n_events
    return loss


class GatedResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.fc1 = nn.Linear(dim, 2 * dim)  # for GLU
        self.fc2 = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

        # learnable residual scale
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        h = self.norm(x)

        # GLU
        v, g = self.fc1(h).chunk(2, dim=-1)
        h = v * torch.sigmoid(g)

        h = self.fc2(h)
        h = self.dropout(h)

        return x + self.alpha * h


class DeepMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        depth: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.blocks = nn.ModuleList(
            [GatedResidualBlock(hidden_dim, dropout) for _ in range(depth)]
        )

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        return self.head(x).squeeze(-1)


class DeepSurvTrainer:
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _to_numpy(self, df_or_arr):
        if hasattr(df_or_arr, "values"):
            return df_or_arr.values
        return np.asarray(df_or_arr)

    def fit(
        self,
        X_train,
        y_train_df,
        X_val=None,
        y_val_df=None,
        hidden_dims=(256, 128, 64, 32),
        dropout=0.2,
        lr=1e-3,
        weight_decay=1e-4,
        n_epochs=2000,
        patience=20,
        verbose=True,
        use_layer_norm: bool = False,
        use_residuals: bool = False,
        # new options for DeepMLP
        model_type: str = "mlp",        # "mlp" (original) or "deepmlp"
        deepmlp_hidden_dim: int = 256,
        deepmlp_depth: int = 6,
    ) -> Tuple[nn.Module, dict]:
        """Train a DeepSurv MLP and return the trained model and history.

        model_type: choose "mlp" for original MLP or "deepmlp" for the gated residual deep model.
        """
        X_train_np = self._to_numpy(X_train).astype(np.float32)
        times_train = self._to_numpy(y_train_df["OS_YEARS"]).astype(np.float32)
        events_train = self._to_numpy(y_train_df["OS_STATUS"]).astype(int)

        if X_val is not None and y_val_df is not None:
            X_val_np = self._to_numpy(X_val).astype(np.float32)
            times_val = self._to_numpy(y_val_df["OS_YEARS"]).astype(np.float32)
            events_val = self._to_numpy(y_val_df["OS_STATUS"]).astype(int)
        else:
            X_val_np = times_val = events_val = None

        input_dim = X_train_np.shape[1]

        # choose model class
        if model_type == "deepmlp":
            model = DeepMLP(
                input_dim=input_dim,
                hidden_dim=deepmlp_hidden_dim,
                depth=deepmlp_depth,
                dropout=dropout,
            ).to(self.device)
        else:
            # original MLP expects hidden_dims tuple
            model = MLP(
                input_dim,
                hidden_dims=tuple(hidden_dims),
                dropout=dropout,
                use_layer_norm=use_layer_norm,
                use_residuals=use_residuals,
            ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_loss = float("inf")
        best_model_state = None
        best_cindex = -1.0
        wait = 0

        history = {"train_loss": [], "val_loss": [], "val_cindex": []}

        # full-batch training (common for Cox partial likelihood)
        X_train_t = torch.from_numpy(X_train_np).to(self.device)
        times_train_t = torch.from_numpy(times_train).to(self.device)
        events_train_t = torch.from_numpy(events_train).to(self.device)

        if X_val_np is not None:
            X_val_t = torch.from_numpy(X_val_np).to(self.device)

        for epoch in range(1, n_epochs + 1):
            model.train()
            optimizer.zero_grad()
            preds = model(X_train_t)
            loss = cox_ph_loss(preds, times_train_t, events_train_t)
            loss.backward()
            optimizer.step()

            history["train_loss"].append(float(loss.detach().cpu().numpy()))

            val_loss = None
            val_cindex = None
            if X_val_np is not None:
                model.eval()
                with torch.no_grad():
                    preds_val = model(X_val_t).detach().cpu().numpy()
                # compute c-index on validation set (use negative preds so larger risk->smaller survival time ordering)
                val_cindex = concordance_index_censored(
                    y_val_df["OS_STATUS"].astype(bool).values,
                    y_val_df["OS_YEARS"].values,
                    -preds_val,
                )[0]
                history["val_cindex"].append(float(val_cindex))

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch:4d}: train_loss={history['train_loss'][-1]:.5f}", end="")
                if val_cindex is not None:
                    print(f", val_cindex={val_cindex:.4f}")
                else:
                    print("")

            # early stopping based on validation c-index if available, otherwise train loss
            metric_for_best = val_cindex if val_cindex is not None else -history["train_loss"][-1]

            improved = False
            if val_cindex is not None:
                if val_cindex > best_cindex:
                    best_cindex = val_cindex
                    best_model_state = {"model": model.state_dict(), "epoch": epoch}
                    improved = True
            else:
                if history["train_loss"][-1] < best_loss:
                    best_loss = history["train_loss"][-1]
                    best_model_state = {"model": model.state_dict(), "epoch": epoch}
                    improved = True

            if not improved:
                wait += 1
            else:
                wait = 0

            if wait >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} (wait={wait})")
                break

        # restore best model state
        if best_model_state is not None:
            model.load_state_dict(best_model_state["model"])

        return model, history

    def predict_risk(self, model: nn.Module, X) -> np.ndarray:
        model.to(self.device)
        model.eval()
        X_np = self._to_numpy(X).astype(np.float32)
        X_t = torch.from_numpy(X_np).to(self.device)
        with torch.no_grad():
            preds = model(X_t).detach().cpu().numpy()
        # return raw risk scores (higher -> higher hazard)
        return preds


__all__ = ["DeepSurvTrainer"]
