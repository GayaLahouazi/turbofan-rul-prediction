"""Training loop with early stopping on validation RMSE."""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .metrics import rmse


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def predict(model: nn.Module, x: np.ndarray, batch: int = 2048) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            out.append(model(torch.from_numpy(x[i:i + batch])).numpy())
    return np.concatenate(out)


def fit(model: nn.Module, x_train, y_train, x_val, y_val, epochs: int = 60,
        lr: float = 1e-3, batch_size: int = 256, patience: int = 10, verbose: bool = False):
    """Adam + MSE loss. Keeps the weights of the epoch with the best validation RMSE.
    Returns the history of validation RMSE (one value per epoch)."""
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
                        batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
    loss_fn = nn.MSELoss()

    best, best_state, wait, history = np.inf, None, 0, []
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        val = rmse(y_val, predict(model, x_val))
        sched.step(val)
        history.append(val)
        if verbose:
            print(f"epoch {epoch:3d}  val RMSE {val:6.2f}")
        if val < best - 1e-3:
            best, best_state, wait = val, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return history
