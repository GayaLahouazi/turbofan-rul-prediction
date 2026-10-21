"""Deep learning models for RUL regression (PyTorch)."""
import torch
from torch import nn


class CNN1D(nn.Module):
    """1D convolutions along time detect local degradation patterns (Li et al., 2018).

    The feature maps are flattened, not averaged: *when* a pattern appears in
    the window matters (recent cycles say more about the current health).
    A first version with global average pooling reached only ~20 RMSE.
    """

    def __init__(self, n_features: int, window: int = 30, hidden: int = 32, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, hidden, kernel_size=9, padding=4), nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=9, padding=4), nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=9, padding=4), nn.ReLU(),
            nn.Conv1d(hidden, 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(8 * window, 64), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x):                      # x: (batch, window, features)
        return self.head(self.conv(x.transpose(1, 2))).squeeze(-1)


class LSTMRegressor(nn.Module):
    """Recurrent model: keeps a memory of the whole window."""

    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 64), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)  # last time step summarises the window


MODELS = {"cnn": CNN1D, "lstm": LSTMRegressor}


def build(name: str, n_features: int, window: int = 30) -> nn.Module:
    return CNN1D(n_features, window) if name == "cnn" else LSTMRegressor(n_features)


@torch.no_grad()
def mc_dropout_predict(model: nn.Module, x: torch.Tensor, passes: int = 100):
    """Monte-Carlo dropout (Gal & Ghahramani, 2016).

    Dropout is kept active at inference: each forward pass samples a slightly
    different network, and the spread of the predictions estimates the
    model's uncertainty. Returns an array of shape (passes, n).
    """
    model.train()  # activates dropout (no batch-norm in our models, so this is safe)
    preds = torch.stack([model(x) for _ in range(passes)])
    model.eval()
    return preds.numpy()
