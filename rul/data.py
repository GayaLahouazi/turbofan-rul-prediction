"""Loading and preprocessing of the NASA C-MAPSS turbofan dataset.

Pipeline (same for every model, fitted on the training engines only):
    1. read the raw space-separated files
    2. compute the RUL label of every cycle (piecewise-linear, capped at `rul_cap`)
    3. drop sensors that carry no information (near-constant)
    4. min-max normalise with statistics computed on the training engines
    5. cut each engine trajectory into sliding windows of `window` cycles
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
SENSORS = [f"s{i}" for i in range(1, 22)]

# Original data: NASA Prognostics Center of Excellence (Saxena et al., 2008).
# The official NASA link changes regularly, so we use a public GitHub mirror
# of the unmodified files (sizes checked against the original archive).
MIRROR = "https://raw.githubusercontent.com/LahiruJayasinghe/RUL-Net/master/CMAPSSData/"


def download(subset: str = "FD001", data_dir: str | Path = "data") -> None:
    """Download train/test/RUL files of one C-MAPSS subset if they are missing."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in (f"train_{subset}.txt", f"test_{subset}.txt", f"RUL_{subset}.txt"):
        target = data_dir / name
        if not target.exists():
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(MIRROR + name, target)


def read_raw(subset: str = "FD001", data_dir: str | Path = "data"):
    """Return (train_df, test_df, test_rul) for one subset."""
    data_dir = Path(data_dir)
    read = lambda f: pd.read_csv(data_dir / f, sep=r"\s+", header=None, names=COLUMNS)
    train = read(f"train_{subset}.txt")
    test = read(f"test_{subset}.txt")
    test_rul = pd.read_csv(data_dir / f"RUL_{subset}.txt", header=None).values.ravel()
    return train, test, test_rul


def add_rul(df: pd.DataFrame, rul_cap: int = 125, final_rul=None) -> pd.DataFrame:
    """Add the 'rul' column.

    Training engines run until failure, so RUL = last_cycle - cycle.
    Test engines are cut before failure: `final_rul[i]` is the true RUL at
    their last recorded cycle, so RUL = last_cycle + final_rul - cycle.

    The label is capped at `rul_cap`: a brand-new engine shows no measurable
    degradation, so asking the model to distinguish RUL=300 from RUL=200
    only adds noise (standard 'piecewise-linear' target in the literature).
    """
    df = df.copy()
    last = df.groupby("unit")["cycle"].transform("max")
    rul = last - df["cycle"]
    if final_rul is not None:
        offset = df["unit"].map(dict(zip(sorted(df["unit"].unique()), final_rul)))
        rul = rul + offset
    df["rul"] = np.minimum(rul, rul_cap).astype(np.float32)
    df["rul_true"] = rul.astype(np.float32)  # uncapped, used for evaluation
    return df


def informative_sensors(train: pd.DataFrame, max_levels: int = 2) -> list[str]:
    """Drop sensors that take at most `max_levels` distinct values on the training set.

    In FD001, 7 sensors are constant or flip between two values (s1, s5, s6,
    s10, s16, s18, s19): they carry no degradation signal. A threshold on the
    standard deviation would be unit-dependent (s8 and s13 vary by only 0.07
    rpm around 2388 but do trend with wear), so we count distinct values instead.
    """
    return [s for s in SENSORS if train[s].nunique() > max_levels]


@dataclass
class Scaler:
    """Min-max scaler fitted on the training engines only (no test leakage)."""

    lo: np.ndarray
    hi: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Scaler":
        return cls(x.min(axis=0), x.max(axis=0))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.lo) / (self.hi - self.lo + 1e-12)).astype(np.float32)


def make_windows(df: pd.DataFrame, features: list[str], window: int):
    """All sliding windows of every engine -> X (n, window, n_features), y (n,), unit (n,)."""
    xs, ys, units = [], [], []
    for unit, g in df.groupby("unit"):
        values = g[features].to_numpy(np.float32)
        target = g["rul"].to_numpy(np.float32)
        values, target = _pad_front(values, target, window)
        for end in range(window, len(values) + 1):
            xs.append(values[end - window:end])
            ys.append(target[end - 1])
            units.append(unit)
    return np.stack(xs), np.array(ys), np.array(units)


def last_windows(df: pd.DataFrame, features: list[str], window: int):
    """Only the last window of every engine (official C-MAPSS test protocol)."""
    xs, ys, ys_true = [], [], []
    for _, g in df.groupby("unit"):
        values = g[features].to_numpy(np.float32)
        target = g["rul"].to_numpy(np.float32)
        values, target = _pad_front(values, target, window)
        xs.append(values[-window:])
        ys.append(target[-1])
        ys_true.append(g["rul_true"].iloc[-1])
    return np.stack(xs), np.array(ys), np.array(ys_true, dtype=np.float32)


def _pad_front(values: np.ndarray, target: np.ndarray, window: int):
    """Engines shorter than the window are padded by repeating their first cycle."""
    if len(values) >= window:
        return values, target
    missing = window - len(values)
    values = np.concatenate([np.repeat(values[:1], missing, axis=0), values])
    target = np.concatenate([np.repeat(target[:1], missing), target])
    return values, target


@dataclass
class Dataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray          # capped RUL (what the model is trained to predict)
    y_test_true: np.ndarray     # true RUL given by NASA (used for the reported metrics)
    features: list[str]
    train_df: pd.DataFrame
    test_df: pd.DataFrame


def prepare(subset: str = "FD001", data_dir: str | Path = "data", window: int = 30,
            rul_cap: int = 125, val_fraction: float = 0.2, seed: int = 0) -> Dataset:
    """Full preprocessing. Validation engines are held out *by engine*, never by row,
    otherwise windows of the same engine would leak between train and validation."""
    train, test, test_rul = read_raw(subset, data_dir)
    train = add_rul(train, rul_cap)
    test = add_rul(test, rul_cap, final_rul=test_rul)

    features = informative_sensors(train)
    scaler = Scaler.fit(train[features].to_numpy())
    train[features] = scaler.transform(train[features].to_numpy())
    test[features] = scaler.transform(test[features].to_numpy())

    rng = np.random.default_rng(seed)
    units = train["unit"].unique()
    val_units = rng.choice(units, size=int(len(units) * val_fraction), replace=False)
    is_val = train["unit"].isin(val_units)

    x_tr, y_tr, _ = make_windows(train[~is_val], features, window)
    x_va, y_va, _ = make_windows(train[is_val], features, window)
    x_te, y_te, y_te_true = last_windows(test, features, window)
    return Dataset(x_tr, y_tr, x_va, y_va, x_te, y_te, y_te_true, features, train, test)
