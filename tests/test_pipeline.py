"""Fast unit tests on synthetic data (no download needed, run in CI)."""
import numpy as np
import pandas as pd
import pytest
import torch

from rul import data, decision
from rul.features import window_features
from rul.metrics import interval_coverage, nasa_score, rmse
from rul.models import build, mc_dropout_predict


def fake_engines(n_units=4, length=40):
    rows = []
    for u in range(1, n_units + 1):
        for c in range(1, length + u + 1):
            # sensors with i % 3 == 0 are constant (uninformative), the others drift with time
            rows.append([u, c, 0, 0, 100] + [5.0 if i % 3 == 0 else c * (i % 3) + np.random.rand()
                                             for i in range(21)])
    return pd.DataFrame(rows, columns=data.COLUMNS)


def test_rul_label_is_capped_and_decreasing():
    df = data.add_rul(fake_engines(), rul_cap=20)
    g = df[df.unit == 1]
    assert g["rul"].max() == 20
    assert g["rul"].iloc[-1] == 0                       # failure at the last training cycle
    assert (np.diff(g["rul_true"]) == -1).all()


def test_test_rul_uses_nasa_offset():
    df = data.add_rul(fake_engines(n_units=2), rul_cap=1000, final_rul=[7, 3])
    last = df.groupby("unit").tail(1)["rul_true"].tolist()
    assert last == [7, 3]


def test_windows_shapes_and_padding():
    df = data.add_rul(fake_engines(n_units=2, length=10))
    feats = ["s2", "s3"]
    x, y, units = data.make_windows(df, feats, window=30)   # engines shorter than window -> padded
    assert x.shape == (2, 30, 2) and y.shape == (2,)
    x_last, _, _ = data.last_windows(df, feats, window=5)
    assert x_last.shape == (2, 5, 2)


def test_constant_sensors_are_dropped():
    df = fake_engines()
    keep = data.informative_sensors(df)
    assert "s1" not in keep           # i % 3 == 0 -> almost constant
    assert "s2" in keep


def test_metrics():
    assert rmse([0, 0], [3, 4]) == pytest.approx(np.sqrt(12.5))
    assert nasa_score([10], [10]) == 0
    assert nasa_score([10], [20]) > nasa_score([10], [0])   # late predictions cost more
    assert interval_coverage([1, 5], [0, 0], [2, 2]) == 0.5


def test_window_features_slope():
    x = np.tile(np.arange(10, dtype=np.float32)[None, :, None], (1, 1, 1))
    f = window_features(x)
    assert f[0, 3] == pytest.approx(1.0)                   # slope of 0,1,...,9 is 1


@pytest.mark.parametrize("name", ["cnn", "lstm"])
def test_models_forward_and_mc_dropout(name):
    model = build(name, n_features=14)
    x = torch.randn(8, 30, 14)
    assert model(x).shape == (8,)
    samples = mc_dropout_predict(model, x, passes=5)
    assert samples.shape == (5, 8)
    assert samples.std(axis=0).mean() > 0                   # dropout really is active


def test_decision_rule():
    rul = np.array([5.0, 100.0])
    samples = np.array([[3.0, 90.0], [6.0, 110.0]])
    res = decision.compare_policies(rul, samples.mean(0), samples)
    assert res["oracle (knows the true RUL)"]["missed_failures"] == 0
    assert res["never maintain"]["missed_failures"] == 1
