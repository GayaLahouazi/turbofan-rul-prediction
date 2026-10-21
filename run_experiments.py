"""Reproduce every number and figure of the README.

    python run_experiments.py --subset FD001 --seeds 0 1 2

Outputs: results/metrics_<subset>.json and figures/*.png
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, f1_score

from rul import bayes, data, decision
from rul.features import window_features
from rul.metrics import conformal_halfwidth, interval_coverage, nasa_score, rmse
from rul.models import build, mc_dropout_predict
from rul.train import fit, predict, set_seed

# colours (validated categorical palette, slots 1-3) and recessive ink
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.dpi": 130, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "lines.linewidth": 2,
})


def run(subset: str, seeds: list[int], window: int, epochs: int, out: Path, figs: Path):
    torch.set_num_threads(1)
    data.download(subset, "data")
    ds = data.prepare(subset, "data", window=window)
    print(f"{subset}: {len(ds.features)} informative sensors {ds.features}")
    print(f"train windows {ds.x_train.shape}, val {ds.x_val.shape}, test engines {len(ds.x_test)}")
    results = {"subset": subset, "window": window, "features": ds.features, "models": {}}

    # ---- 1. classical baseline -------------------------------------------------
    t0 = time.time()
    gbm = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0)
    gbm.fit(window_features(ds.x_train), ds.y_train)
    y_gbm = np.clip(gbm.predict(window_features(ds.x_test)), 0, None)
    results["models"]["gradient_boosting"] = {
        "rmse": [rmse(ds.y_test_true, y_gbm)], "score": [nasa_score(ds.y_test_true, y_gbm)],
        "train_seconds": time.time() - t0}

    # ---- 2. deep models, several seeds ----------------------------------------
    preds, trained, seed_preds = {}, {}, {}
    for name in ("cnn", "lstm"):
        r = {"rmse": [], "score": [], "train_seconds": 0.0, "epochs": []}
        seed_preds[name] = []
        for seed in seeds:
            set_seed(seed)
            model = build(name, len(ds.features), window)
            t0 = time.time()
            hist = fit(model, ds.x_train, ds.y_train, ds.x_val, ds.y_val, epochs=epochs)
            r["train_seconds"] += time.time() - t0
            r["epochs"].append(len(hist))
            y = np.clip(predict(model, ds.x_test), 0, None)
            r["rmse"].append(rmse(ds.y_test_true, y))
            r["score"].append(nasa_score(ds.y_test_true, y))
            seed_preds[name].append(y)
            print(f"{name} seed {seed}: RMSE {r['rmse'][-1]:.2f}  score {r['score'][-1]:.0f}")
            if seed == seeds[0]:
                preds[name], trained[name] = y, model
        results["models"][name] = r

    # simple ensemble: average of gradient boosting and the seed-averaged deep models
    y_ens = (y_gbm + np.mean(seed_preds["cnn"], 0) + np.mean(seed_preds["lstm"], 0)) / 3
    results["models"]["ensemble"] = {"rmse": [rmse(ds.y_test_true, y_ens)],
                                     "score": [nasa_score(ds.y_test_true, y_ens)], "train_seconds": 0.0}
    for name, r in results["models"].items():
        r["rmse_mean"], r["rmse_std"] = float(np.mean(r["rmse"])), float(np.std(r["rmse"]))
        r["score_mean"], r["score_std"] = float(np.mean(r["score"])), float(np.std(r["score"]))

    # ---- 3. uncertainty with MC dropout (best deep model) ----------------------
    best = min(("cnn", "lstm"), key=lambda m: results["models"][m]["rmse_mean"])
    samples = np.clip(mc_dropout_predict(trained[best], torch.from_numpy(ds.x_test), passes=200), 0, None)
    lo, hi = np.percentile(samples, [5, 95], axis=0)
    # split conformal calibration on the validation engines (never used for training)
    q = conformal_halfwidth(ds.y_val - predict(trained[best], ds.x_val), alpha=0.1)
    results["uncertainty"] = {
        "model": best, "passes": 200,
        "mc_dropout_coverage_90": interval_coverage(ds.y_test_true, lo, hi),
        "mc_dropout_mean_width": float(np.mean(hi - lo)),
        "conformal_halfwidth": q,
        "conformal_coverage_90": interval_coverage(ds.y_test_true, preds[best] - q, preds[best] + q),
        "conformal_width": 2 * q,
    }

    # ---- 4. maintenance decision -------------------------------------------------
    results["decision"] = {
        "horizon": decision.HORIZON, "c_prev": decision.C_PREV, "c_fail": decision.C_FAIL,
        "policies": decision.compare_policies(ds.y_test_true, preds[best], samples)}

    # ---- 5. Bayesian network diagnosis --------------------------------------------
    sensors = bayes.top_sensors(ds.train_df, ds.features, k=6)
    train_s = bayes.smooth(ds.train_df, sensors)
    test_s = bayes.smooth(ds.test_df, sensors)
    bn = bayes.HealthBN(sensors).fit(train_s)
    test_last = test_s.groupby("unit").tail(1)
    truth = bayes.health_state(test_last["rul_true"].to_numpy())
    pred_bn = bn.predict(test_last)
    pred_dl = bayes.health_state(preds[best])
    example = {s: "high" if s in ("s4", "s11") else None for s in sensors}
    example = {k: v for k, v in example.items() if v}
    results["bayesian_network"] = {
        "sensors": sensors, "arcs": bn.arcs(),
        "accuracy": accuracy_score(truth, pred_bn), "macro_f1": f1_score(truth, pred_bn, average="macro"),
        f"{best}_as_classifier_accuracy": accuracy_score(truth, pred_dl),
        f"{best}_as_classifier_macro_f1": f1_score(truth, pred_dl, average="macro"),
        "example_evidence": example, "example_posterior": bn.posterior(example) if example else None,
        "prior": bn.posterior({}),
    }

    out.mkdir(exist_ok=True)
    (out / f"metrics_{subset}.json").write_text(json.dumps(results, indent=2))
    figures(ds, results, preds, samples, trained[best], best, figs, subset)
    print(json.dumps({k: v for k, v in results.items() if k != "features"}, indent=2)[:3000])


def figures(ds, results, preds, samples, model, best, figs: Path, subset: str):
    figs.mkdir(exist_ok=True)

    # (a) test engines: prediction vs truth, sorted by true RUL, with 90% interval
    order = np.argsort(ds.y_test_true)
    lo, hi = np.percentile(samples, [5, 95], axis=0)
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    x = np.arange(len(order))
    ax.fill_between(x, lo[order], hi[order], color=BLUE, alpha=0.18, linewidth=0, label="90% interval (MC dropout)")
    ax.plot(x, preds[best][order], color=BLUE, marker="o", markersize=3, linewidth=1.2, label=f"{best.upper()} prediction")
    ax.plot(x, ds.y_test_true[order], color=INK, linewidth=1.6, label="true RUL")
    ax.set_xlabel("test engine (sorted by true RUL)")
    ax.set_ylabel("RUL (cycles)")
    ax.set_title(f"{subset} test set: predicted vs true remaining useful life", color=INK, loc="left")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(figs / f"test_predictions_{subset}.png"); plt.close(fig)

    # (b) prediction along the whole life of 3 test engines
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8), sharey=True)
    test = ds.test_df
    longest = test.groupby("unit")["cycle"].max().sort_values().index[-3:]
    for ax, unit in zip(axes, longest):
        g = test[test["unit"] == unit]
        xw, yw, _ = data.make_windows(g, ds.features, results["window"])
        yp = np.clip(predict(model, xw), 0, None)
        cyc = g["cycle"].to_numpy()[-len(yp):]
        ax.plot(cyc, np.minimum(g["rul_true"].to_numpy()[-len(yp):], 125), color=INK, linewidth=1.6, label="target (capped at 125)")
        ax.plot(cyc, yp, color=BLUE, linewidth=1.4, label=f"{best.upper()} prediction")
        ax.set_title(f"engine {unit}", color=INK, loc="left")
        ax.set_xlabel("cycle")
    axes[0].set_ylabel("RUL (cycles)")
    axes[0].legend(frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(figs / f"engine_trajectories_{subset}.png"); plt.close(fig)

    # (c) model comparison
    names = ["gradient_boosting", "cnn", "lstm", "ensemble"]
    labels = ["Gradient boosting\n(hand-crafted features)", "1D-CNN (PyTorch)", "LSTM (PyTorch)",
              "Ensemble (mean of 3)"]
    means = [results["models"][n]["rmse_mean"] for n in names]
    stds = [results["models"][n]["rmse_std"] for n in names]
    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    bars = ax.barh(labels, means, xerr=stds, color=[INK2, BLUE, BLUE, AQUA], height=0.5,
                   error_kw={"ecolor": INK, "elinewidth": 1, "capsize": 3})
    for b, m, s in zip(bars, means, stds):
        ax.text(b.get_width() + s + 0.3, b.get_y() + b.get_height() / 2, f"{m:.1f}", va="center", color=INK)
    ax.set_xlabel("test RMSE in cycles (lower is better)")
    ax.set_title(f"{subset}: model comparison", color=INK, loc="left")
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(figs / f"model_comparison_{subset}.png"); plt.close(fig)

    # (d) EDA: degradation trend of the 3 most informative sensors
    sensors = results["bayesian_network"]["sensors"][:3]
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.6), sharex=True)
    tr = ds.train_df
    for ax, s in zip(axes, sensors):
        for unit in tr["unit"].unique()[:25]:
            g = tr[tr["unit"] == unit]
            ax.plot(-g["rul_true"], g[s], color=BLUE, alpha=0.25, linewidth=0.8)
        ax.set_title(f"sensor {s} (normalised)", color=INK, loc="left")
        ax.set_xlabel("cycles before failure")
    fig.tight_layout(); fig.savefig(figs / f"sensor_trends_{subset}.png"); plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subset", default="FD001")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--epochs", type=int, default=60)
    a = p.parse_args()
    run(a.subset, a.seeds, a.window, a.epochs, Path("results"), Path("figures"))
