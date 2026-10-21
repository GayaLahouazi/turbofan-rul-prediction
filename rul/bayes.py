"""Interpretable health diagnosis with a Bayesian network (pyAgrum).

The deep models give a number (the RUL) but no explanation. A Bayesian
network gives a *probabilistic diagnosis* that an engineer can read:
    P(health state | what the sensors show)
and it still works when some sensors are missing (just leave them out of
the evidence).

Steps:
    1. health state from the true RUL: healthy / degrading / critical
    2. smooth each sensor (rolling mean) and discretise it into low/mid/high
       using quantiles computed on the training engines
    3. learn the network structure + parameters from data (greedy hill
       climbing, BIC score), with the constraint that `health` is a root cause
    4. exact inference (Lazy Propagation) on the test engines
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import pyagrum as gum
except ImportError:  # pyAgrum < 2.0
    import pyAgrum as gum

STATES = ["critical", "degrading", "healthy"]
BINS = ["low", "mid", "high"]


def health_state(rul_true: np.ndarray, critical: int = 30, degrading: int = 90) -> np.ndarray:
    return np.where(rul_true <= critical, "critical",
                    np.where(rul_true <= degrading, "degrading", "healthy"))


def top_sensors(train: pd.DataFrame, features: list[str], k: int = 6) -> list[str]:
    """The k sensors most correlated (in absolute value) with the RUL."""
    corr = train[features + ["rul_true"]].corr()["rul_true"].drop("rul_true").abs()
    return corr.sort_values(ascending=False).index[:k].tolist()


def smooth(df: pd.DataFrame, sensors: list[str], span: int = 10) -> pd.DataFrame:
    out = df.copy()
    out[sensors] = df.groupby("unit")[sensors].transform(lambda s: s.rolling(span, min_periods=1).mean())
    return out


class HealthBN:
    def __init__(self, sensors: list[str]):
        self.sensors = sensors
        self.cuts: dict[str, np.ndarray] = {}
        self.bn = None

    def _discretise(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for s in self.sensors:
            idx = np.digitize(df[s].to_numpy(), self.cuts[s])
            out[s] = np.array(BINS)[idx]
        return out

    def fit(self, train: pd.DataFrame) -> "HealthBN":
        for s in self.sensors:
            self.cuts[s] = np.quantile(train[s], [1 / 3, 2 / 3])
        table = self._discretise(train)
        table["health"] = health_state(train["rul_true"].to_numpy())

        template = gum.BayesNet()
        template.add(gum.LabelizedVariable("health", "health", STATES))
        for s in self.sensors:
            template.add(gum.LabelizedVariable(s, s, BINS))

        learner = gum.BNLearner(table, template)
        learner.useGreedyHillClimbing()
        learner.useScoreBIC()
        learner.useSmoothingPrior(1.0)       # Laplace smoothing, avoids zero probabilities
        learner.addNoParentNode("health")    # health is the cause, sensors are symptoms
        self.bn = learner.learnBN()
        return self

    def posterior(self, evidence: dict[str, str]) -> dict[str, float]:
        ie = gum.LazyPropagation(self.bn)
        ie.setEvidence(evidence)
        ie.makeInference()
        p = ie.posterior("health")
        return {state: float(p[i]) for i, state in enumerate(STATES)}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        table = self._discretise(df)
        ie = gum.LazyPropagation(self.bn)
        preds = []
        for _, row in table.iterrows():
            ie.setEvidence({s: row[s] for s in self.sensors})
            ie.makeInference()
            p = ie.posterior("health").toarray()
            preds.append(STATES[int(np.argmax(p))])
        return np.array(preds)

    def arcs(self) -> list[tuple[str, str]]:
        return [(self.bn.variable(a).name(), self.bn.variable(b).name()) for a, b in self.bn.arcs()]
