"""From prediction to decision: when should an engine go to maintenance?

Decision problem (one decision per test engine, at its last observed cycle):
    - maintain now        -> cost C_PREV (planned, cheap)
    - keep flying H cycles -> cost C_FAIL if the engine fails within H cycles, else 0

Expected cost of 'keep flying' = C_FAIL * P(RUL <= H).
Bayes-optimal rule: maintain  <=>  P(RUL <= H) >= C_PREV / C_FAIL.

A point prediction cannot estimate P(RUL <= H); the Monte-Carlo dropout
samples can. We compare several policies on the 100 test engines.
"""
import numpy as np

C_PREV, C_FAIL, HORIZON = 1.0, 10.0, 20


def policy_cost(maintain: np.ndarray, rul_true: np.ndarray, horizon: int = HORIZON,
                c_prev: float = C_PREV, c_fail: float = C_FAIL) -> dict:
    fails = rul_true <= horizon
    missed = (~maintain) & fails
    return {
        "total_cost": float(maintain.sum() * c_prev + missed.sum() * c_fail),
        "maintenances": int(maintain.sum()),
        "missed_failures": int(missed.sum()),
        "unnecessary_maintenances": int((maintain & ~fails).sum()),
    }


def compare_policies(rul_true: np.ndarray, point_pred: np.ndarray, samples: np.ndarray,
                     horizon: int = HORIZON) -> dict:
    """samples: (passes, n) MC-dropout predictions."""
    threshold = C_PREV / C_FAIL
    p_fail = (samples <= horizon).mean(axis=0)
    policies = {
        "never maintain": np.zeros_like(rul_true, dtype=bool),
        "always maintain": np.ones_like(rul_true, dtype=bool),
        "point estimate (pred <= H)": point_pred <= horizon,
        "uncertainty-aware (P(fail) >= c_prev/c_fail)": p_fail >= threshold,
        "oracle (knows the true RUL)": rul_true <= horizon,
    }
    return {name: policy_cost(m, rul_true, horizon) for name, m in policies.items()}
