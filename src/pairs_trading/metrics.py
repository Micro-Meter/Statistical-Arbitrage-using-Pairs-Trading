"""Performance metrics: CAGR, Sharpe, volatility, max drawdown."""
from __future__ import annotations

import numpy as np
import pandas as pd


def metrics(daily: pd.DataFrame, ann: int = 252) -> dict:
    """Robust performance metrics from a backtest 'daily' frame.

    - Cleans NaNs/Infs
    - CAGR via compounded returns
    - Max drawdown (%) from the equity curve, NaN-safe
    """
    ret = daily["daily_ret"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mu = float(ret.mean()) * ann
    sd = float(ret.std(ddof=1)) * (ann ** 0.5)
    sharpe = mu / sd if sd > 0 else float("nan")

    eq = daily["equity"].astype(float).replace([np.inf, -np.inf], np.nan).ffill().bfill()
    if len(eq) >= 2 and eq.iloc[0] > 0:
        cagr = (eq.iloc[-1] / eq.iloc[0]) ** (ann / max(1, len(eq))) - 1.0
    else:
        cagr = float("nan")

    roll_max = eq.cummax()
    dd_pct = eq / roll_max - 1.0
    mdd = float(dd_pct.min()) if dd_pct.notna().any() else float("nan")

    return {
        "cagr": float(cagr),
        "annualized_return": float(mu),
        "annualized_volatility": float(sd),
        "sharpe_ratio": float(sharpe),
        "max_drawdown_pct": mdd,
    }


def sortino_ratio(ret: pd.Series, ann: int = 252) -> float:
    """Sortino ratio using downside deviation."""
    r = pd.Series(ret).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    downside = r[r < 0]
    dd = float(downside.std(ddof=1)) * (ann ** 0.5)
    mu = float(r.mean()) * ann
    return mu / dd if dd > 0 else float("nan")


def stationary_bootstrap_sharpe(ret, B: int = 1000, avg_block: int = 20, seed: int = 42):
    """95% confidence interval for the Sharpe ratio via stationary bootstrap."""
    r = pd.Series(ret).dropna().values
    if r.size < 10:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = r.size
    p = 1.0 / avg_block
    sharpes = []
    for _ in range(B):
        idx = np.empty(n, dtype=int)
        idx[0] = rng.integers(n)
        for t in range(1, n):
            idx[t] = rng.integers(n) if rng.random() < p else (idx[t - 1] + 1) % n
        rb = r[idx]
        sd = rb.std(ddof=1)
        sharpes.append(np.sqrt(252) * rb.mean() / sd if sd > 0 else np.nan)
    lo, hi = np.nanpercentile(sharpes, [2.5, 97.5])
    return (float(lo), float(hi))
