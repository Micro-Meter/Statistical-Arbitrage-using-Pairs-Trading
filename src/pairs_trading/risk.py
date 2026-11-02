"""Risk management: volatility targeting, VaR and Expected Shortfall."""
from __future__ import annotations

import numpy as np
import pandas as pd


def volatility_targeting(
    returns: pd.Series,
    target_vol: float = 0.15,
    lookback: int = 60,
    cap: float = 5.0,
):
    """Scale daily returns so rolling realized vol tracks an annualized target.

    The scaler is lagged one day to avoid look-ahead. Returns (scaled_returns, scaler).
    """
    vol_ann = returns.rolling(lookback).std(ddof=1) * np.sqrt(252)
    scaler = (target_vol / vol_ann).clip(upper=cap)
    scaled = returns * scaler.shift(1)
    return scaled.fillna(0.0), scaler


def var_es(returns: pd.Series, alpha: float = 0.05):
    """Historical Value-at-Risk and Expected Shortfall at level alpha.

    Returns (VaR, ES) as positive loss numbers.
    """
    r = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return float("nan"), float("nan")
    var = -float(np.quantile(r.values, alpha))
    tail = r[r <= -var]
    es = -float(tail.mean()) if not tail.empty else var
    return var, es


def stress_window_metrics(returns: pd.Series, start: str, end: str, ann: int = 252) -> dict:
    """Performance metrics restricted to a stress window (e.g. Covid-19 2020)."""
    s = pd.Series(returns).dropna().loc[start:end]
    if s.empty:
        return {"CAGR": np.nan, "Vol": np.nan, "Sharpe": np.nan, "MaxDD": np.nan}
    eq = (1 + s).cumprod()
    n = len(s)
    cagr = eq.iloc[-1] ** (ann / n) - 1.0
    vol = s.std(ddof=1) * np.sqrt(ann)
    sharpe = (s.mean() * ann) / vol if vol > 0 else np.nan
    mdd = float((eq / eq.cummax() - 1.0).min())
    return {"CAGR": float(cagr), "Vol": float(vol), "Sharpe": float(sharpe), "MaxDD": mdd}
