"""Hedge ratio estimation and cointegration testing."""
from __future__ import annotations

import numpy as np
import pandas as pd


def hedge_ratio_ols(y: pd.Series, x: pd.Series) -> float:
    """Static OLS hedge ratio (beta of y on x, with intercept)."""
    X = np.vstack([np.ones(len(x)), x.values]).T
    b = np.linalg.lstsq(X, y.values, rcond=None)[0][1]
    return float(b)


def rolling_hedge_ratio(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    """Rolling OLS hedge ratio over a fixed lookback window."""
    betas, idxs = [], []
    for i in range(window, len(y) + 1):
        yy = y.iloc[i - window:i]
        xx = x.iloc[i - window:i]
        betas.append(hedge_ratio_ols(yy, xx))
        idxs.append(y.index[i - 1])
    return pd.Series(betas, index=idxs).reindex(y.index).ffill()


def calc_spread(y: pd.Series, x: pd.Series, beta) -> pd.Series:
    """Spread y - beta*x, with beta either scalar or a series."""
    if isinstance(beta, (float, int)):
        return y - float(beta) * x
    return y - beta * x


def engle_granger_adf(y: pd.Series, x: pd.Series):
    """Engle–Granger step: OLS residual + ADF test.

    Returns (adf_stat, p_value, beta). Requires statsmodels.
    """
    from statsmodels.tsa.stattools import adfuller

    beta = hedge_ratio_ols(y, x)
    resid = (y - beta * x).dropna()
    stat, pval = adfuller(resid.values, autolag="AIC")[:2]
    return float(stat), float(pval), beta


def johansen_test(prices: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1):
    """Johansen multivariate cointegration test on a panel of price series.

    Returns the raw statsmodels result (trace statistics vs. critical values).
    """
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    panel = prices.dropna()
    return coint_johansen(panel.values, det_order, k_ar_diff)


def half_life(spread: pd.Series) -> float:
    """Ornstein–Uhlenbeck half-life of mean reversion, estimated by AR(1) fit."""
    s = spread.dropna()
    lag = s.shift(1).dropna()
    ds = (s - s.shift(1)).dropna()
    lag = lag.loc[ds.index]
    X = np.vstack([np.ones(len(lag)), lag.values]).T
    b = np.linalg.lstsq(X, ds.values, rcond=None)[0][1]
    if b >= 0:
        return float("inf")
    return float(-np.log(2) / b)
