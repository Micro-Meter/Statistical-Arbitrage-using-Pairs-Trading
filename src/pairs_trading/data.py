"""Data loading and alignment utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _find_like(cols, keys):
    """Find a column whose name loosely matches one of the given keys."""
    cols_low = {c.lower(): c for c in cols}
    for k in keys:
        if k in cols_low:
            return cols_low[k]
    for c in cols:
        cl = c.lower().replace(" ", "").replace("_", "")
        for k in keys:
            if k.replace(" ", "").replace("_", "") in cl:
                return c
    return None


def read_price_csv(path: str, date_col: str = "Date", price_col: str = "Close") -> pd.Series:
    """Read a CSV of daily prices into a sorted, de-duplicated Series indexed by date."""
    if not Path(path).exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    dcol = date_col if date_col in df.columns else _find_like(df.columns, ["date"])
    pcol = price_col if price_col in df.columns else _find_like(
        df.columns, ["close", "adj close", "adj_close", "price"]
    )
    if dcol is None or pcol is None:
        raise ValueError(f"Couldn't infer Date/Close in {path}. Columns: {list(df.columns)}")
    s = pd.Series(df[pcol].values, index=pd.to_datetime(df[dcol]))
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="first")]
    return s


def align_prices(
    y: pd.Series,
    x: pd.Series,
    start: Optional[str] = None,
    end: Optional[str] = None,
):
    """Align two price series on a common business-day index, forward-filling gaps."""
    idx = pd.date_range(y.index.min(), x.index.max(), freq="B")
    y = y.reindex(idx).ffill()
    x = x.reindex(idx).ffill()
    if start:
        y = y[y.index >= pd.to_datetime(start)]
    if end:
        y = y[y.index <= pd.to_datetime(end)]
    x = x.reindex(y.index)
    return y, x


def make_synthetic_pair(n: int = 1500, seed: int = 42):
    """Generate a synthetic cointegrated pair for demos and tests."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    common = np.cumsum(rng.normal(0, 1, n))
    x = pd.Series(100 + common + np.cumsum(rng.normal(0, 0.3, n)), index=idx, name="X")
    spread_noise = np.zeros(n)
    for i in range(1, n):  # mean-reverting (OU-like) spread
        spread_noise[i] = 0.9 * spread_noise[i - 1] + rng.normal(0, 0.8)
    y = pd.Series(20 + 0.8 * x.values + spread_noise, index=idx, name="Y")
    return y, x
