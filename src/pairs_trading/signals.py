"""Z-score standardization and band-based trading signals."""
from __future__ import annotations

import pandas as pd


def zscore(s: pd.Series, window: int) -> pd.Series:
    """Rolling z-score of a series."""
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std(ddof=1)
    return (s - mu) / sd


def generate_signals(z: pd.Series, entry: float, exit: float) -> pd.DataFrame:
    """Stateful band signals: enter when |z| > entry, exit when |z| < exit.

    position = +1 (short spread) when z > entry, -1 (long spread) when z < -entry.
    """
    state = 0
    pos = []
    for val in z.fillna(0):
        if state == 0:
            if val > entry:
                state = +1
            elif val < -entry:
                state = -1
        elif abs(val) < exit:
            state = 0
        pos.append(state)
    return pd.DataFrame({"position": pos}, index=z.index)


def extract_trades(pos: pd.Series) -> pd.DataFrame:
    """Reconstruct entry/exit rows from a position series."""
    pos = pos.fillna(0).astype(int)
    entries = (pos.shift(1) == 0) & (pos != 0)
    exits = (pos.shift(1) != 0) & (pos == 0)
    df = pd.DataFrame({"pos": pos, "entry": entries, "exit": exits})
    open_side = None
    rows = []
    for dt, r in df.iterrows():
        if r["entry"]:
            open_side = (dt, r["pos"])
        elif r["exit"] and open_side is not None:
            rows.append({"entry_time": open_side[0], "side": int(open_side[1]), "exit_time": dt})
            open_side = None
    return pd.DataFrame(rows)
