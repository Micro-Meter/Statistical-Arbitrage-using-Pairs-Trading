"""Dollar-neutral pairs backtest with transaction costs and slippage."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .data import align_prices, read_price_csv
from .cointegration import calc_spread, rolling_hedge_ratio
from .metrics import metrics
from .signals import extract_trades, generate_signals, zscore


@dataclass
class Config:
    """Backtest configuration."""
    path_y: str
    path_x: str
    date_col: str = "Date"
    price_col: str = "Close"
    start: Optional[str] = None
    end: Optional[str] = None
    roll_window: int = 60
    entry_z: float = 2.0
    exit_z: float = 0.5
    cost_bps: float = 5.0
    slippage_bps: float = 2.0
    notional: float = 1_000_000.0
    out_dir: str = "out"


def backtest(
    y: pd.Series,
    x: pd.Series,
    beta,
    sig: pd.DataFrame,
    cost_bps: float,
    slippage_bps: float,
    notional: float,
) -> Dict[str, pd.DataFrame]:
    """Run the daily backtest. Positions execute next day (no look-ahead)."""
    y_ret = y.pct_change().fillna(0.0)
    x_ret = x.pct_change().fillna(0.0)
    pos = sig["position"].fillna(0.0)
    pos_shift = pos.shift(1).fillna(0.0)  # executed next day

    if isinstance(beta, (float, int)):
        beta_series = pd.Series(float(beta), index=y.index)
    else:
        beta_series = pd.Series(beta, index=y.index).ffill()

    # Exposures: long y / short beta*x when pos = -1 (long spread); inverse for +1
    w_y = np.where(pos_shift.values == -1, +1.0, np.where(pos_shift.values == +1, -1.0, 0.0))
    beta_vals = np.nan_to_num(beta_series.values, nan=0.0)
    w_x = np.where(w_y == 0.0, 0.0, -beta_vals * w_y)

    # PnL before costs (dollar-neutral approximation)
    pnl = notional * (w_y * y_ret.values + w_x * x_ret.values)

    # Transaction costs + slippage charged on position changes
    turns = np.abs(np.diff(np.r_[0, pos.values]))
    cost_perc = (cost_bps + slippage_bps) * 1e-4
    costs = notional * cost_perc * turns
    pnl_after = pnl - costs

    equity = notional + np.cumsum(pnl_after)

    daily = pd.DataFrame(
        {
            "y_ret": y_ret,
            "x_ret": x_ret,
            "pos_exec": pos_shift,
            "beta": beta_series,
            "pnl": pnl_after,
            "daily_ret": pnl_after / notional,
            "equity": equity,
        },
        index=y.index,
    )

    trades = extract_trades(sig["position"])
    return {"daily": daily, "trades": trades}


def save_plots(out_dir: str, y, x, spread, z, daily) -> None:
    """Save diagnostic plots (prices, spread, z-score, equity) as PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    plt.figure(); y.plot(label="Y"); x.plot(label="X"); plt.title("Aligned Prices")
    plt.legend(); plt.tight_layout(); plt.savefig(out / "prices.png", dpi=180); plt.close()

    plt.figure(); spread.plot(); plt.title("Spread (y - beta*x)")
    plt.tight_layout(); plt.savefig(out / "spread.png", dpi=180); plt.close()

    plt.figure(); z.plot(); plt.title("Z-score (rolling)")
    plt.tight_layout(); plt.savefig(out / "zscore.png", dpi=180); plt.close()

    plt.figure(); daily["equity"].plot(); plt.title("Equity Curve (after costs)")
    plt.tight_layout(); plt.savefig(out / "equity.png", dpi=180); plt.close()


def run_pairs_backtest(cfg: Config) -> dict:
    """End-to-end pipeline: load, align, hedge, signal, backtest, report."""
    y = read_price_csv(cfg.path_y, cfg.date_col, cfg.price_col)
    x = read_price_csv(cfg.path_x, cfg.date_col, cfg.price_col)
    return run_pairs_backtest_series(y, x, cfg)


def run_pairs_backtest_series(y: pd.Series, x: pd.Series, cfg: Config) -> dict:
    """Same pipeline, starting from in-memory price series."""
    y, x = align_prices(y, x, cfg.start, cfg.end)
    beta_roll = rolling_hedge_ratio(y, x, cfg.roll_window)
    spread = calc_spread(y, x, beta_roll)
    z = zscore(spread, cfg.roll_window)
    sig = generate_signals(z, cfg.entry_z, cfg.exit_z)
    bt = backtest(y, x, beta_roll, sig, cfg.cost_bps, cfg.slippage_bps, cfg.notional)
    mets = metrics(bt["daily"])
    save_plots(cfg.out_dir, y, x, spread, z, bt["daily"])
    return {"metrics": mets, "trades": bt["trades"], "daily": bt["daily"], "config": cfg}
