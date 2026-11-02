"""End-to-end example on real market data (2007–2016 daily adjusted closes).

Data: MSFT, IBM, SBUX, AAPL (source: plotly/datasets, originally Yahoo Finance).
Pipeline: Engle–Granger screening across all pairs -> backtest the most
cointegrated pair -> performance metrics, Sharpe CI, and 2008-crisis stress window.

Run:  python examples/run_real_data.py
"""
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pairs_trading import (
    Config,
    engle_granger_adf,
    half_life,
    run_pairs_backtest_series,
    stationary_bootstrap_sharpe,
    stress_window_metrics,
)

# ---- Load real prices (log prices: standard for pairs spreads) ----
import numpy as np

raw = pd.read_csv(ROOT / "data" / "stockdata.csv", parse_dates=["Date"]).set_index("Date")
tickers = ["MSFT", "IBM", "SBUX", "AAPL"]
df = np.log(raw[tickers])

# ---- Engle–Granger screening across all pairs ----
print("Engle–Granger cointegration screening (ADF on OLS residual):")
results = []
for a, b in combinations(tickers, 2):
    stat, pval, beta = engle_granger_adf(df[a], df[b])
    results.append((a, b, stat, pval, beta))
    print(f"  {a:>4} / {b:<4}  ADF = {stat:6.2f}   p-value = {pval:.4f}   beta = {beta:.3f}")

best = min(results, key=lambda r: r[3])
y_sym, x_sym = best[0], best[1]
print(f"\nSelected pair: {y_sym}/{x_sym} (lowest p-value = {best[3]:.4f})")

# ---- Backtest the selected pair ----
cfg = Config(
    path_y="", path_x="",
    roll_window=60, entry_z=2.0, exit_z=0.5,
    cost_bps=5.0, slippage_bps=2.0,
    out_dir=str(ROOT / "out"),
)
res = run_pairs_backtest_series(df[y_sym], df[x_sym], cfg)

spread = df[y_sym] - best[4] * df[x_sym]
print(f"OU half-life of the spread: {half_life(spread):.1f} days")

print("\nPerformance metrics (after 7 bps costs+slippage per side):")
for k, v in res["metrics"].items():
    print(f"  {k:>25}: {v: .4f}")
print(f"  {'n_trades':>25}: {len(res['trades'])}")

lo, hi = stationary_bootstrap_sharpe(res["daily"]["daily_ret"])
print(f"  {'sharpe 95% CI':>25}: [{lo:.2f}, {hi:.2f}]")

# ---- Stress window: global financial crisis ----
gfc = stress_window_metrics(res["daily"]["daily_ret"], "2008-09-01", "2009-03-31")
print("\nStress window — Global Financial Crisis (Sep 2008 – Mar 2009):")
for k, v in gfc.items():
    print(f"  {k:>25}: {v: .4f}")

print(f"\nPlots saved to {cfg.out_dir}/")
