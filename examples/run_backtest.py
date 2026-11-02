"""Minimal end-to-end example on a synthetic cointegrated pair.

Run:  python examples/run_backtest.py
For real data, point Config at two CSVs with Date/Close columns.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pairs_trading import Config, make_synthetic_pair, run_pairs_backtest_series, stationary_bootstrap_sharpe

y, x = make_synthetic_pair()
cfg = Config(path_y="", path_x="", roll_window=60, entry_z=2.0, exit_z=0.5,
             cost_bps=5.0, slippage_bps=2.0, out_dir="out")
res = run_pairs_backtest_series(y, x, cfg)

print("Performance metrics:")
for k, v in res["metrics"].items():
    print(f"  {k:>25}: {v: .4f}")
print(f"  {'n_trades':>25}: {len(res['trades'])}")
lo, hi = stationary_bootstrap_sharpe(res["daily"]["daily_ret"])
print(f"  {'sharpe 95% CI':>25}: [{lo:.2f}, {hi:.2f}]")
print("Plots saved to ./out/")
