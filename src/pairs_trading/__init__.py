"""Pairs trading backtesting framework — CQF final project."""
from .backtest import Config, backtest, run_pairs_backtest, run_pairs_backtest_series
from .cointegration import (
    calc_spread,
    engle_granger_adf,
    half_life,
    hedge_ratio_ols,
    johansen_test,
    rolling_hedge_ratio,
)
from .data import align_prices, make_synthetic_pair, read_price_csv
from .metrics import metrics, sortino_ratio, stationary_bootstrap_sharpe
from .risk import stress_window_metrics, var_es, volatility_targeting
from .signals import extract_trades, generate_signals, zscore

__version__ = "1.0.0"
