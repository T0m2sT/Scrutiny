#!/usr/bin/env python3
"""
Portfolio Backtest & Forward Monte Carlo Simulator
====================================================
Two distinct analyses on the same holdings.csv (ticker, name, target_weight):

1. HISTORICAL BACKTEST: pulls actual historical prices for each ticker over a
   real window (default: as far back as data allows, up to 10 years) and
   computes what this exact weight allocation would have returned, including
   annualized return, volatility, Sharpe, max drawdown, and a rebalanced vs.
   buy-and-hold comparison.

2. FORWARD MONTE CARLO: uses the historical daily-return covariance matrix
   (correlations + volatilities) estimated from the same price history,
   combined with user-suppliable (or historically-estimated) expected annual
   returns per stock, to simulate N possible future paths over a chosen
   horizon (default 40 years). Reports the distribution of outcomes (median,
   10th/25th/75th/90th percentile) rather than a single point estimate.

Usage:
    python3 backtest_and_forecast.py holdings.csv [--years-back 10]
        [--sim-years 40] [--n-sims 5000] [--backtest-mode common|full]
"""
import sys
import csv
import argparse
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
REPORTS_DIR = HERE / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="Portfolio Backtest & Monte Carlo Forecaster")
    p.add_argument("holdings_file", help="Path to holdings CSV (ticker, name, target_weight)")
    p.add_argument("--years-back", type=int, default=10, help="Years of history to pull for backtest/covariance (default 10)")
    p.add_argument("--sim-years", type=int, default=40, help="Forward simulation horizon in years (default 40)")
    p.add_argument("--n-sims", type=int, default=5000, help="Number of Monte Carlo paths (default 5000)")
    p.add_argument("--backtest-mode", choices=["common", "full"], default="common",
                   help="'common' = trim backtest to the shortest available history among all tickers (fair comparison); "
                        "'full' = use each ticker's full available history, backtest window = longest common overlap still enforced for portfolio math, "
                        "but per-ticker individual stats shown for their full window")
    p.add_argument("--weekly-contribution", type=float, default=0.0,
                    help="Optional weekly DCA contribution amount (in base currency) to layer into the Monte Carlo forward simulation, e.g. 50")
    p.add_argument("--starting-value", type=float, default=None,
                    help="Actual current portfolio value in base currency (e.g. 1300). REQUIRED if --weekly-contribution > 0, "
                         "since mixing a normalized starting value with real contribution dollars produces nonsensical results. "
                         "If omitted and weekly-contribution is 0, defaults to 1.0 (a 'growth multiple' view).")
    p.add_argument("--rf-rate", type=float, default=0.045, help="Risk-free rate for Sharpe calculations (default 0.045)")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[*] Loaded args for {args.holdings_file} (years_back={args.years_back}, sim_years={args.sim_years})")
    print("[*] TODO: load holdings, fetch price history, backtest, run Monte Carlo, write report.")


if __name__ == "__main__":
    main()
