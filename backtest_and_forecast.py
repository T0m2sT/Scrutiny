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


def load_holdings(path):
    """
    Reads a holdings CSV (ticker, name, target_weight). Accepts weights either
    as whole-number percentages (e.g. "8" or "8%" meaning 8%) or as decimal
    fractions (e.g. "0.08" meaning 8%) - but detects the format ONCE for the
    WHOLE FILE, not per-row. A prior per-row heuristic (treat any single value
    over 1.5 as "must be a percentage") broke silently on files that mix a
    normal percentage-style value (e.g. "1" meaning 1%) with the rest of the
    column, since "1" is indistinguishable from "already a decimal fraction of
    1.0" in isolation - this produced a badly wrong total (e.g. 199%) with no
    error, because the too-small value was left unconverted.
    """
    raw_rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            ticker = r["ticker"].strip().upper()
            name = r["name"].strip()
            raw_weight = float(r["target_weight"].strip().rstrip("%"))
            raw_rows.append({"ticker": ticker, "name": name, "raw_weight": raw_weight})

    if not raw_rows:
        raise ValueError(f"No rows found in {path}")

    raw_sum = sum(r["raw_weight"] for r in raw_rows)

    # Decide the format ONCE, for the whole column, based on which
    # interpretation lands closer to a sane total (100 for percentages, 1.0
    # for fractions) - far more robust than checking any single row in
    # isolation, since one small-but-legitimate row (e.g. "1" meaning 1%)
    # can't be told apart from a decimal fraction on its own.
    dist_as_percent = abs(raw_sum - 100.0)
    dist_as_fraction = abs(raw_sum - 1.0)

    if dist_as_percent <= dist_as_fraction:
        divisor = 100.0
        detected_format = "whole-number percentages (e.g. '8' = 8%)"
    else:
        divisor = 1.0
        detected_format = "decimal fractions (e.g. '0.08' = 8%)"

    rows = []
    for r in raw_rows:
        rows.append({
            "ticker": r["ticker"],
            "name": r["name"],
            "target_weight": r["raw_weight"] / divisor,
        })

    total = sum(r["target_weight"] for r in rows)
    print(f"[*] Detected weight format: {detected_format} (raw column sum was {raw_sum:.2f})")

    # Duplicate ticker check - a common source of a bad total that the format
    # detection above wouldn't catch on its own.
    seen = {}
    for r in rows:
        seen[r["ticker"]] = seen.get(r["ticker"], 0) + 1
    dupes = [t for t, c in seen.items() if c > 1]
    if dupes:
        print(f"[!] WARNING: duplicate ticker(s) found in {path}: {', '.join(dupes)} - "
              "each occurrence is being kept and will be summed together. If this isn't intended, fix the CSV.")

    if not (0.99 <= total <= 1.01):
        print(f"[!] Weights sum to {total*100:.2f}% after format detection, normalizing to 100%. "
              "Double-check your CSV if this wasn't expected - it usually means a genuine typo or mixed format, not just rounding.")
        for r in rows:
            r["target_weight"] /= total
    return rows


def fetch_price_history(tickers, years_back):
    """Downloads daily close prices for all tickers, returns a DataFrame and
    a dict of {ticker: first_valid_date} showing actual history depth."""
    print(f"[*] Downloading up to {years_back} years of price history for {len(tickers)} tickers...")
    period_str = f"{years_back}y"
    raw = yf.download(tickers, period=period_str, interval="1d", progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        px = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw.iloc[:, 0]
    else:
        px = raw[["Close"]] if "Close" in raw.columns else raw
        if isinstance(px, pd.Series):
            px = px.to_frame(name=tickers[0])

    first_valid = {}
    for t in px.columns:
        s = px[t].dropna()
        first_valid[t] = s.index[0] if len(s) > 0 else None

    return px, first_valid


def main():
    args = parse_args()
    holdings = load_holdings(args.holdings_file)
    tickers = [h["ticker"] for h in holdings]

    px, first_valid = fetch_price_history(tickers, args.years_back)
    print(f"[*] Downloaded price history for {len(tickers)} tickers, {len(px)} rows.")
    print("[*] TODO: build backtest, run Monte Carlo, write report.")


if __name__ == "__main__":
    main()
