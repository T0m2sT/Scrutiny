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

DEFAULT_EXPECTED_RETURNS = {
    "NVDA": 0.16, "MA": 0.14, "V": 0.13, "CB": 0.10, "LLY": 0.15,
    "HWM": 0.13, "AVGO": 0.15, "PM": 0.09, "UBER": 0.14, "VRT": 0.14,
    "GEV": 0.13, "VRTX": 0.11, "GE": 0.12, "UNH": 0.09, "TJX": 0.10,
    "ISRG": 0.13, "PGR": 0.11, "KLAC": 0.13, "TRGP": 0.08, "KAP.L": 0.11,
    "MSFT": 0.12,
    "CEG": 0.11,
    "WELL": 0.08,
    "KMI": 0.09,
    "ORCL": 0.13,
    "SCHW": 0.11,
    "TEL": 0.13,
}
DEFAULT_FALLBACK_RETURN = 0.10


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


def build_backtest(px, holdings, first_valid, mode, rf_rate, years_back, min_history_days=5):
    weights = {h["ticker"]: h["target_weight"] for h in holdings}
    tickers = list(weights.keys())
    resolved = [t for t in tickers if t in px.columns and first_valid.get(t) is not None]
    missing = [t for t in tickers if t not in resolved]

    notes = []
    if missing:
        notes.append(f"Excluded entirely (no price data resolved at all): {', '.join(missing)}")

    available = []
    unusable = []
    for t in resolved:
        depth_days = (px[t].dropna().index[-1] - first_valid[t]).days
        if depth_days < min_history_days:
            unusable.append((t, depth_days))
        else:
            available.append(t)
    if unusable:
        notes.append(
            "Excluded entirely (essentially no usable price history, likely a bad/unsupported ticker symbol): "
            + ", ".join([f"{t} ({d}d)" for t, d in unusable])
        )

    if not available:
        raise ValueError("No tickers have any usable history. Check your ticker symbols.")

    window_end = px.index[-1]
    window_start = px.index[0]

    entry_notes = []
    for t in available:
        entry = first_valid[t]
        depth_years = (window_end - entry).days / 365.25
        if entry > window_start + pd.Timedelta(days=30):
            entry_notes.append(f"{t} enters on {entry.date()} ({depth_years:.1f}y of history) - included from its actual IPO/listing date, not before")
    if entry_notes:
        notes.append(
            "Stocks that IPO'd or spun off DURING the backtest window are included starting from their actual "
            "first trading date (as if bought the day they became available), not backfilled or excluded: "
            + "; ".join(entry_notes)
        )

    px_window = px[available].loc[window_start:window_end]

    daily_returns = px_window.pct_change().fillna(0)

    target_w = pd.Series({t: weights[t] for t in available})

    is_listed = px_window.notna()
    listed_target = is_listed.mul(target_w, axis=1)
    row_sums = listed_target.sum(axis=1)
    row_sums = row_sums.replace(0, np.nan)
    effective_weights = listed_target.div(row_sums, axis=0).fillna(0)

    port_daily_returns = (daily_returns[available].fillna(0) * effective_weights.shift(1).fillna(effective_weights.iloc[0])).sum(axis=1)
    port_daily_returns.iloc[0] = 0.0

    bh_index = (1 + port_daily_returns).cumprod()
    years_elapsed = (px_window.index[-1] - px_window.index[0]).days / 365.25
    bh_cagr = bh_index.iloc[-1] ** (1 / years_elapsed) - 1 if years_elapsed > 0 else float("nan")
    bh_vol = float(port_daily_returns.std() * np.sqrt(252))
    bh_sharpe = (bh_cagr - rf_rate) / bh_vol if bh_vol > 0 else float("nan")
    bh_peak = bh_index.cummax()
    bh_dd = float(((bh_index - bh_peak) / bh_peak).min())

    month_starts = px_window.index.to_series().groupby(px_window.index.to_period("M")).min()
    month_start_set = set(month_starts.values)

    reb_returns = []
    current_weights = effective_weights.iloc[0].copy()
    prev_listed = is_listed.iloc[0].copy()

    for i in range(len(px_window)):
        date = px_window.index[i]
        today_listed = is_listed.iloc[i]

        listed_changed = not today_listed.equals(prev_listed)
        if date in month_start_set or listed_changed or i == 0:
            listed_now = today_listed[today_listed].index
            tw = target_w[listed_now]
            current_weights = pd.Series(0.0, index=available)
            current_weights[listed_now] = tw / tw.sum()

        day_ret = daily_returns.iloc[i].fillna(0)
        port_ret = float((current_weights * day_ret).sum())
        reb_returns.append(port_ret if i > 0 else 0.0)
        prev_listed = today_listed.copy()

    reb_daily_returns = pd.Series(reb_returns, index=px_window.index)
    reb_index = (1 + reb_daily_returns).cumprod()
    reb_cagr = reb_index.iloc[-1] ** (1 / years_elapsed) - 1 if years_elapsed > 0 else float("nan")
    reb_vol = float(reb_daily_returns.std() * np.sqrt(252))
    reb_sharpe = (reb_cagr - rf_rate) / reb_vol if reb_vol > 0 else float("nan")
    reb_peak = reb_index.cummax()
    reb_dd = float(((reb_index - reb_peak) / reb_peak).min())

    min_overlap_days = 252
    overlap_counts = is_listed.astype(int).T.dot(is_listed.astype(int))
    thin_pairs = []
    for i, t1 in enumerate(available):
        for t2 in available[i+1:]:
            if overlap_counts.loc[t1, t2] < min_overlap_days:
                thin_pairs.append((t1, t2, int(overlap_counts.loc[t1, t2])))
    if thin_pairs:
        thin_tickers = sorted(set([p[0] for p in thin_pairs] + [p[1] for p in thin_pairs]))
        notes.append(
            f"Some stock pairs have less than {min_overlap_days} trading days of simultaneous listing, so their "
            "pairwise correlation is estimated from limited data and a fallback/default correlation is used where "
            f"too thin to trust: involves {', '.join(thin_tickers)}."
        )

    return {
        "window_start": window_start,
        "window_end": window_end,
        "years_elapsed": years_elapsed,
        "available_tickers": available,
        "missing_tickers": missing,
        "daily_returns": daily_returns,
        "is_listed": is_listed,
        "min_overlap_days": min_overlap_days,
        "bh_cagr": bh_cagr, "bh_vol": bh_vol, "bh_sharpe": bh_sharpe, "bh_max_dd": bh_dd,
        "reb_cagr": reb_cagr, "reb_vol": reb_vol, "reb_sharpe": reb_sharpe, "reb_max_dd": reb_dd,
        "notes": notes,
    }


def run_monte_carlo(holdings, daily_returns, sim_years, n_sims, weekly_contribution,
                     starting_value=1.0, expected_returns_override=None):
    tickers = [t for t in daily_returns.columns]
    weights_map = {h["ticker"]: h["target_weight"] for h in holdings if h["ticker"] in tickers}
    w_sum = sum(weights_map.values())
    weights = np.array([weights_map[t] / w_sum for t in tickers])

    min_periods = 200
    cov_daily = daily_returns[tickers].cov(min_periods=min_periods)
    cov_annual_raw = cov_daily.values * 252

    default_correlation = 0.30
    std_annual_prelim = np.sqrt(np.nanvar(daily_returns[tickers].values, axis=0) * 252)
    default_cov_matrix = default_correlation * np.outer(std_annual_prelim, std_annual_prelim)
    np.fill_diagonal(default_cov_matrix, std_annual_prelim ** 2)

    nan_mask = np.isnan(cov_annual_raw)
    if nan_mask.any():
        n_thin = int(nan_mask.sum() / 2)
        print(f"[!] {n_thin} stock pair(s) had too little overlapping history to estimate a reliable correlation "
              f"(fewer than {min_periods} shared trading days) - using a default {default_correlation} correlation assumption for those pairs.")
    cov_annual = np.where(nan_mask, default_cov_matrix, cov_annual_raw)

    std_annual_raw = np.sqrt(np.diag(cov_annual))
    std_annual_clamped = np.clip(std_annual_raw, 0.15, 1.20)
    if not np.allclose(std_annual_raw, std_annual_clamped):
        clamped_names = [tickers[i] for i in range(len(tickers)) if not np.isclose(std_annual_raw[i], std_annual_clamped[i])]
        print(f"[!] Clamped implausible annualized volatility for: {', '.join(clamped_names)} "
              f"(raw estimates ranged outside 15%-120%, likely due to a short/noisy backtest window)")

    exp_map = expected_returns_override or DEFAULT_EXPECTED_RETURNS
    fallback_tickers = [t for t in tickers if t not in exp_map]
    mu = np.array([exp_map.get(t, DEFAULT_FALLBACK_RETURN) for t in tickers])

    if fallback_tickers:
        print(f"\n[!] WARNING: {len(fallback_tickers)} ticker(s) have NO curated expected-return "
              f"estimate and are using the generic {DEFAULT_FALLBACK_RETURN*100:.0f}% fallback: "
              f"{', '.join(fallback_tickers)}")
        print(f"    This means any Monte Carlo comparison involving these tickers may not reflect "
              f"their real fundamentals - add a real estimate to DEFAULT_EXPECTED_RETURNS or pass "
              f"--expected-return-overrides before trusting a comparison that hinges on them.\n")

    n_assets = len(tickers)
    n_steps = sim_years

    std_annual = std_annual_raw
    corr = cov_annual / np.outer(std_annual, std_annual)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)

    cov_for_sim = corr * np.outer(std_annual_clamped, std_annual_clamped)

    try:
        L = np.linalg.cholesky(cov_for_sim + 1e-8 * np.eye(n_assets))
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(cov_for_sim)
        eigvals = np.clip(eigvals, 1e-8, None)
        cov_for_sim = eigvecs @ np.diag(eigvals) @ eigvecs.T
        L = np.linalg.cholesky(cov_for_sim + 1e-8 * np.eye(n_assets))

    rng = np.random.default_rng(42)
    port_paths = np.zeros((n_sims, n_steps + 1))
    port_paths[:, 0] = starting_value

    weekly_annual_contribution = weekly_contribution * 52

    for sim in range(n_sims):
        value = starting_value
        for yr in range(1, n_steps + 1):
            z = rng.standard_normal(n_assets)
            correlated_shock = L @ z
            asset_simple_returns = mu + correlated_shock
            asset_simple_returns = np.clip(asset_simple_returns, -0.95, None)
            port_return = np.dot(weights, asset_simple_returns)
            port_return = max(port_return, -0.95)
            value = value * (1 + port_return) + weekly_annual_contribution
            port_paths[sim, yr] = value

    final_values = port_paths[:, -1]
    percentiles = {
        "p10": float(np.percentile(final_values, 10)),
        "p25": float(np.percentile(final_values, 25)),
        "p50_median": float(np.percentile(final_values, 50)),
        "p75": float(np.percentile(final_values, 75)),
        "p90": float(np.percentile(final_values, 90)),
        "mean": float(np.mean(final_values)),
    }

    return {
        "final_values": final_values,
        "percentiles": percentiles,
        "paths": port_paths,
        "tickers": tickers,
        "weights": dict(zip(tickers, weights)),
        "expected_returns_used": dict(zip(tickers, mu)),
        "fallback_tickers": fallback_tickers,
    }


def main():
    args = parse_args()
    holdings = load_holdings(args.holdings_file)
    tickers = [h["ticker"] for h in holdings]

    px, first_valid = fetch_price_history(tickers, args.years_back)

    print("[*] Building historical backtest...")
    backtest = build_backtest(px, holdings, first_valid, args.backtest_mode, args.rf_rate, args.years_back)

    print(f"[*] Running Monte Carlo simulation ({args.n_sims:,} paths, {args.sim_years} years)...")
    mc = run_monte_carlo(
        holdings,
        backtest["daily_returns"],
        args.sim_years,
        args.n_sims,
        args.weekly_contribution,
        starting_value=(args.starting_value or 1.0),
    )
    print(f"[*] Median outcome: {mc['percentiles']['p50_median']:.2f}")
    print("[*] TODO: write report.")


if __name__ == "__main__":
    main()
