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

IMPORTANT HONEST LIMITATIONS (also printed in the report):
- Several holdings have limited price history (e.g. recent spinoffs/IPOs).
  The backtest window is automatically capped to the shortest available
  history among included tickers, OR those tickers are excluded from the
  backtest with a clear note (user's choice via --backtest-mode).
- A historical backtest measures what the STOCKS did, not necessarily what
  a company with today's business mix and thesis would have done - this is
  flagged explicitly in the output.
- The Monte Carlo simulation is a statistical projection based on historical
  volatility/correlation, not a guarantee of future results. Expected returns
  are a required, debatable input - the script uses conservative, sourced
  estimates but the user should treat the whole distribution, not just the
  median, as the real answer.

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

DEFAULT_EXPECTED_RETURNS_FILE = HERE / "expected_returns.csv"
DEFAULT_FALLBACK_RETURN = 0.10  # for any ticker with no estimate in the expected-returns file


def load_expected_returns(path=DEFAULT_EXPECTED_RETURNS_FILE):
    """
    Loads conservative, sourced long-run expected annual return estimates per
    ticker from a CSV (ticker,expected_return). These are DEBATABLE INPUTS,
    not facts - shown transparently in the report so the user can override
    any of them. Returns {} if the file doesn't exist (every ticker then
    falls back to DEFAULT_FALLBACK_RETURN, loudly flagged in the report).
    """
    if not Path(path).exists():
        return {}
    result = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            ticker = r["ticker"].strip().upper()
            result[ticker] = float(r["expected_return"])
    return result


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


def build_backtest(px, holdings, first_valid, mode, rf_rate, years_back, min_history_days=5):
    """
    "As-you-go" backtest: rather than trimming the whole portfolio down to the
    single latest IPO date among all holdings, each stock enters the backtest
    on its OWN actual first-trading-date (as if you'd started buying it the
    day it became available), and the full target weight is only reached once
    every stock has IPO'd. Before that, capital is allocated pro-rata across
    whichever target stocks are ALREADY public, exactly like a real investor
    building this exact portfolio over time would have experienced it.

    The overall backtest window is fixed at `years_back` (e.g. 10 years) - it
    does NOT get silently collapsed by one very-recently-listed ticker. Only
    stocks that quite literally have no resolvable data at all (min_history_days)
    are excluded outright.
    """
    weights = {h["ticker"]: h["target_weight"] for h in holdings}
    tickers = list(weights.keys())
    resolved = [t for t in tickers if t in px.columns and first_valid.get(t) is not None]
    missing = [t for t in tickers if t not in resolved]

    notes = []
    if missing:
        notes.append(f"Excluded entirely (no price data resolved at all): {', '.join(missing)}")

    # Drop only truly-unusable tickers (essentially zero real data)
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

    # The full backtest window: from years_back ago (or the earliest available
    # data point, if less history exists in the DOWNLOAD, though the intent
    # is that the window itself is fixed at years_back regardless of any
    # single stock's IPO date) through to today.
    window_end = px.index[-1]
    window_start = px.index[0]  # yfinance was asked for years_back, so this reflects that already

    # Report each stock's actual entry point relative to the fixed window,
    # purely informational - this is expected/normal, not a data problem.
    entry_notes = []
    for t in available:
        entry = first_valid[t]
        depth_years = (window_end - entry).days / 365.25
        if entry > window_start + pd.Timedelta(days=30):  # meaningfully later than window start
            entry_notes.append(f"{t} enters on {entry.date()} ({depth_years:.1f}y of history) - included from its actual IPO/listing date, not before")
    if entry_notes:
        notes.append(
            "Stocks that IPO'd or spun off DURING the backtest window are included starting from their actual "
            "first trading date (as if bought the day they became available), not backfilled or excluded: "
            + "; ".join(entry_notes)
        )

    px_window = px[available].loc[window_start:window_end]

    daily_returns = px_window.pct_change().fillna(0)

    # --- "As-you-go" weighted index: at each day, only currently-listed
    # stocks share the target weight, renormalized pro-rata among themselves.
    # As each new stock IPOs, it joins at its target weight and existing
    # holdings' effective weights shrink back toward their true targets.
    target_w = pd.Series({t: weights[t] for t in available})

    is_listed = px_window.notna()
    # Effective weight each day = target weight if listed, else 0, renormalized so weights sum to 1 among listed names
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

    # --- Monthly-rebalanced version: same as-you-go entry logic, but weights
    # among currently-listed stocks are reset to their pro-rata target at each
    # month start rather than left to drift daily between rebalances.
    month_starts = px_window.index.to_series().groupby(px_window.index.to_period("M")).min()
    month_start_set = set(month_starts.values)

    reb_returns = []
    current_weights = effective_weights.iloc[0].copy()
    prev_listed = is_listed.iloc[0].copy()

    for i in range(len(px_window)):
        date = px_window.index[i]
        today_listed = is_listed.iloc[i]

        # Reset to pro-rata target weights at each month start OR whenever the
        # set of listed stocks changes (a new stock just IPO'd) - both are
        # natural "re-check allocation" moments for a real investor.
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

    # For the Monte Carlo covariance estimate downstream: pass the FULL daily
    # returns (including NaNs for periods before a stock existed) rather than
    # truncating to a single "all stocks overlap" window. Requiring every
    # stock to overlap is too strict when one stock (e.g. a very recent
    # IPO/spinoff) has barely any history - it would force the correlation
    # estimate down to that one stock's tiny window, corrupting everything
    # else. Instead, run_monte_carlo() computes each pairwise correlation over
    # whichever period THOSE TWO stocks actually overlap (pandas .cov()'s
    # native pairwise-complete-observation behavior), and falls back to a
    # sane default correlation/volatility for any stock with too little
    # overlap with the rest to estimate reliably.
    min_overlap_days = 252  # require at least ~1 trading year of overlap for a pairwise correlation to be trusted
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
    """
    starting_value: the actual starting portfolio value in your base currency
    (e.g. 1300 for a EUR1,300 portfolio). Defaults to 1.0 for a "growth
    multiple" view when no contributions are modeled. IMPORTANT: if
    weekly_contribution > 0, starting_value should be set to a real currency
    amount, not 1.0 - mixing a normalized starting value (1.0) with a real
    currency contribution amount was a previous bug that produced absurd
    (millions-x) results, since the contribution dollars swamped the
    normalized scale within a few simulated years.
    """
    tickers = [t for t in daily_returns.columns]
    weights_map = {h["ticker"]: h["target_weight"] for h in holdings if h["ticker"] in tickers}
    w_sum = sum(weights_map.values())
    weights = np.array([weights_map[t] / w_sum for t in tickers])

    # Annualized covariance matrix from historical daily returns.
    # min_periods enforces that any pairwise correlation estimated from fewer
    # than ~1 trading year of simultaneous overlap comes back as NaN (handled
    # below) rather than a wild, unstable estimate from too few data points -
    # e.g. two stocks that only overlap for 27 days could show a spurious
    # +/-0.95 correlation purely by chance, which would badly distort the
    # simulated co-movement between them for the other ~39 years being modeled.
    min_periods = 200
    cov_daily = daily_returns[tickers].cov(min_periods=min_periods)
    cov_annual_raw = cov_daily.values * 252

    # Any NaN entries (pairs with too little overlap to trust) get replaced
    # with a moderate default correlation (0.3, a reasonable "these are both
    # equities, expect some co-movement but not perfect" assumption) rather
    # than left as NaN (which would break the simulation) or 0 (which would
    # understate real equity market co-movement).
    default_correlation = 0.30
    std_annual_prelim = np.sqrt(np.nanvar(daily_returns[tickers].values, axis=0) * 252)
    default_cov_matrix = default_correlation * np.outer(std_annual_prelim, std_annual_prelim)
    np.fill_diagonal(default_cov_matrix, std_annual_prelim ** 2)

    nan_mask = np.isnan(cov_annual_raw)
    if nan_mask.any():
        n_thin = int(nan_mask.sum() / 2)  # symmetric matrix, count unique pairs
        print(f"[!] {n_thin} stock pair(s) had too little overlapping history to estimate a reliable correlation "
              f"(fewer than {min_periods} shared trading days) - using a default {default_correlation} correlation assumption for those pairs.")
    cov_annual = np.where(nan_mask, default_cov_matrix, cov_annual_raw)

    # Defensive sanity clamp: if the backtest window was short (even if above
    # the hard exclusion threshold), the annualized volatility estimate can be
    # noisy and implausibly extreme. Cap individual-stock annualized volatility
    # to a generous but sane 15%-120% band before it gets compounded over many
    # simulated years - this prevents a single noisy estimate from producing
    # absurd multi-decade outcomes (e.g. millions-x) that reflect an estimation
    # artifact rather than real risk.
    std_annual_raw = np.sqrt(np.diag(cov_annual))
    std_annual_clamped = np.clip(std_annual_raw, 0.15, 1.20)
    if not np.allclose(std_annual_raw, std_annual_clamped):
        clamped_names = [tickers[i] for i in range(len(tickers)) if not np.isclose(std_annual_raw[i], std_annual_clamped[i])]
        print(f"[!] Clamped implausible annualized volatility for: {', '.join(clamped_names)} "
              f"(raw estimates ranged outside 15%-120%, likely due to a short/noisy backtest window)")

    # Expected returns: user override > default map > fallback.
    # CRITICAL: any ticker not explicitly in the curated map falls back to a
    # generic DEFAULT_FALLBACK_RETURN. This previously happened SILENTLY -
    # several real portfolio comparisons in practice were unknowingly comparing
    # tickers that all fell back to the same generic number, making any
    # Monte Carlo difference between them look like a real fundamentals-driven
    # result when it was actually just noise from volatility/correlation
    # structure. This is now loud: every fallback is printed AND written into
    # the report itself, tagged with an asterisk, so it can never be missed again.
    exp_map = expected_returns_override or load_expected_returns()
    fallback_tickers = [t for t in tickers if t not in exp_map]
    mu = np.array([exp_map.get(t, DEFAULT_FALLBACK_RETURN) for t in tickers])

    if fallback_tickers:
        print(f"\n[!] WARNING: {len(fallback_tickers)} ticker(s) have NO curated expected-return "
              f"estimate and are using the generic {DEFAULT_FALLBACK_RETURN*100:.0f}% fallback: "
              f"{', '.join(fallback_tickers)}")
        print(f"    This means any Monte Carlo comparison involving these tickers may not reflect "
              f"their real fundamentals - add a real estimate to {DEFAULT_EXPECTED_RETURNS_FILE.name} or pass "
              f"--expected-return-overrides before trusting a comparison that hinges on them.\n")

    n_assets = len(tickers)
    n_steps = sim_years  # simulate year-by-year for tractability over 40 years

    # Convert annual covariance to correlation, then reconstruct a covariance
    # matrix using the CLAMPED annualized volatility (to avoid noisy/short-
    # window estimates blowing up the simulation) combined with the historical
    # correlation structure (which is scale-invariant and doesn't need clamping).
    std_annual = std_annual_raw  # used only to derive correlation, which is scale-invariant
    corr = cov_annual / np.outer(std_annual, std_annual)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)

    cov_for_sim = corr * np.outer(std_annual_clamped, std_annual_clamped)

    # Cholesky for correlated normal draws (yearly steps, log-return space)
    try:
        L = np.linalg.cholesky(cov_for_sim + 1e-8 * np.eye(n_assets))
    except np.linalg.LinAlgError:
        # fallback: nearest positive-semidefinite adjustment
        eigvals, eigvecs = np.linalg.eigh(cov_for_sim)
        eigvals = np.clip(eigvals, 1e-8, None)
        cov_for_sim = eigvecs @ np.diag(eigvals) @ eigvecs.T
        L = np.linalg.cholesky(cov_for_sim + 1e-8 * np.eye(n_assets))

    # Cholesky for correlated normal draws (yearly steps)
    try:
        L = np.linalg.cholesky(cov_for_sim + 1e-8 * np.eye(n_assets))
    except np.linalg.LinAlgError:
        # fallback: nearest positive-semidefinite adjustment
        eigvals, eigvecs = np.linalg.eigh(cov_for_sim)
        eigvals = np.clip(eigvals, 1e-8, None)
        cov_for_sim = eigvecs @ np.diag(eigvals) @ eigvecs.T
        L = np.linalg.cholesky(cov_for_sim + 1e-8 * np.eye(n_assets))

    # Simulate SIMPLE (arithmetic) annual returns directly: mu is already the
    # expected simple annual return per stock, and cov_for_sim is the
    # covariance of simple annual returns. Do NOT convert to log-return space
    # here - mixing log-drift with simple-return-space covariance was the
    # source of a serious bug that produced absurd (millions-x) 40-year
    # outcomes. Each simulated asset return is clipped at -95% (a stock can't
    # lose more than 100%, and -95% is already an extreme single-year outcome)
    # to prevent compounding artifacts in low-probability tail draws.
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
            port_return = max(port_return, -0.95)  # floor the portfolio-level return too
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

    # Implied CAGR for median/mean paths (on the growth-of-1 portion only, excluding contributions, for intuition)
    return {
        "final_values": final_values,
        "percentiles": percentiles,
        "paths": port_paths,
        "tickers": tickers,
        "weights": dict(zip(tickers, weights)),
        "expected_returns_used": dict(zip(tickers, mu)),
        "fallback_tickers": fallback_tickers,
    }


def find_previous_report(current_out_path):
    """Finds the most recently modified report in REPORTS_DIR other than the
    one currently being written, so we can diff assumed-return assumptions
    across runs and flag whether a comparison is apples-to-apples."""
    candidates = [p for p in REPORTS_DIR.glob("backtest_forecast_*.md") if p != current_out_path]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_assumed_returns_from_report(path):
    """Extracts the 'Ticker | Assumed Annual Return' table from a previously
    written report, so a new run can be diffed against it. Returns a dict of
    {ticker: return_as_fraction}, stripping any '*' fallback marker."""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None
    result = {}
    in_table = False
    for line in text.splitlines():
        if line.strip().startswith("### Expected annual return assumptions"):
            in_table = True
            continue
        if in_table:
            if line.strip().startswith("| Ticker"):
                continue
            if line.strip().startswith("|:---"):
                continue
            if line.strip().startswith("|") and "%" in line:
                parts = [c.strip() for c in line.strip().strip("|").split("|")]
                if len(parts) == 2:
                    ticker = parts[0].rstrip("*").strip()
                    try:
                        pct = float(parts[1].rstrip("%").strip())
                        result[ticker] = pct / 100.0
                    except ValueError:
                        pass
            elif line.strip() == "" and result:
                break  # end of table
    return result if result else None


def build_assumption_diff_section(mc, current_out_path):
    """Compares this run's assumed-return table against the most recent prior
    report, if one exists, and reports whether the comparison is apples-to-
    apples (identical assumptions - any MC difference is pure structure/
    volatility/correlation) or assumptions-changed (a real, deliberate
    difference exists, listed explicitly)."""
    prev_path = find_previous_report(current_out_path)
    if prev_path is None:
        return ["## Assumption Diff vs. Previous Run\n",
                "No previous report found in `reports/` to compare against - this is treated as the first baseline run.\n"]

    prev_returns = parse_assumed_returns_from_report(prev_path)
    if prev_returns is None:
        return ["## Assumption Diff vs. Previous Run\n",
                f"Could not parse assumed-return table from the most recent previous report (`{prev_path.name}`) - skipping diff.\n"]

    current_returns = {t: r for t, r in mc["expected_returns_used"].items()}
    common = set(prev_returns) & set(current_returns)
    only_prev = set(prev_returns) - set(current_returns)
    only_current = set(current_returns) - set(prev_returns)
    changed = {t: (prev_returns[t], current_returns[t]) for t in common if abs(prev_returns[t] - current_returns[t]) > 1e-9}

    lines = ["## Assumption Diff vs. Previous Run\n",
             f"Compared against: `{prev_path.name}`\n"]

    if not changed and not only_prev and not only_current:
        lines.append("**IDENTICAL assumed-return assumptions on every shared ticker.** "
                      "Any difference between this run's Monte Carlo results and the previous run's is "
                      "purely from historical volatility/correlation structure and/or random-seed noise - "
                      "NOT from a fundamentals-driven change. Do not interpret such a difference as one "
                      "portfolio being fundamentally better.\n")
    else:
        lines.append("**Assumptions DIFFER from the previous run** - a Monte Carlo difference here may "
                      "reflect a real, deliberate change, not just noise. Details:\n")
        if changed:
            lines.append("Changed assumed returns for shared tickers:")
            for t, (old, new) in sorted(changed.items()):
                lines.append(f"  - {t}: {old*100:.1f}% → {new*100:.1f}%")
        if only_current:
            lines.append(f"New tickers in this run (not in previous): {', '.join(sorted(only_current))}")
        if only_prev:
            lines.append(f"Tickers dropped since previous run: {', '.join(sorted(only_prev))}")
        lines.append("")

    return lines


def build_report(holdings, backtest, mc, sim_years, n_sims, weekly_contribution, years_back, starting_value):
    lines = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines.append(f"# Portfolio Backtest & Forward Simulation Report — {ts}\n")

    lines.append("## Part 1 — Historical Backtest\n")
    lines.append(f"**Backtest window:** {backtest['window_start'].date()} to {backtest['window_end'].date()} (~{backtest['years_elapsed']:.1f} years, fixed at the requested --years-back)\n")
    if backtest["notes"]:
        lines.append("> **Data limitation notes:**")
        for n in backtest["notes"]:
            lines.append(f"> - {n}")
        lines.append("")

    lines.append("| Strategy | CAGR | Annualized Vol | Sharpe (Rf=4.5%) | Max Drawdown |")
    lines.append("|:---|---:|---:|---:|---:|")
    lines.append(f"| Buy & Hold (no rebalancing) | {backtest['bh_cagr']*100:.2f}% | {backtest['bh_vol']*100:.2f}% | {backtest['bh_sharpe']:.2f} | {backtest['bh_max_dd']*100:.2f}% |")
    lines.append(f"| Monthly Rebalanced to Target | {backtest['reb_cagr']*100:.2f}% | {backtest['reb_vol']*100:.2f}% | {backtest['reb_sharpe']:.2f} | {backtest['reb_max_dd']*100:.2f}% |")
    lines.append("")
    lines.append("**Included tickers:** " + ", ".join(backtest["available_tickers"]))
    if backtest["missing_tickers"]:
        lines.append(f"\n**Excluded (no data):** {', '.join(backtest['missing_tickers'])}")
    lines.append("\n> [!WARNING]")
    lines.append("> A historical backtest shows what these STOCKS actually did over this window - it does not mean today's business, thesis, or valuation will repeat those results. Several holdings (e.g. recent spinoffs) have short trading histories, which forced this backtest into a shorter common window than the full portfolio's 40-year intended horizon. Treat this as a sanity check on volatility and drawdown behavior, not a return forecast.\n")

    lines.append("## Part 2 — Forward Monte Carlo Simulation\n")
    contribution_label = f"€{weekly_contribution:,.0f}/week (€{weekly_contribution*52:,.0f}/year)" if weekly_contribution else "None (lump sum only)"
    lines.append(f"**Horizon:** {sim_years} years | **Simulations:** {n_sims:,} | **Starting value:** €{starting_value:,.0f} | **Weekly contribution modeled:** {contribution_label}\n")

    p = mc["percentiles"]
    unit_label = "Ending Portfolio Value (€)" if starting_value != 1.0 else "Ending Portfolio Multiple"
    fmt = (lambda v: f"€{v:,.0f}") if starting_value != 1.0 else (lambda v: f"{v:.1f}x")
    lines.append(f"| Outcome | {unit_label} | Notes |")
    lines.append("|:---|---:|:---|")
    lines.append(f"| 10th percentile (bad luck case) | {fmt(p['p10'])} | 1-in-10 chance of doing *worse* than this |")
    lines.append(f"| 25th percentile | {fmt(p['p25'])} | |")
    lines.append(f"| **Median (50th percentile)** | **{fmt(p['p50_median'])}** | Half of simulations landed above, half below |")
    lines.append(f"| 75th percentile | {fmt(p['p75'])} | |")
    lines.append(f"| 90th percentile (good luck case) | {fmt(p['p90'])} | 1-in-10 chance of doing *better* than this |")
    lines.append(f"| Mean (average) | {fmt(p['mean'])} | Skewed higher than median by compounding tail |")
    lines.append("")

    fallback_set = set(mc.get("fallback_tickers", []))
    lines.append("### Expected annual return assumptions used per stock\n")
    lines.append("These are debatable, sourced estimates, not facts - override any of them if you disagree.\n")
    if fallback_set:
        lines.append(f"> [!CAUTION]")
        lines.append(f"> **{len(fallback_set)} ticker(s) below (marked `*`) have NO curated estimate and are using the generic "
                      f"{DEFAULT_FALLBACK_RETURN*100:.0f}% fallback**, not a real researched number: {', '.join(sorted(fallback_set))}. "
                      f"Do not treat any Monte Carlo comparison that hinges on differences between these tickers and others as "
                      f"fundamentals-driven - add them to `{DEFAULT_EXPECTED_RETURNS_FILE.name}` first.\n")
    lines.append("| Ticker | Assumed Annual Return |")
    lines.append("|:---|---:|")
    for t, r in sorted(mc["expected_returns_used"].items(), key=lambda x: -x[1]):
        marker = " *" if t in fallback_set else ""
        lines.append(f"| {t}{marker} | {r*100:.1f}% |")
    lines.append("")

    lines.append("> [!WARNING]")
    lines.append("> This simulation uses HISTORICAL volatility and correlation (real, measured) combined with FORWARD-LOOKING expected returns (estimated, debatable). The spread between the 10th and 90th percentile outcomes is the real story here - a 40-year horizon has enormous path uncertainty. No one can know the true expected returns in advance; small changes to these assumptions materially change the projected outcomes. Treat the whole distribution as the answer, not any single number.\n")

    return "\n".join(lines)


def main():
    args = parse_args()
    holdings = load_holdings(args.holdings_file)
    tickers = [h["ticker"] for h in holdings]

    if args.weekly_contribution > 0 and args.starting_value is None:
        print("[!] --weekly-contribution was set but --starting-value was not.")
        print("    Mixing a normalized (1.0) starting value with real contribution dollars")
        print("    produces meaningless results. Defaulting --starting-value to 1000")
        print("    (interpret results as 'per 1000 currency units invested today').")
        print("    Pass --starting-value <your real portfolio value> for accurate absolute numbers.\n")
        starting_value = 1000.0
    elif args.starting_value is not None:
        starting_value = args.starting_value
    else:
        starting_value = 1.0

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
        starting_value=starting_value,
    )

    report = build_report(holdings, backtest, mc, args.sim_years, args.n_sims, args.weekly_contribution, args.years_back, starting_value)
    out_path = REPORTS_DIR / f"backtest_forecast_{datetime.now().strftime('%Y-%m-%d_%H%M')}.md"

    # Build the assumption diff BEFORE writing this run's file, so
    # find_previous_report() correctly finds the prior run and not this one.
    diff_lines = build_assumption_diff_section(mc, out_path)
    report_with_diff = report + "\n" + "\n".join(diff_lines)

    out_path.write_text(report_with_diff, encoding="utf-8")
    print(f"[+] Report written to {out_path.resolve()}")
    print("\n" + "="*70)
    print(report[:2000])
    print("="*70)


if __name__ == "__main__":
    main()
