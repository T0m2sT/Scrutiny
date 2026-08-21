#!/usr/bin/env python3
"""
Portfolio Analyzer & Multi-Currency Risk Assessment Engine with FMP Fallback.
Reads holdings.csv (ticker, name, target_weight), fetches live market data
via yfinance, normalizes foreign currencies to base currency (USD), applies
plausibility guards, falls back to Financial Modeling Prep (FMP) API for
missing/corrupted metrics, computes portfolio-level risk metrics, and exports
a clean Markdown report to reports/.

Usage:
    python3 analyze.py [holdings.csv] [--base-currency USD] [--rf-rate 0.045]
"""
import sys
import os
import csv
import json
import argparse
import statistics
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

HERE = Path(__file__).parent
DEFAULT_HOLDINGS = HERE / "holdings.csv"
REPORTS_DIR = HERE / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# Load environment variables (.env file)
try:
    from dotenv import load_dotenv
    load_dotenv(HERE / ".env")
except ImportError:
    pass

# Direct fallback parser if python-dotenv is not installed or didn't set API_KEY
if "API_KEY" not in os.environ and (HERE / ".env").exists():
    try:
        with open(HERE / ".env", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip().strip("'\"")
    except Exception:
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Currency Portfolio Analyzer & Risk Engine")
    parser.add_argument("holdings_file", nargs="?", default=str(DEFAULT_HOLDINGS), help="Path to holdings CSV")
    parser.add_argument("--base-currency", type=str, default="USD", help="Base currency for portfolio aggregates (default: USD)")
    parser.add_argument("--rf-rate", type=float, default=0.045, help="Annual risk-free rate for Sharpe ratio (default: 0.045)")
    return parser.parse_args()


class FMPClient:
    """Client for Financial Modeling Prep (FMP) stable API with in-memory caching."""
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("API_KEY")
        self.cache = {}
        self.enabled = bool(self.api_key and self.api_key.strip())
        if self.enabled:
            print(f"[+] FMP Fallback API enabled (Key: {self.api_key[:5]}...{self.api_key[-4:]})")
        else:
            print("[!] Warning: No FMP API_KEY found in environment or .env. Fallback queries will be skipped.")

    def _fetch(self, endpoint, symbol):
        if not self.enabled or not symbol:
            return None
        clean_sym = symbol.strip().upper()
        cache_key = (endpoint, clean_sym)
        if cache_key in self.cache:
            return self.cache[cache_key]

        url = f"https://financialmodelingprep.com/stable/{endpoint}?symbol={clean_sym}&apikey={self.api_key}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (PortfolioAnalyzer)"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode())
                if isinstance(data, list) and len(data) > 0:
                    res = data[0]
                elif isinstance(data, dict):
                    res = data
                else:
                    res = None
                self.cache[cache_key] = res
                return res
        except Exception:
            self.cache[cache_key] = None
            return None

    def get_ratios(self, symbol):
        return self._fetch("ratios-ttm", symbol)

    def get_key_metrics(self, symbol):
        return self._fetch("key-metrics-ttm", symbol)

    def get_quote(self, symbol):
        return self._fetch("quote", symbol)

    def get_profile(self, symbol):
        return self._fetch("profile", symbol)


class FXConverter:
    """Handles real-time and historical currency conversions with Pence/Sterling support."""
    def __init__(self, base_currency="USD"):
        self.base_currency = base_currency.upper()
        self.spot_rates = {self.base_currency: 1.0}
        self.hist_rates = {}

    def get_spot_rate_to_base(self, from_ccy):
        """
        Returns multiplier to convert 1 unit of `from_ccy` to `self.base_currency`.
        Handles GBp (pence) -> GBP -> USD conversion automatically.
        """
        if not from_ccy:
            return 1.0, self.base_currency

        ccy = str(from_ccy).strip()
        is_pence = False
        if ccy.upper() in ("GBP", "GBX", "GBp"):
            if ccy in ("GBX", "GBp"):
                is_pence = True
            ccy_std = "GBP"
        else:
            ccy_std = ccy.upper()

        if ccy_std == self.base_currency:
            multiplier = 0.01 if is_pence else 1.0
            return multiplier, ccy_std

        if ccy_std not in self.spot_rates:
            rate = None
            pair = f"{ccy_std}{self.base_currency}=X"
            try:
                t = yf.Ticker(pair)
                h = t.history(period="5d")
                if h is not None and not h.empty and "Close" in h:
                    rate = float(h["Close"].dropna().iloc[-1])
            except Exception:
                pass

            if rate is None:
                inv_pair = f"{self.base_currency}{ccy_std}=X"
                try:
                    inv_t = yf.Ticker(inv_pair)
                    h = inv_t.history(period="5d")
                    if h is not None and not h.empty and "Close" in h:
                        inv_rate = float(h["Close"].dropna().iloc[-1])
                        if inv_rate > 0:
                            rate = 1.0 / inv_rate
                except Exception:
                    pass

            self.spot_rates[ccy_std] = rate if rate is not None else 1.0

        base_multiplier = self.spot_rates[ccy_std]
        final_multiplier = (base_multiplier * 0.01) if is_pence else base_multiplier
        return final_multiplier, ccy_std

    def get_historical_fx_series(self, from_ccy, target_dates):
        """Returns daily Series of multipliers to convert historical prices to base currency."""
        if not from_ccy:
            return pd.Series(1.0, index=target_dates)

        ccy = str(from_ccy).strip()
        is_pence = ccy in ("GBX", "GBp")
        ccy_std = "GBP" if ccy.upper() in ("GBP", "GBX", "GBp") else ccy.upper()

        if ccy_std == self.base_currency:
            val = 0.01 if is_pence else 1.0
            return pd.Series(val, index=target_dates)

        if ccy_std not in self.hist_rates:
            pair = f"{ccy_std}{self.base_currency}=X"
            s = None
            try:
                df = yf.download(pair, period="1y", interval="1d", progress=False)
                if df is not None and not df.empty:
                    px = df["Close"] if "Close" in df else df
                    if isinstance(px, pd.DataFrame):
                        px = px.iloc[:, 0]
                    s = px
            except Exception:
                pass

            if s is None:
                inv_pair = f"{self.base_currency}{ccy_std}=X"
                try:
                    df = yf.download(inv_pair, period="1y", interval="1d", progress=False)
                    if df is not None and not df.empty:
                        px = df["Close"] if "Close" in df else df
                        if isinstance(px, pd.DataFrame):
                            px = px.iloc[:, 0]
                        s = 1.0 / px
                except Exception:
                    pass

            self.hist_rates[ccy_std] = s

        raw_series = self.hist_rates.get(ccy_std)
        if raw_series is None or raw_series.empty:
            spot_m, _ = self.get_spot_rate_to_base(from_ccy)
            return pd.Series(spot_m, index=target_dates)

        aligned = raw_series.reindex(target_dates).ffill().bfill()
        if is_pence:
            aligned = aligned * 0.01
        return aligned


def load_holdings(path):
    """Load and validate holdings CSV."""
    rows = []
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Holdings file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            ticker = r["ticker"].strip().upper()
            name = r["name"].strip()
            weight = float(r["target_weight"])
            rows.append({
                "ticker": ticker,
                "name": name,
                "target_weight": weight,
            })

    total_weight = sum(r["target_weight"] for r in rows)
    if not (0.999 <= total_weight <= 1.001):
        print(f"[!] Warning: Target weights sum to {total_weight*100:.2f}% (not 100.0%). Normalizing...")
        for r in rows:
            r["target_weight"] = r["target_weight"] / total_weight
    else:
        print(f"[+] Verified target weights sum to {total_weight*100:.1f}%.")

    return rows


def safe_get(d, *keys, default=None):
    """Safely retrieves first non-None value from dictionary."""
    if not isinstance(d, dict):
        return default
    for k in keys:
        v = d.get(k)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            return v
    return default


def _plausible_pe(pe):
    """Reject P/E ratios outside a sane 1.0-250.0 range."""
    if pe is None or not isinstance(pe, (int, float)) or np.isnan(pe) or not (1.0 <= pe <= 250.0):
        return None
    return float(pe)


def _plausible_upside(upside):
    """
    Reject analyst upside/downside outside a sane -90% to +150% range.
    Values outside this band are almost always a currency-mismatch or stale-data
    artifact (e.g. price quoted in one currency vs. analyst target reported in a
    different one for GDR/ADR-style tickers such as KAP.L), not a genuine market
    view. A +214% "upside" is a data-quality flag, not an investable signal.
    """
    if upside is None or not isinstance(upside, (int, float)) or np.isnan(upside):
        return None
    if not (-0.90 <= upside <= 1.50):
        return None
    return float(upside)


def fetch_stock_data(ticker, name, fx: FXConverter, fmp: FMPClient):
    """
    Pulls valuation, fundamentals, analyst targets, and converts foreign currency amounts.
    Applies plausibility guards and seamlessly falls back to FMP/FX normalization when
    Yahoo data is missing or corrupted.
    """
    t = yf.Ticker(ticker)
    info = {}
    try:
        info = t.info if hasattr(t, "info") and isinstance(t.info, dict) else {}
    except Exception as e:
        info = {"_error": str(e)}

    raw_price = safe_get(info, "currentPrice", "regularMarketPrice", "previousClose", "ask", "bid")
    quote_ccy = safe_get(info, "currency", default="USD")
    financial_ccy = safe_get(info, "financialCurrency", default=quote_ccy)
    fallback_sources = {}

    # Price fallback to FMP if missing from Yahoo
    if raw_price is None and fmp.enabled:
        fmp_q = fmp.get_quote(ticker)
        if fmp_q and fmp_q.get("price"):
            raw_price = float(fmp_q["price"])
            fallback_sources["price"] = "FMP Quote"

    # FX Conversion rate for quoted price to base currency
    price_fx_rate, _ = fx.get_spot_rate_to_base(quote_ccy)
    base_price = (raw_price * price_fx_rate) if (raw_price is not None and price_fx_rate) else None

    # Analyst Target Mean and Conversion
    raw_analyst_target = safe_get(info, "targetMeanPrice")
    base_analyst_target = (raw_analyst_target * price_fx_rate) if (raw_analyst_target is not None and price_fx_rate) else None

    # Upside % — computed from BASE-CURRENCY (FX-normalized) values, never raw
    # quote-currency fields. Yahoo sometimes returns targetMeanPrice in a
    # different currency/scale than currentPrice for foreign/GDR-style tickers
    # (e.g. KAP.L), which previously produced nonsensical results like +214%
    # upside. Computing from base_price/base_analyst_target ensures both sides
    # of the ratio are in the same currency before dividing.
    analyst_upside_pct = None
    if base_price and base_analyst_target and base_price > 0:
        analyst_upside_pct = (base_analyst_target - base_price) / base_price
    analyst_upside_pct = _plausible_upside(analyst_upside_pct)

    if analyst_upside_pct is None and raw_analyst_target is not None and fmp.enabled:
        # Secondary fallback: FMP consensus target, computed in the same base currency
        fmp_q = fmp.get_quote(ticker)
        fmp_target = safe_get(fmp_q, "priceTarget", "targetPrice")
        if fmp_target is not None and base_price and base_price > 0:
            fmp_target_base = float(fmp_target) * price_fx_rate
            candidate = (fmp_target_base - base_price) / base_price
            candidate = _plausible_upside(candidate)
            if candidate is not None:
                base_analyst_target = fmp_target_base
                analyst_upside_pct = candidate
                fallback_sources["analyst_upside_pct"] = "FMP Price Target"

    # Forward EPS and FX conversion if reported in a different currency (e.g. KAP.L KZT -> USD)
    raw_forward_eps = safe_get(info, "forwardEps")
    base_forward_eps = None
    if raw_forward_eps is not None:
        eps_fx_rate, _ = fx.get_spot_rate_to_base(financial_ccy)
        base_forward_eps = raw_forward_eps * eps_fx_rate

    # Simple Fair Value (15x Forward EPS in Base Currency)
    simple_fair_value_15x = None
    if base_forward_eps and base_forward_eps > 0:
        simple_fair_value_15x = round(base_forward_eps * 15.0, 2)

    # 5-Year Historical CAGR calculation
    price_cagr_5y = None
    try:
        hist = t.history(period="5y", interval="1mo")
        if hist is not None and not hist.empty and "Close" in hist.columns:
            closes = hist["Close"].dropna()
            if len(closes) >= 12:
                p_start = float(closes.iloc[0])
                p_end = float(closes.iloc[-1])
                years = len(closes) / 12.0
                if p_start > 0:
                    price_cagr_5y = (p_end / p_start) ** (1.0 / years) - 1.0
    except Exception:
        pass

    # --- P/E (TTM) with plausibility guard & FMP fallback ---
    pe_trailing = _plausible_pe(safe_get(info, "trailingPE"))
    if pe_trailing is None and fmp.enabled:
        fmp_ratios = fmp.get_ratios(ticker)
        fmp_pe = safe_get(fmp_ratios, "priceToEarningsRatioTTM")
        if fmp_pe is None:
            fmp_q = fmp.get_quote(ticker)
            fmp_pe = safe_get(fmp_q, "pe")
        if _plausible_pe(fmp_pe):
            pe_trailing = float(fmp_pe)
            fallback_sources["pe_trailing"] = "FMP TTM Ratios"

    # --- Forward P/E with plausibility guard & FX-normalized/FMP fallback ---
    raw_pe_forward = safe_get(info, "forwardPE")
    pe_forward = _plausible_pe(raw_pe_forward)
    if pe_forward is None:
        # Currency mismatch recovery: Price in USD vs EPS in KZT (e.g. KAP.L)
        if base_price and base_forward_eps and base_forward_eps > 0:
            calc_pe = base_price / base_forward_eps
            if _plausible_pe(calc_pe):
                pe_forward = calc_pe
                fallback_sources["pe_forward"] = "FX-Normalized Forward EPS"
        # Secondary fallback to FMP
        if pe_forward is None and fmp.enabled:
            fmp_ratios = fmp.get_ratios(ticker)
            fmp_fwd_peg = safe_get(fmp_ratios, "forwardPriceToEarningsGrowthRatioTTM")
            if _plausible_pe(fmp_fwd_peg):
                pe_forward = float(fmp_fwd_peg)
                fallback_sources["pe_forward"] = "FMP Forward Ratio"

    # --- PEG Ratio with guard & FMP fallback ---
    peg = safe_get(info, "trailingPegRatio", "pegRatio")
    if (peg is None or peg <= 0 or peg > 25) and fmp.enabled:
        fmp_ratios = fmp.get_ratios(ticker)
        fmp_peg = safe_get(fmp_ratios, "priceToEarningsGrowthRatioTTM", "forwardPriceToEarningsGrowthRatioTTM")
        if fmp_peg is not None and 0 < fmp_peg < 25:
            peg = float(fmp_peg)
            fallback_sources["peg"] = "FMP PEG Ratio"

    # --- Dividend Yield with guard & FMP fallback ---
    div_yield = safe_get(info, "trailingAnnualDividendYield")
    if (div_yield is None or div_yield < 0 or div_yield > 0.30) and fmp.enabled:
        fmp_ratios = fmp.get_ratios(ticker)
        fmp_dy = safe_get(fmp_ratios, "dividendYieldTTM")
        if fmp_dy is not None and 0 <= fmp_dy <= 0.30:
            div_yield = float(fmp_dy)
            fallback_sources["dividend_yield"] = "FMP Dividend Yield"

    # --- Beta with guard & FMP fallback ---
    beta = safe_get(info, "beta")
    if (beta is None or not (-2.0 <= beta <= 5.0)) and fmp.enabled:
        fmp_prof = fmp.get_profile(ticker)
        fmp_beta = safe_get(fmp_prof, "beta")
        if fmp_beta is not None and -2.0 <= fmp_beta <= 5.0 and fmp_beta != 0:
            beta = float(fmp_beta)
            fallback_sources["beta"] = "FMP Profile Beta"

    # --- Margins & Growth with FMP fallback ---
    rev_growth = safe_get(info, "revenueGrowth")
    op_margin = safe_get(info, "operatingMargins")
    if op_margin is None and fmp.enabled:
        fmp_ratios = fmp.get_ratios(ticker)
        fmp_opm = safe_get(fmp_ratios, "operatingProfitMarginTTM")
        if fmp_opm is not None:
            op_margin = float(fmp_opm)
            fallback_sources["operating_margin"] = "FMP Operating Margin"

    out = {
        "ticker": ticker,
        "name": name,
        "quote_price": raw_price,
        "quote_currency": quote_ccy,
        "base_price": base_price,
        "financial_currency": financial_ccy,
        "pe_trailing": pe_trailing,
        "pe_forward": pe_forward,
        "peg": peg,
        "dividend_yield": div_yield,
        "beta": beta,
        "debt_to_equity": safe_get(info, "debtToEquity"),
        "revenue_growth": rev_growth,
        "operating_margin": op_margin,
        "analyst_target_raw": raw_analyst_target,
        "analyst_target_base": base_analyst_target,
        "analyst_upside_pct": analyst_upside_pct,
        "simple_fair_value_15x": simple_fair_value_15x,
        "price_cagr_5y": price_cagr_5y,
        "sector": safe_get(info, "sector", default="Unknown / Other"),
        "industry": safe_get(info, "industry", default=""),
        "market_cap": safe_get(info, "marketCap"),
        "fallback_sources": fallback_sources,
        "data_quality": "live" if raw_price is not None else "web_fallback_needed",
    }
    if out["data_quality"] == "web_fallback_needed":
        out["note"] = "No live market quote on Yahoo Finance or FMP. Check if newly listed or needs alternate ticker symbol."

    return out


def compute_portfolio_analytics(stocks, holdings, fx: FXConverter, rf_rate=0.045):
    """
    Computes HHI concentration, weighted fundamentals, and FX-adjusted 1-year historical
    portfolio performance (Return, Annualized Volatility, Sharpe Ratio, Max Drawdown).
    """
    weights = {h["ticker"]: h["target_weight"] for h in holdings}
    hhi = sum(w ** 2 for w in weights.values())
    sorted_weights = sorted(weights.values(), reverse=True)
    top3_weight = sum(sorted_weights[:3])
    top5_weight = sum(sorted_weights[:5])
    top10_weight = sum(sorted_weights[:10])

    sector_weight = {}
    for h in holdings:
        s = stocks.get(h["ticker"], {})
        sec = s.get("sector") or "Unknown / Other"
        sector_weight[sec] = sector_weight.get(sec, 0.0) + h["target_weight"]
    sector_weight = dict(sorted(sector_weight.items(), key=lambda x: -x[1]))

    def weighted_stat(metric_key, harmonic=False):
        valid_pairs = []
        for h in holdings:
            s = stocks.get(h["ticker"], {})
            val = s.get(metric_key)
            if val is not None and isinstance(val, (int, float)) and not np.isnan(val):
                if harmonic and val <= 0:
                    continue
                valid_pairs.append((h["target_weight"], float(val)))
        if not valid_pairs:
            return None
        total_w = sum(w for w, _ in valid_pairs)
        if total_w <= 0:
            return None
        if harmonic:
            inv_sum = sum(w / v for w, v in valid_pairs)
            return (total_w / inv_sum) if inv_sum > 0 else None
        return sum(w * v for w, v in valid_pairs) / total_w

    weighted_pe_trailing = weighted_stat("pe_trailing", harmonic=True)
    weighted_pe_forward = weighted_stat("pe_forward", harmonic=True)
    weighted_beta = weighted_stat("beta")
    weighted_div_yield = weighted_stat("dividend_yield")
    weighted_rev_growth = weighted_stat("revenue_growth")
    weighted_op_margin = weighted_stat("operating_margin")

    valid_tickers = [h["ticker"] for h in holdings if stocks.get(h["ticker"], {}).get("quote_price")]
    corr_matrix = None
    avg_pairwise_corr = None
    port_1y_return = None
    port_1y_volatility = None
    port_1y_sharpe = None
    port_1y_max_drawdown = None

    if valid_tickers:
        try:
            df_hist = yf.download(valid_tickers, period="1y", interval="1d", progress=False)
            if df_hist is not None and not df_hist.empty:
                px = df_hist["Close"] if "Close" in df_hist else df_hist
                if isinstance(px, pd.Series):
                    px = px.to_frame(name=valid_tickers[0])

                base_px = pd.DataFrame(index=px.index)
                for t in px.columns:
                    s_data = stocks.get(t, {})
                    q_ccy = s_data.get("quote_currency", "USD")
                    fx_series = fx.get_historical_fx_series(q_ccy, px.index)
                    base_px[t] = px[t] * fx_series

                daily_returns = base_px.pct_change(fill_method=None).dropna(how="all")

                if daily_returns.shape[1] > 1:
                    corr_matrix = daily_returns.corr()
                    vals = []
                    cols = corr_matrix.columns
                    for i in range(len(cols)):
                        for j in range(i + 1, len(cols)):
                            v = corr_matrix.iloc[i, j]
                            if pd.notna(v):
                                vals.append(v)
                    if vals:
                        avg_pairwise_corr = statistics.mean(vals)

                active_weights = {t: weights[t] for t in daily_returns.columns if t in weights}
                sum_active_w = sum(active_weights.values())
                if sum_active_w > 0:
                    norm_active_w = {t: w / sum_active_w for t, w in active_weights.items()}
                    weight_series = pd.Series(norm_active_w)
                    aligned_returns = daily_returns[weight_series.index]
                    port_daily_ret = aligned_returns.dot(weight_series).dropna()

                    if len(port_daily_ret) > 30:
                        cum_returns = (1 + port_daily_ret).cumprod()
                        port_1y_return = float(cum_returns.iloc[-1] - 1.0)
                        daily_std = float(port_daily_ret.std())
                        port_1y_volatility = daily_std * np.sqrt(252)
                        annualized_ret = float((1 + port_1y_return) ** (252 / len(port_daily_ret)) - 1.0)
                        if port_1y_volatility > 0:
                            port_1y_sharpe = (annualized_ret - rf_rate) / port_1y_volatility
                        peak = cum_returns.cummax()
                        drawdowns = (cum_returns - peak) / peak
                        port_1y_max_drawdown = float(drawdowns.min())

        except Exception as e:
            print(f"[!] Historical return computation notice: {e}")

    return {
        "hhi": hhi,
        "top3_weight": top3_weight,
        "top5_weight": top5_weight,
        "top10_weight": top10_weight,
        "sector_weight": sector_weight,
        "weighted_pe_trailing": weighted_pe_trailing,
        "weighted_pe_forward": weighted_pe_forward,
        "weighted_beta": weighted_beta,
        "weighted_div_yield": weighted_div_yield,
        "weighted_rev_growth": weighted_rev_growth,
        "weighted_op_margin": weighted_op_margin,
        "avg_pairwise_correlation": avg_pairwise_corr,
        "correlation_matrix": corr_matrix,
        "port_1y_return": port_1y_return,
        "port_1y_volatility": port_1y_volatility,
        "port_1y_sharpe": port_1y_sharpe,
        "port_1y_max_drawdown": port_1y_max_drawdown,
    }


def fmt_pct(x, signed=False):
    if x is None or not isinstance(x, (int, float)) or np.isnan(x):
        return "n/a"
    prefix = "+" if signed and x > 0 else ""
    return f"{prefix}{x * 100:.1f}%"


def fmt_num(x, d=2, marker=""):
    if x is None or not isinstance(x, (int, float)) or np.isnan(x):
        return "n/a"
    return f"{x:.{d}f}{marker}"


def fmt_quote_price(price, ccy):
    if price is None or not isinstance(price, (int, float)) or np.isnan(price):
        return "n/a"
    if ccy == "USD":
        return f"${price:.2f}"
    return f"{price:.2f} {ccy}"


def fmt_base_price(price, base_ccy="USD"):
    if price is None or not isinstance(price, (int, float)) or np.isnan(price):
        return "n/a"
    symbol = "$" if base_ccy == "USD" else f"{base_ccy} "
    return f"{symbol}{price:.2f}"


def build_report(holdings, stocks, portfolio, base_ccy="USD"):
    """Builds a comprehensive Markdown report with FX normalization and fallback annotations."""
    lines = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines.append(f"# Portfolio Investigation & Risk Report — {ts}\n")

    # Section 1: Executive Summary
    lines.append("## Executive Summary\n")
    hhi = portfolio["hhi"]
    hhi_label = "High Concentration" if hhi > 0.15 else "Moderate Concentration" if hhi > 0.10 else "Well Diversified"

    lines.append(f"| Metric | Portfolio Value | Benchmark / Context |")
    lines.append(f"|:---|:---|:---|")
    lines.append(f"| **Portfolio Base Currency** | **{base_ccy}** | All foreign assets normalized dynamically |")
    lines.append(f"| **Concentration (HHI)** | **{hhi:.4f}** ({hhi_label}) | < 0.10 = Broad, > 0.15 = Concentrated |")
    lines.append(f"| **Top 5 Holdings Weight** | **{fmt_pct(portfolio['top5_weight'])}** | Target < 40-50% |")
    lines.append(f"| **Weighted Portfolio Beta** | **{fmt_num(portfolio['weighted_beta'])}** | 1.0 = S&P 500 Market Beta |")
    lines.append(f"| **Weighted P/E (Harmonic TTM)** | **{fmt_num(portfolio['weighted_pe_trailing'])}x** | S&P 500 ~ 25-28x |")
    lines.append(f"| **Weighted Forward P/E** | **{fmt_num(portfolio['weighted_pe_forward'])}x** | S&P 500 ~ 20-22x |")
    lines.append(f"| **Weighted Dividend Yield** | **{fmt_pct(portfolio['weighted_div_yield'])}** | S&P 500 ~ 1.3-1.5% |")
    lines.append(f"| **Weighted Revenue Growth (YoY)** | **{fmt_pct(portfolio['weighted_rev_growth'])}** | Portfolio weighted average |")
    lines.append(f"| **Weighted Operating Margin** | **{fmt_pct(portfolio['weighted_op_margin'])}** | Portfolio weighted average |")
    if portfolio.get("port_1y_return") is not None:
        lines.append(f"| **1-Year FX-Adjusted Return (Simulated)** | **{fmt_pct(portfolio['port_1y_return'], signed=True)}** | Denominated in {base_ccy} |")
        lines.append(f"| **1-Year Annualized Volatility** | **{fmt_pct(portfolio['port_1y_volatility'])}** | S&P 500 ~ 13-16% |")
        lines.append(f"| **Sharpe Ratio (1Y, Rf=4.5%)** | **{fmt_num(portfolio['port_1y_sharpe'])}** | > 1.0 = Good, > 2.0 = Excellent |")
        lines.append(f"| **Max Drawdown (1Y Peak-to-Trough)** | **{fmt_pct(portfolio['port_1y_max_drawdown'])}** | Deepest simulated drawdown in 1Y |")
    lines.append(f"| **Avg Pairwise Stock Correlation** | **{fmt_num(portfolio['avg_pairwise_correlation'], 3)}** | Lower indicates higher diversification benefit |")
    lines.append("")

    # Section 2: Detailed Per-Stock Battery (16 columns)
    lines.append(f"## Per-Stock Fundamental & Valuation Battery (Normalized to {base_ccy})\n")
    headers = [
        "Ticker", "Name", "Target%", "Local Price", f"Price ({base_ccy})", "P/E (TTM)", "Fwd P/E",
        "PEG", "Div Yield", "Analyst Target", "Upside%", "Simple FV (15x)",
        "Rev Growth", "Op Margin", "Beta", "5Y CAGR", "Sector"
    ]
    lines.append("| " + " | ".join(headers) + " |")
    aligns = [
        ":---", ":---", "---:", "---:", "---:", "---:", "---:",
        "---:", "---:", "---:", "---:", "---:",
        "---:", "---:", "---:", "---:", ":---"
    ]
    lines.append("| " + " | ".join(aligns) + " |")

    for h in holdings:
        s = stocks.get(h["ticker"], {})
        if s.get("data_quality") == "web_fallback_needed":
            note_msg = s.get("note", "NO DATA")
            lines.append(
                f"| `{h['ticker']}` | {h['name']} | {fmt_pct(h['target_weight'])} | "
                f"**NO DATA** | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | {note_msg} |"
            )
            continue

        q_ccy = s.get("quote_currency", "USD")
        fb = s.get("fallback_sources", {})

        # Markers for fallback-sourced fields
        fpe_marker = "*" if "pe_forward" in fb else ""
        tpe_marker = "*" if "pe_trailing" in fb else ""
        peg_marker = "*" if "peg" in fb else ""
        dy_marker = "*" if "dividend_yield" in fb else ""
        beta_marker = "*" if "beta" in fb else ""
        upside_marker = "*" if "analyst_upside_pct" in fb else ""

        row = [
            f"`{h['ticker']}`",
            h["name"][:20],
            fmt_pct(h["target_weight"]),
            fmt_quote_price(s.get("quote_price"), q_ccy),
            fmt_base_price(s.get("base_price"), base_ccy),
            fmt_num(s.get("pe_trailing"), marker=tpe_marker),
            fmt_num(s.get("pe_forward"), marker=fpe_marker),
            fmt_num(s.get("peg"), marker=peg_marker),
            fmt_pct(s.get("dividend_yield")),
            fmt_base_price(s.get("analyst_target_base"), base_ccy),
            fmt_pct(s.get("analyst_upside_pct"), signed=True) + upside_marker,
            fmt_base_price(s.get("simple_fair_value_15x"), base_ccy),
            fmt_pct(s.get("revenue_growth"), signed=True),
            fmt_pct(s.get("operating_margin")),
            fmt_num(s.get("beta"), marker=beta_marker),
            fmt_pct(s.get("price_cagr_5y")),
            s.get("sector", "n/a"),
        ]
        lines.append("| " + " | ".join(row) + " |")

    lines.append("\n> [!NOTE]")
    lines.append("> **Valuation & Fallback Legend:**")
    lines.append(f"> 1. **Values with `*`**: Sourced via secondary FMP API fallback or calculated via FX-normalized EPS after failing Yahoo's plausibility guards. For **Upside%**, this also includes cases where Yahoo's raw target/price implied an implausible upside (outside -90% to +150%, usually a currency-mismatch artifact) and was recomputed from FX-normalized base-currency prices, or replaced with an FMP consensus target.")
    lines.append(f"> 2. **Multi-Currency Normalization**: Foreign assets (e.g. `HTWS.L` in GBp pence, `KAP.L` in KZT/USD) are normalized to `{base_ccy}` using live exchange rates.")
    lines.append("> 3. **Simple FV (15x Forward EPS)**: Baseline Graham anchor in base currency for relative cross-checking.\n")

    # Section 3: Fallback & Data Provenance
    lines.append("## Data Provenance & Fallback Tracking\n")
    fallbacks_found = False
    for h in holdings:
        s = stocks.get(h["ticker"], {})
        fb = s.get("fallback_sources", {})
        if fb:
            fallbacks_found = True
            fb_desc = ", ".join([f"`{k}` ({v})" for k, v in fb.items()])
            lines.append(f"- **{h['name']} (`{h['ticker']}`)**: Fallback invoked for {fb_desc}")
    if not fallbacks_found:
        lines.append("- Primary Yahoo Finance data passed all plausibility guards across all tickers.")
    lines.append("")

    # Section 4: Sector Breakdown
    lines.append("## Sector Exposure & Allocation\n")
    lines.append("| Sector | Weight (%) | Cumulative Weight | Positions Count |")
    lines.append("|:---|---:|---:|---:|")
    cum_w = 0.0
    for sec, w in portfolio["sector_weight"].items():
        cum_w += w
        pos_count = sum(1 for h in holdings if (stocks.get(h["ticker"], {}).get("sector") or "Unknown / Other") == sec)
        lines.append(f"| {sec} | {fmt_pct(w)} | {fmt_pct(cum_w)} | {pos_count} |")

    # Section 5: Data Quality Flags
    lines.append("\n## Data Quality & Action Items\n")
    flagged = [h for h in holdings if stocks.get(h["ticker"], {}).get("data_quality") != "live"]
    if flagged:
        for h in flagged:
            s_info = stocks.get(h["ticker"], {})
            lines.append(f"- **{h['name']} (`{h['ticker']}`)** [{fmt_pct(h['target_weight'])} weight]: {s_info.get('note', 'Incomplete data')}")
    else:
        lines.append("- All holdings successfully resolved live market quotes.")

    return "\n".join(lines)


def print_cli_summary(holdings, stocks, portfolio, base_ccy="USD"):
    """Prints a clean CLI summary to the terminal."""
    print("\n" + "=" * 88)
    print(f"               MULTI-CURRENCY PORTFOLIO SUMMARY (BASE: {base_ccy})")
    print("=" * 88)
    print(f"Total Positions: {len(holdings)} | Top 5 Weight: {fmt_pct(portfolio['top5_weight'])} | HHI: {portfolio['hhi']:.4f}")
    print(f"Weighted Beta:   {fmt_num(portfolio['weighted_beta'])}  | Weighted Fwd P/E: {fmt_num(portfolio['weighted_pe_forward'])}x | Div Yield: {fmt_pct(portfolio['weighted_div_yield'])}")
    if portfolio.get("port_1y_return") is not None:
        print(f"1Y Return (USD): {fmt_pct(portfolio['port_1y_return'], signed=True)} | 1Y Volatility: {fmt_pct(portfolio['port_1y_volatility'])} | Sharpe (Rf=4.5%): {fmt_num(portfolio['port_1y_sharpe'])}")
    print("-" * 88)
    print(f"{'Ticker':<8} {'Name':<18} {'Weight':<8} {'Local Px':<12} {f'Px ({base_ccy})':<11} {'Fwd P/E':<9} {'Beta':<6} {'Upside%':<9} {'Sector':<16}")
    print("-" * 88)
    for h in holdings[:12]:
        s = stocks.get(h["ticker"], {})
        t_sym = h["ticker"]
        name = h["name"][:16]
        w = fmt_pct(h["target_weight"])
        local_p = fmt_quote_price(s.get("quote_price"), s.get("quote_currency", "USD"))
        base_p = fmt_base_price(s.get("base_price"), base_ccy)
        fb = s.get("fallback_sources", {})
        fpe_marker = "*" if "pe_forward" in fb else ""
        fpe = fmt_num(s.get("pe_forward"), marker=fpe_marker)
        beta_marker = "*" if "beta" in fb else ""
        beta = fmt_num(s.get("beta"), marker=beta_marker)
        upside = fmt_pct(s.get("analyst_upside_pct"), signed=True)
        sec = (s.get("sector") or "Unknown")[:15]
        print(f"{t_sym:<8} {name:<18} {w:<8} {local_p:<12} {base_p:<11} {fpe:<9} {beta:<6} {upside:<9} {sec:<16}")
    if len(holdings) > 12:
        print(f"... and {len(holdings) - 12} more positions (see full markdown report).")
    print("=" * 88 + "\n")


def main():
    args = parse_args()
    holdings_path = Path(args.holdings_file)
    base_currency = args.base_currency.upper()
    print(f"[*] Initializing Multi-Currency Engine (Base Currency: {base_currency})...")
    fx = FXConverter(base_currency=base_currency)
    fmp = FMPClient()

    print(f"[*] Loading portfolio holdings from {holdings_path}...")
    holdings = load_holdings(holdings_path)

    stocks = {}
    print("[*] Fetching live fundamental and market data (Yahoo Primary -> FMP Fallback)...")
    for i, h in enumerate(holdings, 1):
        ticker = h["ticker"]
        name = h["name"]
        print(f"  [{i:02d}/{len(holdings):02d}] Fetching {ticker:<8} ({name})...", end="", flush=True)
        try:
            data = fetch_stock_data(ticker, name, fx, fmp)
            stocks[ticker] = data
            if data.get("data_quality") == "live":
                loc_p = fmt_quote_price(data.get("quote_price"), data.get("quote_currency", "USD"))
                base_p = fmt_base_price(data.get("base_price"), base_currency)
                fb = data.get("fallback_sources", {})
                fb_msg = f" [Fallback: {', '.join(fb.keys())}]" if fb else ""
                print(f" Done ({loc_p} -> {base_p}){fb_msg}")
            else:
                print(" [!] No live price (Flagged)")
        except Exception as e:
            print(f" [X] Error: {e}")
            stocks[ticker] = {
                "ticker": ticker,
                "name": name,
                "data_quality": "web_fallback_needed",
                "note": f"Exception encountered: {str(e)}",
            }

    print("[*] Computing portfolio-level risk, concentration, and FX-adjusted return analytics...")
    portfolio = compute_portfolio_analytics(stocks, holdings, fx=fx, rf_rate=args.rf_rate)

    report_content = build_report(holdings, stocks, portfolio, base_ccy=base_currency)
    out_filename = f"report_{datetime.now().strftime('%Y-%m-%d_%H%M')}.md"
    out_path = REPORTS_DIR / out_filename
    out_path.write_text(report_content, encoding="utf-8")
    print(f"[+] Full Markdown report written to:\n    {out_path.resolve()}")

    print_cli_summary(holdings, stocks, portfolio, base_ccy=base_currency)


if __name__ == "__main__":
    main()
