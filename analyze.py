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
                print(" Done")
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

    print("[*] TODO: compute portfolio analytics and write report.")


if __name__ == "__main__":
    main()
