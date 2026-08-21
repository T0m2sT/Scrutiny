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


def main():
    args = parse_args()
    print(f"[*] Initializing engine (Base Currency: {args.base_currency.upper()})...")
    fmp = FMPClient()
    print("[*] TODO: load holdings, fetch data, compute analytics, write report.")


if __name__ == "__main__":
    main()
