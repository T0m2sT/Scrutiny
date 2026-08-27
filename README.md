# Portfolio Tracker

Two scripts for analyzing and stress-testing a stock portfolio defined as target weights in a CSV.

- `analyze.py` — pulls current fundamentals/valuation/risk metrics per holding (via Financial Modeling Prep) and writes a Markdown report.
- `backtest_and_forecast.py` — pulls historical prices (via yfinance), backtests the portfolio, and runs a Monte Carlo forward simulation.

## Setup

```bash
pip install -r requirements.txt
```

`analyze.py` needs a Financial Modeling Prep API key. Create a `.env` file (gitignored) or export it directly:

```
API_KEY=your_fmp_api_key
```

## Holdings file

Both scripts take a CSV of `ticker,name,target_weight` (weights should sum to 1.0). See `holdings.example.csv`:

```csv
ticker,name,target_weight
AAPL,Apple,0.25
MSFT,Microsoft,0.25
JNJ,Johnson & Johnson,0.25
VNQ,Vanguard Real Estate ETF,0.25
```

## Running the fundamentals/risk report

```bash
python3 analyze.py holdings.csv --base-currency USD --rf-rate 0.045
```

Flags:
- `holdings_file` (positional, optional) — path to holdings CSV, defaults to `holdings.csv`
- `--base-currency` — currency to normalize all holdings into (default `USD`)
- `--rf-rate` — annual risk-free rate for Sharpe ratio (default `0.045`)

## Expected returns file (optional)

`backtest_and_forecast.py` uses per-ticker expected annual returns to drive the Monte Carlo forecast. By default it looks for `expected_returns.csv` (gitignored, since these are your own debatable estimates) with `ticker,expected_return`. See `expected_returns.example.csv`:

```csv
ticker,expected_return
AAPL,0.11
MSFT,0.12
JNJ,0.08
VNQ,0.07
```

Any ticker missing from the file falls back to a generic 10% estimate, and the report flags this loudly so it's never silently assumed.

## Running the backtest + Monte Carlo forecast

```bash
python3 backtest_and_forecast.py holdings.csv \
  --years-back 10 \
  --sim-years 40 \
  --n-sims 5000 \
  --backtest-mode common \
  --weekly-contribution 50 \
  --starting-value 1300 \
  --rf-rate 0.045
```

Flags:
- `holdings_file` (positional, required) — path to holdings CSV
- `--years-back` — years of price history to pull for the backtest/covariance (default `10`)
- `--sim-years` — forward Monte Carlo horizon in years (default `40`)
- `--n-sims` — number of Monte Carlo paths (default `5000`)
- `--backtest-mode` — `common` (trim to shortest available history across all tickers, for a fair comparison) or `full` (use each ticker's full history; portfolio math still enforces the common overlap) (default `common`)
- `--weekly-contribution` — optional weekly DCA amount in base currency layered into the forward simulation, e.g. `50` (default `0`)
- `--starting-value` — actual current portfolio value in base currency, e.g. `1300`. Required if `--weekly-contribution` is nonzero; otherwise defaults to a growth-multiple view (`1.0`)
- `--rf-rate` — risk-free rate for Sharpe ratio (default `0.045`)

Reports are written to `reports/` as timestamped Markdown files.
