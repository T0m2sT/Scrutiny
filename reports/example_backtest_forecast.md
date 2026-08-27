# Portfolio Backtest & Forward Simulation Report — 2026-08-27 03:15

## Part 1 — Historical Backtest

**Backtest window:** 2016-08-29 to 2026-08-26 (~10.0 years, fixed at the requested --years-back)

| Strategy | CAGR | Annualized Vol | Sharpe (Rf=4.5%) | Max Drawdown |
|:---|---:|---:|---:|---:|
| Buy & Hold (no rebalancing) | 18.90% | 18.43% | 0.78 | -31.32% |
| Monthly Rebalanced to Target | 18.90% | 18.43% | 0.78 | -31.32% |

**Included tickers:** AAPL, MSFT, JNJ, VNQ

> [!WARNING]
> A historical backtest shows what these STOCKS actually did over this window - it does not mean today's business, thesis, or valuation will repeat those results. Several holdings (e.g. recent spinoffs) have short trading histories, which forced this backtest into a shorter common window than the full portfolio's 40-year intended horizon. Treat this as a sanity check on volatility and drawdown behavior, not a return forecast.

## Part 2 — Forward Monte Carlo Simulation

**Horizon:** 40 years | **Simulations:** 5,000 | **Starting value:** €1 | **Weekly contribution modeled:** None (lump sum only)

| Outcome | Ending Portfolio Multiple | Notes |
|:---|---:|:---|
| 10th percentile (bad luck case) | 5.1x | 1-in-10 chance of doing *worse* than this |
| 25th percentile | 9.8x | |
| **Median (50th percentile)** | **20.6x** | Half of simulations landed above, half below |
| 75th percentile | 43.5x | |
| 90th percentile (good luck case) | 85.3x | 1-in-10 chance of doing *better* than this |
| Mean (average) | 37.1x | Skewed higher than median by compounding tail |

### Expected annual return assumptions used per stock

These are debatable, sourced estimates, not facts - override any of them if you disagree.

| Ticker | Assumed Annual Return |
|:---|---:|
| MSFT | 12.0% |
| AAPL | 11.0% |
| JNJ | 8.0% |
| VNQ | 7.0% |

> [!WARNING]
> This simulation uses HISTORICAL volatility and correlation (real, measured) combined with FORWARD-LOOKING expected returns (estimated, debatable). The spread between the 10th and 90th percentile outcomes is the real story here - a 40-year horizon has enormous path uncertainty. No one can know the true expected returns in advance; small changes to these assumptions materially change the projected outcomes. Treat the whole distribution as the answer, not any single number.

## Assumption Diff vs. Previous Run

Compared against: `backtest_forecast_2026-08-23_0111.md`

**Assumptions DIFFER from the previous run** - a Monte Carlo difference here may reflect a real, deliberate change, not just noise. Details:

New tickers in this run (not in previous): AAPL, JNJ, VNQ
Tickers dropped since previous run: AVGO, CB, CEG, GE, GEV, HWM, ISRG, KAP.L, KLAC, LLY, MA, NVDA, ORCL, PGR, PM, SCHW, TJX, TRGP, UBER, UNH, V, VRT, VRTX
