# Portfolio Backtest & Forward Simulation Report — 2026-08-26 19:31

## Part 1 — Historical Backtest

**Backtest window:** 2016-08-26 to 2026-08-26 (~10.0 years, fixed at the requested --years-back)

| Strategy | CAGR | Annualized Vol | Sharpe (Rf=4.5%) | Max Drawdown |
|:---|---:|---:|---:|---:|
| Buy & Hold (no rebalancing) | 18.92% | 18.43% | 0.78 | -31.32% |
| Monthly Rebalanced to Target | 18.92% | 18.43% | 0.78 | -31.32% |

**Included tickers:** AAPL, MSFT, JNJ, VNQ

> [!WARNING]
> A historical backtest shows what these STOCKS actually did over this window - it does not mean today's business, thesis, or valuation will repeat those results. Several holdings (e.g. recent spinoffs) have short trading histories, which forced this backtest into a shorter common window than the full portfolio's 40-year intended horizon. Treat this as a sanity check on volatility and drawdown behavior, not a return forecast.

## Part 2 — Forward Monte Carlo Simulation

**Horizon:** 40 years | **Simulations:** 5,000 | **Starting value:** €1 | **Weekly contribution modeled:** None (lump sum only)

| Outcome | Ending Portfolio Multiple | Notes |
|:---|---:|:---|
| 10th percentile (bad luck case) | 7.5x | 1-in-10 chance of doing *worse* than this |
| 25th percentile | 14.4x | |
| **Median (50th percentile)** | **30.0x** | Half of simulations landed above, half below |
| 75th percentile | 63.0x | |
| 90th percentile (good luck case) | 122.2x | 1-in-10 chance of doing *better* than this |
| Mean (average) | 53.4x | Skewed higher than median by compounding tail |

### Expected annual return assumptions used per stock

These are debatable, sourced estimates, not facts - override any of them if you disagree.

> [!CAUTION]
> **3 ticker(s) below (marked `*`) have NO curated estimate and are using the generic 10% fallback**, not a real researched number: AAPL, JNJ, VNQ. Do not treat any Monte Carlo comparison that hinges on differences between these tickers and others as fundamentals-driven - add them to `DEFAULT_EXPECTED_RETURNS` first.

| Ticker | Assumed Annual Return |
|:---|---:|
| MSFT | 12.0% |
| AAPL * | 10.0% |
| JNJ * | 10.0% |
| VNQ * | 10.0% |

> [!WARNING]
> This simulation uses HISTORICAL volatility and correlation (real, measured) combined with FORWARD-LOOKING expected returns (estimated, debatable). The spread between the 10th and 90th percentile outcomes is the real story here - a 40-year horizon has enormous path uncertainty. No one can know the true expected returns in advance; small changes to these assumptions materially change the projected outcomes. Treat the whole distribution as the answer, not any single number.

## Assumption Diff vs. Previous Run

Compared against: `backtest_forecast_2026-08-23_0111.md`

**Assumptions DIFFER from the previous run** - a Monte Carlo difference here may reflect a real, deliberate change, not just noise. Details:

New tickers in this run (not in previous): AAPL, JNJ, VNQ
Tickers dropped since previous run: AVGO, CB, CEG, GE, GEV, HWM, ISRG, KAP.L, KLAC, LLY, MA, NVDA, ORCL, PGR, PM, SCHW, TJX, TRGP, UBER, UNH, V, VRT, VRTX
