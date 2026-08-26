# Portfolio Investigation & Risk Report — 2026-08-26 19:30

## Executive Summary

| Metric | Portfolio Value | Benchmark / Context |
|:---|:---|:---|
| **Portfolio Base Currency** | **USD** | All foreign assets normalized dynamically |
| **Concentration (HHI)** | **0.2500** (High Concentration) | < 0.10 = Broad, > 0.15 = Concentrated |
| **Top 5 Holdings Weight** | **100.0%** | Target < 40-50% |
| **Weighted Portfolio Beta** | **0.85** | 1.0 = S&P 500 Market Beta |
| **Weighted P/E (Harmonic TTM)** | **31.19x** | S&P 500 ~ 25-28x |
| **Weighted Forward P/E** | **24.23x** | S&P 500 ~ 20-22x |
| **Weighted Dividend Yield** | **1.0%** | S&P 500 ~ 1.3-1.5% |
| **Weighted Revenue Growth (YoY)** | **13.6%** | Portfolio weighted average |
| **Weighted Operating Margin** | **35.6%** | Portfolio weighted average |
| **1-Year FX-Adjusted Return (Simulated)** | **+26.7%** | Denominated in USD |
| **1-Year Annualized Volatility** | **12.3%** | S&P 500 ~ 13-16% |
| **Sharpe Ratio (1Y, Rf=4.5%)** | **1.81** | > 1.0 = Good, > 2.0 = Excellent |
| **Max Drawdown (1Y Peak-to-Trough)** | **-7.3%** | Deepest simulated drawdown in 1Y |
| **Avg Pairwise Stock Correlation** | **0.074** | Lower indicates higher diversification benefit |

## Per-Stock Fundamental & Valuation Battery (Normalized to USD)

| Ticker | Name | Target% | Local Price | Price (USD) | P/E (TTM) | Fwd P/E | PEG | Div Yield | Analyst Target | Upside% | Simple FV (15x) | Rev Growth | Op Margin | Beta | 5Y CAGR | Sector |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `AAPL` | Apple | 25.0% | $312.97 | $312.97 | 35.93 | 32.81 | 2.50 | 0.3% | $324.45 | +3.7% | $143.08 | +16.4% | 32.6% | 1.09 | 17.8% | Technology |
| `MSFT` | Microsoft | 25.0% | $494.46 | $494.46 | 27.54 | 20.97 | 1.60 | 0.7% | $569.45 | +15.2% | $353.60 | +17.7% | 45.1% | 1.10 | 12.8% | Technology |
| `JNJ` | Johnson & Johnson | 25.0% | $269.71 | $269.71 | 31.25 | 21.91 | 4.78 | 1.9% | $272.50 | +1.0% | $184.61 | +6.6% | 29.2% | 0.23 | 13.9% | Healthcare |
| `VNQ` | Vanguard Real Estate | 25.0% | $98.69 | $98.69 | 31.13 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.99* | 3.2% | Unknown / Other |

> [!NOTE]
> **Valuation & Fallback Legend:**
> 1. **Values with `*`**: Sourced via secondary FMP API fallback or calculated via FX-normalized EPS after failing Yahoo's plausibility guards. For **Upside%**, this also includes cases where Yahoo's raw target/price implied an implausible upside (outside -90% to +150%, usually a currency-mismatch artifact) and was recomputed from FX-normalized base-currency prices, or replaced with an FMP consensus target.
> 2. **Multi-Currency Normalization**: Foreign assets (e.g. `HTWS.L` in GBp pence, `KAP.L` in KZT/USD) are normalized to `USD` using live exchange rates.
> 3. **Simple FV (15x Forward EPS)**: Baseline Graham anchor in base currency for relative cross-checking.

## Data Provenance & Fallback Tracking

- **Vanguard Real Estate ETF (`VNQ`)**: Fallback invoked for `beta` (FMP Profile Beta)

## Sector Exposure & Allocation

| Sector | Weight (%) | Cumulative Weight | Positions Count |
|:---|---:|---:|---:|
| Technology | 50.0% | 50.0% | 2 |
| Healthcare | 25.0% | 75.0% | 1 |
| Unknown / Other | 25.0% | 100.0% | 1 |

## Data Quality & Action Items

- All holdings successfully resolved live market quotes.