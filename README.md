# AI-Directed Portfolio Construction Toolkit

Screens a 60-ETF universe across asset classes and constructs a risk-weighted portfolio using inverse-volatility allocation.

## What it does

1. **Fetch** (`fetch.py`) — pulls 1 year of daily price data for 60 ETFs via Yahoo Finance
2. **Screen** (`screen.py`) — scores each ETF on return, Sharpe ratio, volatility, drawdown, and momentum
3. **Allocate** (`allocate.py`) — selects the top 15 by composite score and allocates capital using inverse-volatility weighting

## Run it

```bash
pip install yfinance duckdb pandas numpy
python3 fetch.py
python3 screen.py
python3 allocate.py
```

## Key concepts

- **Sharpe ratio**: return per unit of risk. Weighted at 35% in the composite score — the most important metric
- **Inverse-volatility weighting**: lower-volatility ETFs receive higher allocations, controlling concentration risk systematically
- **Momentum**: recent 3-month vs 6-month trend to assess whether performance is accelerating or fading

## Universe

58 ETFs across 8 categories: Broad Market, Growth & Tech, Value, Sector, International, Fixed Income, Real Assets, Thematic/AI
