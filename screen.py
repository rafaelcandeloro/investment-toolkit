import duckdb
import pandas as pd
import numpy as np
from universe import ETF_UNIVERSE

DB = "portfolio.duckdb"
RISK_FREE_RATE = 0.045  # current ~4.5% T-bill rate

db = duckdb.connect(DB, read_only=True)
prices_raw = db.execute("SELECT * FROM prices ORDER BY ticker, Date").fetchdf()
db.close()

results = []

for ticker in prices_raw["ticker"].unique():
    df = prices_raw[prices_raw["ticker"] == ticker].copy()
    df = df.sort_values("Date").reset_index(drop=True)

    if len(df) < 50:
        continue

    closes = df["Close"]

    # 1-year return
    ret_1y = (closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0]

    # 3-month and 6-month returns (momentum)
    ret_3m = (closes.iloc[-1] - closes.iloc[-63]) / closes.iloc[-63] if len(df) >= 63 else None
    ret_6m = (closes.iloc[-1] - closes.iloc[-126]) / closes.iloc[-126] if len(df) >= 126 else None

    # Annualized volatility (standard deviation of daily returns * sqrt(252))
    daily_returns = closes.pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252)

    # Sharpe ratio: (annual return - risk free rate) / annual volatility
    sharpe = (ret_1y - RISK_FREE_RATE) / volatility if volatility > 0 else 0

    # Max drawdown: biggest peak-to-trough drop over the year
    rolling_max = closes.cummax()
    drawdowns = (closes - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()

    # Momentum: 3-month return minus 6-month return (is recent trend accelerating?)
    momentum = (ret_3m - ret_6m) if ret_3m and ret_6m else 0

    # Find which category this ETF belongs to
    category = next((cat for cat, tickers in ETF_UNIVERSE.items() if ticker in tickers), "Other")

    results.append({
        "ticker": ticker,
        "category": category,
        "return_1y": round(ret_1y * 100, 1),
        "return_3m": round(ret_3m * 100, 1) if ret_3m else None,
        "return_6m": round(ret_6m * 100, 1) if ret_6m else None,
        "volatility": round(volatility * 100, 1),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(max_drawdown * 100, 1),
        "momentum": round(momentum * 100, 2) if momentum else 0,
        "price": round(closes.iloc[-1], 2),
    })

df_results = pd.DataFrame(results)

# Composite score: rank each metric, then average the ranks
# Higher is better for return, sharpe, momentum
# Lower is better for volatility, drawdown
df_results["rank_return"]   = df_results["return_1y"].rank(ascending=True)
df_results["rank_sharpe"]   = df_results["sharpe"].rank(ascending=True)
df_results["rank_momentum"] = df_results["momentum"].rank(ascending=True)
df_results["rank_vol"]      = df_results["volatility"].rank(ascending=False)
df_results["rank_dd"]       = df_results["max_drawdown"].rank(ascending=False)

df_results["score"] = (
    df_results["rank_return"] * 0.25 +
    df_results["rank_sharpe"] * 0.35 +
    df_results["rank_momentum"] * 0.20 +
    df_results["rank_vol"] * 0.10 +
    df_results["rank_dd"] * 0.10
)

df_results = df_results.sort_values("score", ascending=False).reset_index(drop=True)
df_results["rank"] = df_results.index + 1

# Save screened results to database
db = duckdb.connect(DB)
db.execute("CREATE OR REPLACE TABLE screened AS SELECT * FROM df_results")
db.close()

# Print top 20
print("=" * 85)
print("ETF SCREENING RESULTS — RANKED BY COMPOSITE SCORE")
print(f"Risk-free rate assumption: {RISK_FREE_RATE*100:.1f}% (current T-bill)")
print("=" * 85)
print(f"\n{'Rank':<5} {'Ticker':<7} {'Category':<22} {'1Y Ret':>7} {'Sharpe':>7} {'Vol':>6} {'Drawdn':>7} {'Mmtm':>6}")
print("-" * 75)

for _, row in df_results.head(20).iterrows():
    print(
        f"{int(row['rank']):<5} {row['ticker']:<7} {row['category']:<22} "
        f"{row['return_1y']:>+6.1f}% {row['sharpe']:>7.2f} "
        f"{row['volatility']:>5.1f}% {row['max_drawdown']:>+6.1f}% {row['momentum']:>+5.1f}%"
    )

print(f"\nSharpe > 1.0 = strong risk-adjusted return (beating risk-free rate per unit of risk)")
print(f"Momentum > 0 = recent 3-month trend stronger than 6-month trend (accelerating)")
print(f"Max Drawdown = worst peak-to-trough drop over the past year")
