import duckdb
import pandas as pd
import numpy as np

DB = "portfolio.duckdb"
PORTFOLIO_SIZE = 100_000  # dollars to allocate
TOP_N = 15                # number of ETFs to include in portfolio

db = duckdb.connect(DB, read_only=True)
screened = db.execute("SELECT * FROM screened ORDER BY rank").fetchdf()
db.close()

# Take top N ETFs from the screener
selected = screened.head(TOP_N).copy()

# Inverse-volatility weighting:
# The idea: give MORE weight to LESS volatile ETFs.
# A stock with 10% volatility gets twice the weight of one with 20% volatility.
# This controls concentration risk — you're not betting everything on the wildest names.
inv_vol = 1 / selected["volatility"]
weights = inv_vol / inv_vol.sum()
selected["weight"] = weights
selected["allocation"] = (weights * PORTFOLIO_SIZE).round(2)
selected["shares"] = (selected["allocation"] / selected["price"]).apply(np.floor).astype(int)
selected["actual_value"] = (selected["shares"] * selected["price"]).round(2)

total_deployed = selected["actual_value"].sum()
cash_remaining = PORTFOLIO_SIZE - total_deployed

print("=" * 80)
print(f"PORTFOLIO ALLOCATION — ${PORTFOLIO_SIZE:,.0f} PORTFOLIO")
print(f"Top {TOP_N} ETFs by composite score | Inverse-volatility weighted")
print("=" * 80)
print(f"\n{'Ticker':<7} {'Category':<22} {'Weight':>7} {'Alloc $':>10} {'Shares':>7} {'Price':>8}")
print("-" * 65)

for _, row in selected.iterrows():
    print(
        f"{row['ticker']:<7} {row['category']:<22} "
        f"{row['weight']*100:>6.1f}% "
        f"${row['allocation']:>9,.0f} "
        f"{int(row['shares']):>7} "
        f"${row['price']:>7.2f}"
    )

print("-" * 65)
print(f"{'TOTAL':<30} {100:>6.0f}%  ${total_deployed:>9,.0f}")
print(f"Cash remaining: ${cash_remaining:,.2f}")

print("\n--- PORTFOLIO RISK SUMMARY ---")
print(f"Number of positions: {TOP_N}")
print(f"Largest single weight: {selected['weight'].max()*100:.1f}% ({selected.loc[selected['weight'].idxmax(), 'ticker']})")
print(f"Average 1Y return of selected ETFs: {selected['return_1y'].mean():+.1f}%")
print(f"Average Sharpe ratio: {selected['sharpe'].mean():.2f}")
print(f"Average volatility: {selected['volatility'].mean():.1f}%")
print(f"Average max drawdown: {selected['max_drawdown'].mean():+.1f}%")

# Category breakdown
print("\n--- CATEGORY BREAKDOWN ---")
cat_weights = selected.groupby("category")["weight"].sum().sort_values(ascending=False)
for cat, w in cat_weights.items():
    print(f"  {cat:<25} {w*100:.1f}%")

# Save final allocation
db = duckdb.connect(DB)
db.execute("CREATE OR REPLACE TABLE allocation AS SELECT * FROM selected")
db.close()
print(f"\nAllocation saved to {DB}")
