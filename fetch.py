import yfinance as yf
import duckdb
import pandas as pd
from universe import TICKERS

DB = "portfolio.duckdb"

print(f"Fetching 1 year of data for {len(TICKERS)} ETFs...\n")

all_data = []
failed = []

for ticker in TICKERS:
    try:
        df = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
        if df.empty:
            failed.append(ticker)
            print(f"  {ticker}: no data")
            continue
        df = df.reset_index()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df["ticker"] = ticker
        all_data.append(df[["Date", "ticker", "Close", "Volume"]])
        print(f"  {ticker}: {len(df)} days")
    except Exception as e:
        failed.append(ticker)
        print(f"  {ticker}: error — {e}")

prices = pd.concat(all_data, ignore_index=True)

db = duckdb.connect(DB)
db.execute("CREATE OR REPLACE TABLE prices AS SELECT * FROM prices")
count = db.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
db.close()

print(f"\n{count} rows saved to {DB}")
if failed:
    print(f"Failed: {failed}")
