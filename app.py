import os
import subprocess
import sys
from datetime import date, timedelta

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from hf_13f import FUNDS, fetch_fund_holdings, get_aggregate_holdings
from universe import ETF_UNIVERSE, TICKERS

# ── Constants ─────────────────────────────────────────────────────────────────

DB              = "portfolio.duckdb"
RISK_FREE       = 0.045
CORR_THRESHOLD  = 0.80
ETF_SLOTS       = 15
HF_SLOTS        = 20
MIXED_ETF       = 8
MIXED_HF        = 12

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Investment Toolkit",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #0f0f1a; }
.metric-label { font-size: 0.8rem; color: #888; text-transform: uppercase; letter-spacing: 0.05em; }
.metric-value { font-size: 1.6rem; font-weight: 700; }
.pos { color: #4ade80; } .neg { color: #f87171; } .neu { color: #94a3b8; }
div[data-testid="stTabs"] button { font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)


# ── Data helpers ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(tickers=None):
    db = duckdb.connect(DB, read_only=True)
    if tickers:
        placeholders = ",".join(f"'{t}'" for t in tickers)
        df = db.execute(f"SELECT * FROM prices WHERE ticker IN ({placeholders}) ORDER BY ticker, Date").fetchdf()
    else:
        df = db.execute("SELECT * FROM prices ORDER BY ticker, Date").fetchdf()
    db.close()
    return df


def compute_metrics(prices_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker, grp in prices_df.groupby("ticker"):
        grp = grp.sort_values("Date").reset_index(drop=True)
        if len(grp) < 50:
            continue
        c = grp["Close"]
        ret_1y = (c.iloc[-1] - c.iloc[0]) / c.iloc[0]
        ret_3m = (c.iloc[-1] - c.iloc[-63]) / c.iloc[-63] if len(grp) >= 63 else np.nan
        ret_6m = (c.iloc[-1] - c.iloc[-126]) / c.iloc[-126] if len(grp) >= 126 else np.nan
        dr     = c.pct_change().dropna()
        vol    = dr.std() * np.sqrt(252)
        sharpe = (ret_1y - RISK_FREE) / vol if vol > 0 else 0
        dd     = ((c - c.cummax()) / c.cummax()).min()
        mom    = (ret_3m - ret_6m) if not (np.isnan(ret_3m) or np.isnan(ret_6m)) else 0
        cat    = next((k for k, v in ETF_UNIVERSE.items() if ticker in v), "Stock")
        rows.append(dict(
            ticker=ticker, category=cat,
            ret_1y=round(ret_1y * 100, 1),
            ret_3m=round(ret_3m * 100, 1) if not np.isnan(ret_3m) else None,
            vol=round(vol * 100, 1),
            sharpe=round(sharpe, 2),
            drawdown=round(dd * 100, 1),
            momentum=round(mom * 100, 2),
            price=round(c.iloc[-1], 2),
        ))
    df = pd.DataFrame(rows)
    df["r_ret"]  = df["ret_1y"].rank(pct=True)
    df["r_shrp"] = df["sharpe"].rank(pct=True)
    df["r_mom"]  = df["momentum"].rank(pct=True)
    df["r_vol"]  = df["vol"].rank(ascending=False, pct=True)
    df["r_dd"]   = df["drawdown"].rank(ascending=False, pct=True)
    df["score"]  = df["r_ret"]*0.25 + df["r_shrp"]*0.35 + df["r_mom"]*0.20 + df["r_vol"]*0.10 + df["r_dd"]*0.10
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def filter_correlated(ranked_tickers: list, prices_df: pd.DataFrame, threshold=CORR_THRESHOLD):
    """Remove tickers that correlate > threshold with a higher-ranked ticker already kept."""
    pivot = (
        prices_df[prices_df["ticker"].isin(ranked_tickers)]
        .pivot(index="Date", columns="ticker", values="Close")
        .pct_change().dropna()
    )
    kept, dropped = [], {}
    for t in ranked_tickers:
        if t not in pivot.columns:
            continue
        conflict = next(
            (k for k in kept if k in pivot.columns and pivot[t].corr(pivot[k]) > threshold),
            None
        )
        if conflict:
            dropped[t] = (conflict, round(pivot[t].corr(pivot[conflict]), 2))
        else:
            kept.append(t)
    return kept, dropped


def build_allocation(tickers: list, prices_df: pd.DataFrame, portfolio_size: float) -> pd.DataFrame:
    rows = []
    for t in tickers:
        grp = prices_df[prices_df["ticker"] == t].sort_values("Date")
        if grp.empty:
            continue
        c   = grp["Close"]
        vol = c.pct_change().dropna().std() * np.sqrt(252)
        rows.append({"ticker": t, "vol": vol, "price": round(c.iloc[-1], 2)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["inv_vol"]       = 1 / df["vol"].replace(0, np.nan)
    df["weight"]        = df["inv_vol"] / df["inv_vol"].sum()
    df["alloc"]         = (df["weight"] * portfolio_size).round(2)
    # fractional shares — Robinhood supports these; show 4 decimal places for small portfolios
    df["shares"]        = (df["alloc"] / df["price"]).round(4)
    df["whole_shares"]  = df["shares"].apply(np.floor).astype(int)
    df["value"]         = (df["shares"] * df["price"]).round(2)
    return df


def portfolio_stats(allocation: pd.DataFrame) -> dict:
    """allocation must already have ret_1y, sharpe, drawdown columns merged in."""
    if allocation.empty:
        return {"avg_return": 0, "avg_sharpe": 0, "avg_drawdown": 0,
                "largest_pos": "—", "largest_wt": 0, "total_value": 0}
    return {
        "avg_return":   round(allocation["ret_1y"].mean(skipna=True), 1),
        "avg_sharpe":   round(allocation["sharpe"].mean(skipna=True), 2),
        "avg_drawdown": round(allocation["drawdown"].mean(skipna=True), 1),
        "largest_pos":  allocation.loc[allocation["weight"].idxmax(), "ticker"],
        "largest_wt":   round(allocation["weight"].max() * 100, 1),
        "total_value":  allocation["value"].sum(),
    }


# ── Paper trading ─────────────────────────────────────────────────────────────

def init_paper_trading(portfolios: dict, prices_df: pd.DataFrame):
    """Record starting positions if no paper trade exists yet."""
    db = duckdb.connect(DB)
    db.execute("""
        CREATE TABLE IF NOT EXISTS paper_start (
            portfolio TEXT, ticker TEXT, shares INTEGER,
            start_price REAL, start_date DATE,
            PRIMARY KEY (portfolio, ticker)
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS paper_snapshots (
            snap_date DATE, portfolio TEXT, value REAL,
            PRIMARY KEY (snap_date, portfolio)
        )
    """)

    existing = db.execute("SELECT DISTINCT portfolio FROM paper_start").fetchdf()
    started  = set(existing["portfolio"].tolist()) if not existing.empty else set()

    today = date.today().isoformat()
    for name, alloc in portfolios.items():
        if name in started or alloc.empty:
            continue
        for _, row in alloc.iterrows():
            grp = prices_df[prices_df["ticker"] == row["ticker"]].sort_values("Date")
            if grp.empty:
                continue
            db.execute(
                "INSERT OR IGNORE INTO paper_start VALUES (?,?,?,?,?)",
                [name, row["ticker"], int(row["shares"]), float(grp["Close"].iloc[-1]), today]
            )
    db.close()


def record_paper_snapshot(portfolios: dict, prices_df: pd.DataFrame):
    db   = duckdb.connect(DB)
    today = date.today().isoformat()
    for name, alloc in portfolios.items():
        if alloc.empty:
            continue
        total = 0.0
        for _, row in alloc.iterrows():
            grp = prices_df[prices_df["ticker"] == row["ticker"]].sort_values("Date")
            if not grp.empty:
                total += int(row["shares"]) * float(grp["Close"].iloc[-1])
        db.execute(
            "INSERT OR REPLACE INTO paper_snapshots VALUES (?,?,?)",
            [today, name, round(total, 2)]
        )
    db.close()


def load_paper_history() -> pd.DataFrame:
    db  = duckdb.connect(DB, read_only=True)
    tables = [r[0] for r in db.execute("SHOW TABLES").fetchall()]
    if "paper_snapshots" not in tables:
        db.close()
        return pd.DataFrame()
    df = db.execute("SELECT * FROM paper_snapshots ORDER BY snap_date").fetchdf()
    db.close()
    return df


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📈 Investment Toolkit")
    st.caption("AI-directed portfolio construction")
    st.divider()

    st.markdown("**Your Portfolio**")
    portfolio_size = st.number_input("Total amount to invest ($)", value=600, step=100, min_value=100, format="%d")
    monthly_contribution = st.number_input("Monthly contribution ($)", value=600, step=50, min_value=0, format="%d")
    if monthly_contribution > 0:
        st.caption(f"At ${monthly_contribution}/mo → ${monthly_contribution*12:,}/yr → ${monthly_contribution*12*5:,} in 5 years (excl. returns)")

    st.divider()
    if st.button("🔄 Refresh market data", use_container_width=True):
        with st.spinner("Fetching latest prices..."):
            subprocess.run([sys.executable, "fetch.py"])
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.markdown("**AI Analyst API Key**")
    api_key_input = st.text_input("Anthropic API key", type="password",
                                  value=os.environ.get("ANTHROPIC_API_KEY", ""),
                                  placeholder="sk-ant-...")
    if api_key_input:
        os.environ["ANTHROPIC_API_KEY"] = api_key_input

    st.divider()
    st.caption("Data: Yahoo Finance · SEC EDGAR")
    st.caption(f"Risk-free rate: {RISK_FREE*100:.1f}% (T-bill)")
    st.caption(f"Correlation filter: >{CORR_THRESHOLD}")


# ── Load ETF data ─────────────────────────────────────────────────────────────

try:
    prices = load_prices()
except Exception:
    st.error("No data found. Click **Refresh market data** in the sidebar.")
    st.stop()

etf_metrics  = compute_metrics(prices)
ranked_etfs  = etf_metrics["ticker"].tolist()
kept_etfs, dropped_etfs = filter_correlated(ranked_etfs, prices)
etf_alloc    = build_allocation(kept_etfs[:ETF_SLOTS], prices, portfolio_size)
etf_alloc    = etf_alloc.merge(
    etf_metrics[["ticker","ret_1y","sharpe","drawdown","momentum","category"]],
    on="ticker", how="left"
)


# ── Load HF data ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def build_hf_portfolio(portfolio_size):
    agg = get_aggregate_holdings()
    if agg.empty:
        return pd.DataFrame(), pd.DataFrame()

    hf_tickers = agg["ticker"].head(40).tolist()

    # Fetch price data for HF stocks
    hf_prices_list = []
    for t in hf_tickers:
        try:
            df = yf.download(t, period="1y", progress=False, auto_adjust=True)
            if df.empty:
                continue
            df = df.reset_index()
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df["ticker"] = t
            hf_prices_list.append(df[["Date", "ticker", "Close"]])
        except Exception:
            continue

    if not hf_prices_list:
        return pd.DataFrame(), pd.DataFrame()

    hf_prices = pd.concat(hf_prices_list, ignore_index=True)
    hf_metrics = compute_metrics(hf_prices)
    ranked_hf = hf_metrics["ticker"].tolist()
    kept_hf, _ = filter_correlated(ranked_hf, hf_prices)
    alloc = build_allocation(kept_hf[:HF_SLOTS], hf_prices, portfolio_size)
    alloc = alloc.merge(
        hf_metrics[["ticker","ret_1y","sharpe","drawdown","momentum"]].assign(category="Stock"),
        on="ticker", how="left"
    )
    alloc = alloc.merge(agg[["ticker","funds","fund_count"]], on="ticker", how="left")
    return alloc, hf_prices


# ── Build mixed portfolio ─────────────────────────────────────────────────────

def build_mixed(etf_alloc, hf_alloc, prices_df, hf_prices_df, portfolio_size):
    if etf_alloc.empty or hf_alloc.empty:
        return pd.DataFrame()
    etf_picks = etf_alloc["ticker"].head(MIXED_ETF).tolist()
    hf_picks  = hf_alloc["ticker"].head(MIXED_HF).tolist()
    combined  = etf_picks + hf_picks
    all_prices = pd.concat([prices_df, hf_prices_df], ignore_index=True)
    all_prices = all_prices.drop_duplicates(subset=["Date","ticker"])
    kept, _ = filter_correlated(combined, all_prices)
    alloc = build_allocation(kept, all_prices, portfolio_size)
    etf_meta = etf_alloc[["ticker","ret_1y","sharpe","drawdown","momentum","category"]]
    hf_meta  = hf_alloc[["ticker","ret_1y","sharpe","drawdown","momentum","category"]]
    meta = pd.concat([etf_meta, hf_meta]).drop_duplicates("ticker")
    return alloc.merge(meta, on="ticker", how="left")


# ── Tabs ──────────────────────────────────────────────────────────────────────

tabs = st.tabs(["Dashboard", "ETF Portfolio", "Hedge Fund Portfolio",
                "Mixed Portfolio", "Paper Trader", "AI Analyst"])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ════════════════════════════════════════════════════════════════════════════════

with tabs[0]:
    st.title("Portfolio Dashboard")

    with st.spinner("Loading hedge fund data…"):
        hf_alloc, hf_prices = build_hf_portfolio(portfolio_size)

    mixed_alloc = build_mixed(etf_alloc, hf_alloc, prices, hf_prices, portfolio_size)

    # Top metrics
    cols = st.columns(3)
    for col, name, alloc in zip(cols,
            ["ETF Strategy", "Hedge Fund Mirror", "Mixed Strategy"],
            [etf_alloc, hf_alloc, mixed_alloc]):
        if alloc.empty:
            col.warning(f"{name}: no data")
            continue
        s = portfolio_stats(alloc)
        col.markdown(f"### {name}")
        col.metric("Avg 1Y Return",  f"{s['avg_return']:+.1f}%")
        col.metric("Avg Sharpe",     f"{s['avg_sharpe']:.2f}")
        col.metric("Avg Drawdown",   f"{s['avg_drawdown']:+.1f}%")
        col.metric("Largest position", f"{s['largest_pos']} ({s['largest_wt']:.1f}%)")

    st.divider()

    # Side-by-side allocation pies
    st.subheader("Allocation Breakdown")
    pie_cols = st.columns(3)
    for col, name, alloc in zip(pie_cols,
            ["ETF Strategy", "Hedge Fund Mirror", "Mixed Strategy"],
            [etf_alloc, hf_alloc, mixed_alloc]):
        if alloc.empty:
            col.info("No data")
            continue
        alloc["wt_pct"] = (alloc["weight"] * 100).round(1)
        fig = px.pie(alloc, values="wt_pct", names="ticker", hole=0.45,
                     color_discrete_sequence=px.colors.qualitative.Pastel, title=name)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(showlegend=False, margin=dict(t=40,b=0,l=0,r=0), height=300)
        col.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — ETF PORTFOLIO
# ════════════════════════════════════════════════════════════════════════════════

with tabs[1]:
    st.title("ETF Portfolio")
    st.caption(f"58-ETF universe → correlation filtered (>{int(CORR_THRESHOLD*100)}%) → top {ETF_SLOTS} → inverse-volatility weighted")

    if dropped_etfs:
        with st.expander(f"🔀 {len(dropped_etfs)} ETFs removed by correlation filter"):
            for t, (conflict, corr) in dropped_etfs.items():
                st.caption(f"{t} removed — corr {corr} with {conflict} (already held)")

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Positions",        len(etf_alloc))
    c2.metric("Avg 1Y Return",    f"{etf_alloc['ret_1y'].mean():+.1f}%")
    c3.metric("Avg Sharpe",       f"{etf_alloc['sharpe'].mean():.2f}")
    c4.metric("Avg Max Drawdown", f"{etf_alloc['drawdown'].mean():+.1f}%")

    st.divider()
    left, right = st.columns([1,1])

    with left:
        etf_alloc["wt_pct"] = (etf_alloc["weight"] * 100).round(1)
        fig_pie = px.pie(etf_alloc, values="wt_pct", names="ticker", hole=0.4,
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(showlegend=False, margin=dict(t=0,b=0,l=0,r=0), height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

    with right:
        cat_df = etf_alloc.groupby("category")["wt_pct"].sum().reset_index()
        fig_bar = px.bar(cat_df.sort_values("wt_pct"), x="wt_pct", y="category",
                         orientation="h", color="wt_pct", color_continuous_scale="Blues",
                         labels={"wt_pct": "Weight %", "category": ""})
        fig_bar.update_layout(coloraxis_showscale=False, margin=dict(t=0,b=0,l=0,r=0), height=350)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Holdings table
    st.subheader("Holdings")
    disp = etf_alloc[["ticker","category","wt_pct","alloc","shares","price","ret_1y","sharpe","drawdown","momentum"]].copy()
    disp.columns = ["Ticker","Category","Weight %","Alloc $","Shares (fractional)","Price","1Y Ret %","Sharpe","Max DD %","Momentum %"]
    st.dataframe(
        disp.style
            .format({"Alloc $": "${:,.2f}", "Price": "${:.2f}",
                     "Shares (fractional)": "{:.4f}",
                     "Weight %": "{:.1f}%", "1Y Ret %": "{:+.1f}%",
                     "Max DD %": "{:+.1f}%", "Momentum %": "{:+.2f}%"})
            .background_gradient(subset=["Sharpe"], cmap="Greens")
            .background_gradient(subset=["1Y Ret %"], cmap="Blues"),
        use_container_width=True, hide_index=True
    )
    if portfolio_size < 1000:
        st.caption("💡 Robinhood supports fractional shares — you can invest exact dollar amounts without needing whole shares.")

    # Full screener
    st.subheader("Full ETF Universe — Screener")
    cat_filter = st.selectbox("Filter category",
                              ["All"] + sorted(etf_metrics["category"].unique()), key="cat_etf")
    view = etf_metrics if cat_filter == "All" else etf_metrics[etf_metrics["category"] == cat_filter]
    view = view[["ticker","category","ret_1y","sharpe","vol","drawdown","momentum","price"]].copy()
    view.columns = ["Ticker","Category","1Y Ret %","Sharpe","Vol %","Max DD %","Momentum %","Price"]
    view.insert(0, "Rank", range(1, len(view)+1))
    st.dataframe(
        view.style
            .format({"1Y Ret %": "{:+.1f}%", "Sharpe": "{:.2f}", "Vol %": "{:.1f}%",
                     "Max DD %": "{:+.1f}%", "Momentum %": "{:+.2f}%", "Price": "${:.2f}"})
            .background_gradient(subset=["Sharpe"], cmap="Greens"),
        use_container_width=True, hide_index=True, height=500
    )

    # Correlation heatmap for selected portfolio
    st.subheader("Correlation Matrix — ETF Portfolio")
    st.caption("Values above 0.80 are blocked by the filter. This shows what remains.")
    pivot  = prices[prices["ticker"].isin(etf_alloc["ticker"])].pivot(
        index="Date", columns="ticker", values="Close").pct_change().dropna()
    corr   = pivot.corr().round(2)
    fig_hm = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale="RdBu_r", zmin=-1, zmax=1,
        text=corr.values.round(2), texttemplate="%{text}", textfont={"size": 9},
    ))
    fig_hm.update_layout(height=500, margin=dict(t=20,b=0,l=0,r=0))
    st.plotly_chart(fig_hm, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — HEDGE FUND PORTFOLIO
# ════════════════════════════════════════════════════════════════════════════════

with tabs[2]:
    st.title("Hedge Fund Mirror Portfolio")
    st.caption("Built from SEC 13F filings — Citadel, Millennium, Two Sigma, D.E. Shaw, Point72")
    st.info("⚠️ 13F filings are delayed 45 days and filed quarterly. These are disclosed positions, not current ones.", icon="ℹ️")

    if hf_alloc.empty:
        st.warning("Could not load 13F data from SEC EDGAR.")
        st.info("SEC EDGAR rate-limits requests. This usually resolves in 1-2 minutes. Click **Refresh market data** in the sidebar or reload the page.")
        if st.button("🔄 Retry hedge fund data"):
            get_aggregate_holdings.clear()
            build_hf_portfolio.clear()
            st.rerun()
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Positions",        len(hf_alloc))
        c2.metric("Avg 1Y Return",    f"{hf_alloc['ret_1y'].mean():+.1f}%")
        c3.metric("Avg Sharpe",       f"{hf_alloc['sharpe'].mean():.2f}")
        c4.metric("Avg Max Drawdown", f"{hf_alloc['drawdown'].mean():+.1f}%")

        left, right = st.columns([1,1])
        with left:
            hf_alloc["wt_pct"] = (hf_alloc["weight"] * 100).round(1)
            fig = px.pie(hf_alloc, values="wt_pct", names="ticker", hole=0.4,
                         color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(showlegend=False, margin=dict(t=0,b=0,l=0,r=0), height=350)
            left.plotly_chart(fig, use_container_width=True)

        with right:
            st.subheader("Individual Fund Holdings")
            fund_sel = st.selectbox("Select fund", list(FUNDS.keys()))
            fund_df, filing_date = fetch_fund_holdings(fund_sel, FUNDS[fund_sel])
            if not fund_df.empty:
                st.caption(f"Latest filing: {filing_date}")
                top25 = fund_df.sort_values("value_thousands", ascending=False).head(25)
                top25["Value ($M)"]    = (top25["value_thousands"] / 1000).round(1)
                top25["% of Filing"]   = (top25["value_thousands"] / top25["value_thousands"].sum() * 100).round(1)
                st.dataframe(
                    top25[["name","ticker","Value ($M)","% of Filing"]].style
                        .format({"Value ($M)": "${:,.0f}M", "% of Filing": "{:.1f}%"})
                        .background_gradient(subset=["% of Filing"], cmap="Blues"),
                    hide_index=True, use_container_width=True, height=320
                )

        st.subheader("Mirror Portfolio Holdings")
        hf_disp = hf_alloc[["ticker","wt_pct","alloc","shares","price","ret_1y","sharpe","drawdown","fund_count","funds"]].copy()
        hf_disp.columns = ["Ticker","Weight %","Alloc $","Shares","Price","1Y Ret %","Sharpe","Max DD %","# Funds","Held By"]
        st.dataframe(
            hf_disp.style
                .format({"Alloc $": "${:,.0f}", "Price": "${:.2f}",
                         "Weight %": "{:.1f}%", "1Y Ret %": "{:+.1f}%", "Max DD %": "{:+.1f}%"})
                .background_gradient(subset=["Sharpe"], cmap="Greens"),
            use_container_width=True, hide_index=True
        )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — MIXED PORTFOLIO
# ════════════════════════════════════════════════════════════════════════════════

with tabs[3]:
    st.title("Mixed Strategy Portfolio")
    st.caption(f"Top {MIXED_ETF} ETFs + top {MIXED_HF} hedge fund picks → correlation filtered → inverse-vol weighted")

    if mixed_alloc.empty:
        st.warning("Mixed portfolio requires hedge fund data. Try refreshing.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Positions",        len(mixed_alloc))
        c2.metric("Avg 1Y Return",    f"{mixed_alloc['ret_1y'].mean():+.1f}%")
        c3.metric("Avg Sharpe",       f"{mixed_alloc['sharpe'].mean():.2f}")
        c4.metric("Avg Max Drawdown", f"{mixed_alloc['drawdown'].mean():+.1f}%")

        mixed_alloc["wt_pct"] = (mixed_alloc["weight"] * 100).round(1)
        left, right = st.columns([1,1])

        with left:
            fig = px.pie(mixed_alloc, values="wt_pct", names="ticker", hole=0.4,
                         color_discrete_sequence=px.colors.qualitative.Vivid)
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(showlegend=False, margin=dict(t=0,b=0,l=0,r=0), height=350)
            st.plotly_chart(fig, use_container_width=True)

        with right:
            cat_df = mixed_alloc.groupby("category")["wt_pct"].sum().reset_index()
            fig_bar = px.bar(cat_df.sort_values("wt_pct"), x="wt_pct", y="category",
                             orientation="h", color="wt_pct", color_continuous_scale="Purples",
                             labels={"wt_pct": "Weight %", "category": ""})
            fig_bar.update_layout(coloraxis_showscale=False, margin=dict(t=0,b=0,l=0,r=0), height=350)
            st.plotly_chart(fig_bar, use_container_width=True)

        disp = mixed_alloc[["ticker","category","wt_pct","alloc","shares","price","ret_1y","sharpe","drawdown"]].copy()
        disp.columns = ["Ticker","Type","Weight %","Alloc $","Shares","Price","1Y Ret %","Sharpe","Max DD %"]
        st.dataframe(
            disp.style
                .format({"Alloc $": "${:,.0f}", "Price": "${:.2f}",
                         "Weight %": "{:.1f}%", "1Y Ret %": "{:+.1f}%", "Max DD %": "{:+.1f}%"})
                .background_gradient(subset=["Sharpe"], cmap="Greens"),
            use_container_width=True, hide_index=True
        )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — PAPER TRADER
# ════════════════════════════════════════════════════════════════════════════════

with tabs[4]:
    st.title("Paper Trader")
    st.caption("Tracks all three portfolios vs S&P 500. No real money. Pure performance simulation.")

    port_map = {"ETF Strategy": etf_alloc, "Hedge Fund Mirror": hf_alloc, "Mixed Strategy": mixed_alloc}
    all_prices_combined = pd.concat([prices, hf_prices], ignore_index=True).drop_duplicates(subset=["Date","ticker"])

    if st.button("▶ Start / Update paper trading snapshot"):
        init_paper_trading(port_map, all_prices_combined)
        record_paper_snapshot(port_map, all_prices_combined)
        st.success("Snapshot recorded.")
        load_paper_history.clear() if hasattr(load_paper_history, "clear") else None

    history = load_paper_history()

    if history.empty:
        st.info("No paper trading history yet. Click **▶ Start / Update** above to begin tracking.")
    else:
        fig = go.Figure()
        colors = {"ETF Strategy": "#60a5fa", "Hedge Fund Mirror": "#34d399", "Mixed Strategy": "#f472b6"}

        for port_name in history["portfolio"].unique():
            sub = history[history["portfolio"] == port_name].copy()
            sub["snap_date"] = pd.to_datetime(sub["snap_date"])
            if sub.empty or sub["value"].iloc[0] == 0:
                continue
            start_val = sub["value"].iloc[0]
            sub["indexed"] = (sub["value"] / start_val) * 100
            fig.add_trace(go.Scatter(
                x=sub["snap_date"], y=sub["indexed"],
                name=port_name, line=dict(width=2, color=colors.get(port_name, "white"))
            ))

        # SPY benchmark — only add if we have at least 2 snapshots
        if len(history["snap_date"].unique()) >= 2:
            try:
                start_date = pd.to_datetime(history["snap_date"].min()).strftime("%Y-%m-%d")
                spy_raw = yf.download("SPY", start=start_date, progress=False, auto_adjust=True)
                if not spy_raw.empty:
                    spy_raw = spy_raw.reset_index()
                    spy_raw.columns = [c[0] if isinstance(c, tuple) else c for c in spy_raw.columns]
                    start_spy = spy_raw["Close"].iloc[0]
                    spy_raw["SPY"] = (spy_raw["Close"] / start_spy) * 100
                    fig.add_trace(go.Scatter(
                        x=pd.to_datetime(spy_raw["Date"]), y=spy_raw["SPY"],
                        name="S&P 500 (SPY)", line=dict(width=2, dash="dash", color="#94a3b8")
                    ))
            except Exception:
                pass

        fig.update_layout(
            title="Portfolio Performance vs S&P 500 (indexed to 100)",
            yaxis_title="Value (indexed to 100)", xaxis_title="Date",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)

        # Stats table
        st.subheader("Performance Summary")
        rows = []
        for port_name in history["portfolio"].unique():
            sub = history[history["portfolio"] == port_name]
            start, end = sub["value"].iloc[0], sub["value"].iloc[-1]
            rows.append({
                "Portfolio": port_name,
                "Start Value": f"${start:,.0f}",
                "Current Value": f"${end:,.0f}",
                "Return": f"{(end-start)/start*100:+.1f}%",
                "Days Tracked": (pd.to_datetime(sub["snap_date"].max()) - pd.to_datetime(sub["snap_date"].min())).days
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        st.subheader("Adaptive Weighting Recommendation")
        st.caption("After 90+ days, allocate more toward the top performer.")
        days = (pd.to_datetime(history["snap_date"].max()) - pd.to_datetime(history["snap_date"].min())).days
        if days < 90:
            st.info(f"Need {90 - days} more days of data before adaptive weighting kicks in.")
        else:
            returns = {}
            for port_name in history["portfolio"].unique():
                sub = history[history["portfolio"] == port_name]
                returns[port_name] = (sub["value"].iloc[-1] - sub["value"].iloc[0]) / sub["value"].iloc[0]
            best = max(returns, key=returns.get)
            st.success(f"**{best}** has the best return ({returns[best]*100:+.1f}%). Recommend increasing allocation to this strategy.")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 6 — AI ANALYST
# ════════════════════════════════════════════════════════════════════════════════

def build_portfolio_context():
    lines = ["You are a professional portfolio analyst. Here is the current portfolio data:\n"]
    lines.append("## ETF PORTFOLIO")
    if not etf_alloc.empty:
        for _, r in etf_alloc.iterrows():
            lines.append(f"- {r['ticker']} ({r.get('category','')}) — "
                         f"weight {r['wt_pct']:.1f}%, 1Y return {r['ret_1y']:+.1f}%, "
                         f"Sharpe {r['sharpe']:.2f}, max drawdown {r['drawdown']:+.1f}%")
    if not hf_alloc.empty:
        lines.append("\n## HEDGE FUND MIRROR PORTFOLIO")
        for _, r in hf_alloc.iterrows():
            lines.append(f"- {r['ticker']} — weight {r['wt_pct']:.1f}%, "
                         f"1Y return {r['ret_1y']:+.1f}%, Sharpe {r['sharpe']:.2f}, "
                         f"held by {r.get('fund_count','?')} fund(s)")
    lines.append(f"\n## METHODOLOGY")
    lines.append(f"- Weighting: inverse-volatility (lower vol = higher weight)")
    lines.append(f"- Correlation filter: pairs above {CORR_THRESHOLD} are removed")
    lines.append(f"- Risk-free rate: {RISK_FREE*100:.1f}% (T-bill)")
    lines.append(f"- 13F filings are delayed 45 days — disclosed positions, not current")
    lines.append("\nAnswer clearly and concisely. Use professional language, explain jargon.")
    return "\n".join(lines)


def generate_quick_brief() -> str:
    """Template-based portfolio brief — no API key needed."""
    from datetime import date as dt
    lines = []
    lines.append(f"# Portfolio Morning Brief — {dt.today().strftime('%B %d, %Y')}\n")

    if not etf_alloc.empty:
        avg_ret    = etf_alloc["ret_1y"].mean()
        avg_sharpe = etf_alloc["sharpe"].mean()
        avg_dd     = etf_alloc["drawdown"].mean()
        best       = etf_alloc.loc[etf_alloc["ret_1y"].idxmax()]
        worst      = etf_alloc.loc[etf_alloc["ret_1y"].idxmin()]
        top3       = etf_alloc.nlargest(3, "weight")["ticker"].tolist()
        cat_wts    = etf_alloc.groupby("category")["wt_pct"].sum().sort_values(ascending=False)

        lines.append("## ETF Portfolio")
        lines.append(f"**Avg 1Y Return:** {avg_ret:+.1f}%  |  "
                     f"**Avg Sharpe:** {avg_sharpe:.2f}  |  "
                     f"**Avg Max Drawdown:** {avg_dd:+.1f}%\n")
        lines.append(f"**Largest positions:** {', '.join(top3)}")
        lines.append(f"**Best performer:** {best['ticker']} ({best['ret_1y']:+.1f}%, Sharpe {best['sharpe']:.2f})")
        lines.append(f"**Worst performer:** {worst['ticker']} ({worst['ret_1y']:+.1f}%)\n")
        lines.append("**Category breakdown:**")
        for cat, wt in cat_wts.items():
            lines.append(f"- {cat}: {wt:.1f}%")

        sharpe_note = ("above the 1.0 benchmark — solid risk-adjusted return"
                       if avg_sharpe >= 1.0 else
                       "below the 1.0 benchmark — consider higher-momentum names or reducing low-return positions")
        lines.append(f"\n**Sharpe assessment:** {avg_sharpe:.2f} is {sharpe_note}.")

        if avg_dd < -15:
            lines.append(f"**Drawdown note:** Average max drawdown of {avg_dd:.1f}% is elevated. "
                         "Review positions with the largest drops for thesis changes.")

    if not hf_alloc.empty:
        lines.append("\n## Hedge Fund Mirror Portfolio")
        top_hf = hf_alloc.nlargest(5, "weight")
        lines.append(f"**Top 5 positions:** {', '.join(top_hf['ticker'].tolist())}")
        lines.append(f"**Avg 1Y Return:** {hf_alloc['ret_1y'].mean():+.1f}%  |  "
                     f"**Avg Sharpe:** {hf_alloc['sharpe'].mean():.2f}")
        multi_fund = hf_alloc[hf_alloc.get("fund_count", pd.Series([0]*len(hf_alloc))) >= 2]
        if not multi_fund.empty:
            lines.append(f"**High conviction (2+ funds):** {', '.join(multi_fund['ticker'].tolist())}")

    lines.append("\n## Key Reminders")
    lines.append(f"- Correlation filter blocked pairs above {int(CORR_THRESHOLD*100)}% — remaining holdings are genuinely diversified")
    lines.append("- 13F filings lag reality by 45 days — hedge fund tab shows disclosed, not current, positions")
    lines.append(f"- Risk-free rate: {RISK_FREE*100:.1f}% — any Sharpe below 1.0 means you're not being fully compensated for risk")

    return "\n".join(lines)


with tabs[5]:
    st.title("AI Portfolio Analyst")

    analyst_mode = st.radio(
        "Mode",
        ["Quick Brief (free, no API key)", "Claude AI Chat (requires API key)"],
        horizontal=True
    )

    st.divider()

    if analyst_mode == "Quick Brief (free, no API key)":
        st.caption("Generates a structured portfolio brief from your live data. No API key needed.")
        if st.button("Generate Morning Brief", type="primary"):
            with st.spinner("Analyzing your portfolio..."):
                brief = generate_quick_brief()
            st.markdown(brief)
            st.divider()
            st.caption("For conversational Q&A and deeper analysis, switch to Claude AI Chat mode and add an API key.")

    else:
        st.caption("Conversational analyst powered by Claude. Enter your Anthropic API key in the sidebar.")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            st.warning("Enter your Anthropic API key in the sidebar to enable AI chat.")
            st.info("Get one at console.anthropic.com → API Keys. $5 in credits lasts months of personal use.")
        else:
            try:
                import anthropic
            except ImportError:
                st.error("Run: pip3 install anthropic")
                st.stop()

            system_prompt = build_portfolio_context()

            if "messages" not in st.session_state:
                st.session_state.messages = []

            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            if not st.session_state.messages:
                st.markdown("**Try asking:**")
                suggestions = [
                    "Give me a morning brief on my ETF portfolio",
                    "Which portfolio has the best risk-adjusted return and why?",
                    "Are my portfolios actually diversified or just correlated bets?",
                    "What macro factors should I be watching given these holdings?",
                    "Explain inverse-volatility weighting in plain English",
                ]
                cols = st.columns(len(suggestions))
                for col, s in zip(cols, suggestions):
                    if col.button(s, key=s):
                        st.session_state.messages.append({"role": "user", "content": s})
                        st.rerun()

            if prompt := st.chat_input("Ask about your portfolio…"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                    with st.spinner("Analyzing…"):
                        response = client.messages.create(
                            model="claude-sonnet-4-6",
                            max_tokens=1024,
                            system=system_prompt,
                            messages=[{"role": m["role"], "content": m["content"]}
                                      for m in st.session_state.messages],
                        )
                    reply = response.content[0].text
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
