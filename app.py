import os, subprocess, sys
from datetime import date

import duckdb, numpy as np, pandas as pd
import plotly.express as px, plotly.graph_objects as go
import streamlit as st, yfinance as yf

from universe import ETF_UNIVERSE, RISK_PROFILES, TICKERS

# ── Constants ─────────────────────────────────────────────────────────────────

DB             = "portfolio.duckdb"
RISK_FREE      = 0.045
CORR_THRESHOLD = 0.80
SLOTS          = {"Conservative": 6, "Moderate": 8, "Aggressive": 7}

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Investment Toolkit", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #0f0f1a; }
.big-number { font-size: 2.2rem; font-weight: 700; }
.label { font-size: 0.78rem; color: #888; text-transform: uppercase; letter-spacing:.05em; }
</style>""", unsafe_allow_html=True)


# ── DB helpers ────────────────────────────────────────────────────────────────

def get_db(read_only=False):
    return duckdb.connect(DB, read_only=read_only)


def init_db():
    db = get_db()
    db.execute("""CREATE TABLE IF NOT EXISTS user_profile (
        key TEXT PRIMARY KEY, value TEXT)""")
    db.execute("""CREATE TABLE IF NOT EXISTS paper_holdings (
        ticker TEXT PRIMARY KEY, shares REAL, avg_cost REAL, added_date DATE)""")
    db.execute("""CREATE TABLE IF NOT EXISTS paper_snapshots (
        snap_date DATE, portfolio_value REAL, spy_value REAL,
        PRIMARY KEY (snap_date))""")
    db.execute("""CREATE TABLE IF NOT EXISTS monthly_log (
        log_date DATE, ticker TEXT, dollars REAL, shares REAL,
        PRIMARY KEY (log_date, ticker))""")
    db.close()

init_db()


def get_profile():
    db = get_db(read_only=True)
    rows = db.execute("SELECT key, value FROM user_profile").fetchall()
    db.close()
    return {r[0]: r[1] for r in rows}


def save_profile(risk: str, monthly: float):
    db = get_db()
    db.execute("INSERT OR REPLACE INTO user_profile VALUES ('risk', ?)", [risk])
    db.execute("INSERT OR REPLACE INTO user_profile VALUES ('monthly', ?)", [str(monthly)])
    db.close()


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_prices():
    db = get_db(read_only=True)
    df = db.execute("SELECT * FROM prices ORDER BY ticker, Date").fetchdf()
    db.close()
    return df


def latest_price(prices_df, ticker):
    sub = prices_df[prices_df["ticker"] == ticker]
    return float(sub["Close"].iloc[-1]) if not sub.empty else None


# ── Core finance functions ────────────────────────────────────────────────────

def compute_metrics(prices_df: pd.DataFrame, tickers: list) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        grp = prices_df[prices_df["ticker"] == ticker].sort_values("Date").reset_index(drop=True)
        if len(grp) < 50:
            continue
        c      = grp["Close"]
        ret_1y = (c.iloc[-1] - c.iloc[0]) / c.iloc[0]
        ret_3m = (c.iloc[-1] - c.iloc[-63]) / c.iloc[-63] if len(grp) >= 63 else np.nan
        ret_6m = (c.iloc[-1] - c.iloc[-126]) / c.iloc[-126] if len(grp) >= 126 else np.nan
        dr     = c.pct_change().dropna()
        vol    = dr.std() * np.sqrt(252)
        sharpe = (ret_1y - RISK_FREE) / vol if vol > 0 else 0
        dd     = ((c - c.cummax()) / c.cummax()).min()
        mom    = (ret_3m - ret_6m) if not (np.isnan(ret_3m) or np.isnan(ret_6m)) else 0
        cat    = next((k for k, v in ETF_UNIVERSE.items() if ticker in v), "Other")
        rows.append(dict(ticker=ticker, category=cat,
                         ret_1y=round(ret_1y*100,1), vol=round(vol*100,1),
                         sharpe=round(sharpe,2), drawdown=round(dd*100,1),
                         momentum=round(mom*100,2), price=round(c.iloc[-1],2)))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["r_ret"]  = df["ret_1y"].rank(pct=True)
    df["r_shrp"] = df["sharpe"].rank(pct=True)
    df["r_mom"]  = df["momentum"].rank(pct=True)
    df["r_vol"]  = df["vol"].rank(ascending=False, pct=True)
    df["r_dd"]   = df["drawdown"].rank(ascending=False, pct=True)
    df["score"]  = (df["r_ret"]*0.15 + df["r_shrp"]*0.50 +
                    df["r_mom"]*0.20 + df["r_vol"]*0.10 + df["r_dd"]*0.05)
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def filter_correlated(ranked_tickers, prices_df, threshold=CORR_THRESHOLD):
    avail = [t for t in ranked_tickers if t in prices_df["ticker"].values]
    if len(avail) < 2:
        return avail, {}
    pivot = (prices_df[prices_df["ticker"].isin(avail)]
             .pivot(index="Date", columns="ticker", values="Close")
             .pct_change().dropna())
    kept, dropped = [], {}
    for t in avail:
        if t not in pivot.columns:
            continue
        conflict = next((k for k in kept if k in pivot.columns
                         and pivot[t].corr(pivot[k]) > threshold), None)
        if conflict:
            dropped[t] = (conflict, round(pivot[t].corr(pivot[conflict]), 2))
        else:
            kept.append(t)
    return kept, dropped


def build_target_allocation(tickers, prices_df, n_slots):
    rows = []
    for t in tickers[:n_slots*3]:  # screen wider, filter down
        grp = prices_df[prices_df["ticker"] == t].sort_values("Date")
        if grp.empty:
            continue
        c   = grp["Close"]
        vol = c.pct_change().dropna().std() * np.sqrt(252)
        rows.append({"ticker": t, "vol": vol, "price": round(c.iloc[-1], 2)})
        if len(rows) >= n_slots:
            break
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["inv_vol"] = 1 / df["vol"].replace(0, np.nan)
    df["weight"]  = df["inv_vol"] / df["inv_vol"].sum()
    return df[["ticker","weight","price","vol"]]


def monthly_buy_plan(target: pd.DataFrame, holdings: pd.DataFrame,
                     monthly_budget: float, prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given target weights, current holdings, and this month's budget,
    return how much to buy of each ETF to drift back toward target.
    """
    latest = prices_df.sort_values("Date").groupby("ticker")["Close"].last().to_dict()

    plan = target.copy()
    plan["current_price"] = plan["ticker"].map(lambda t: latest.get(t, plan.loc[plan["ticker"]==t,"price"].values[0]))

    # Merge in current holdings
    if not holdings.empty:
        plan = plan.merge(holdings[["ticker","shares"]], on="ticker", how="left")
    else:
        plan["shares"] = 0.0

    plan["shares"]        = plan["shares"].fillna(0)
    plan["current_value"] = plan["shares"] * plan["current_price"]
    total_current         = plan["current_value"].sum()
    total_projected       = total_current + monthly_budget

    plan["target_value"]  = plan["weight"] * total_projected
    plan["underweight"]   = (plan["target_value"] - plan["current_value"]).clip(lower=0)

    total_uw = plan["underweight"].sum()
    if total_uw > 0:
        plan["buy_dollars"] = (plan["underweight"] / total_uw * monthly_budget).round(2)
    else:
        plan["buy_dollars"] = (plan["weight"] * monthly_budget).round(2)

    plan["buy_shares"]    = (plan["buy_dollars"] / plan["current_price"]).round(4)
    plan["current_pct"]   = (plan["current_value"] / total_current * 100).round(1) if total_current > 0 else 0
    plan["target_pct"]    = (plan["weight"] * 100).round(1)
    plan["drift_pct"]     = (plan["current_pct"] - plan["target_pct"]).round(1)

    return plan


# ── Load data ─────────────────────────────────────────────────────────────────

try:
    prices = load_prices()
except Exception:
    st.error("No market data found. Click **Refresh Data** in the sidebar.")
    st.stop()

profile = get_profile()
risk    = profile.get("risk", None)
monthly = float(profile.get("monthly", 600))


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📈 Investment Toolkit")
    st.divider()

    if risk:
        badge = {"Conservative": "🟢", "Moderate": "🟡", "Aggressive": "🔴"}
        st.markdown(f"**Profile:** {badge.get(risk,'')} {risk}")
        st.markdown(f"**Monthly:** ${monthly:,.0f}")
        if st.button("Change profile", use_container_width=True):
            st.session_state["setup"] = True
    else:
        st.session_state["setup"] = True

    st.divider()

    if st.button("🔄 Refresh market data", use_container_width=True):
        with st.spinner("Fetching latest prices…"):
            subprocess.run([sys.executable, "fetch.py"])
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.markdown("**AI Analyst (optional)**")
    api_key = st.text_input("Anthropic API key", type="password",
                            value=os.environ.get("ANTHROPIC_API_KEY",""),
                            placeholder="sk-ant-…")
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

    st.divider()
    st.markdown("**Robinhood (optional)**")
    rh_user = st.text_input("Username/email", key="rh_user")
    rh_pass = st.text_input("Password", type="password", key="rh_pass")
    if st.button("Connect Robinhood", use_container_width=True):
        try:
            import robin_stocks.robinhood as rh
            rh.login(rh_user, rh_pass)
            st.session_state["rh_connected"] = True
            st.success("Connected!")
        except Exception as e:
            st.error(f"Login failed: {e}")

    st.divider()
    st.caption("Data: Yahoo Finance · SEC EDGAR")


# ── Setup wizard ──────────────────────────────────────────────────────────────

if st.session_state.get("setup") or not risk:
    st.title("Welcome — Let's set up your profile")
    st.markdown("Answer two questions. You can change these anytime.")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### What's your risk tolerance?")
        st.caption("This determines which ETFs we pick for you.")
        new_risk = st.radio("", [
            "Conservative — protect my money, slow growth",
            "Moderate — balanced growth and stability",
            "Aggressive — maximum growth, I can handle big swings",
        ], label_visibility="collapsed")
        new_risk = new_risk.split(" — ")[0]

    with col2:
        st.markdown("#### How much will you invest monthly?")
        st.caption("The app tells you exactly how to split this each month.")
        new_monthly = st.number_input("Monthly contribution ($)",
                                      value=600, min_value=50, step=50, format="%d")

    st.divider()
    if st.button("Save and continue →", type="primary"):
        save_profile(new_risk, new_monthly)
        st.session_state["setup"] = False
        st.cache_data.clear()
        st.rerun()
    st.stop()


# ── Build portfolio for this risk profile ─────────────────────────────────────

eligible   = RISK_PROFILES[risk]
metrics    = compute_metrics(prices, eligible)
n_slots    = SLOTS[risk]
ranked     = metrics["ticker"].tolist()
kept, dropped_map = filter_correlated(ranked, prices)
target     = build_target_allocation(kept, prices, n_slots)
target     = target.merge(metrics[["ticker","ret_1y","sharpe","drawdown","momentum","category"]],
                          on="ticker", how="left")
target["weight_pct"] = (target["weight"] * 100).round(1)

# Load paper holdings
db = get_db(read_only=True)
tables = [r[0] for r in db.execute("SHOW TABLES").fetchall()]
holdings_df = db.execute("SELECT * FROM paper_holdings").fetchdf() if "paper_holdings" in tables else pd.DataFrame()
db.close()


# ── Robinhood holdings override ───────────────────────────────────────────────

if st.session_state.get("rh_connected"):
    try:
        import robin_stocks.robinhood as rh
        positions = rh.get_open_stock_positions()
        rh_rows = []
        for p in positions:
            ticker = rh.get_symbol_by_url(p["instrument"])
            rh_rows.append({"ticker": ticker,
                            "shares": float(p["quantity"]),
                            "avg_cost": float(p["average_buy_price"])})
        if rh_rows:
            holdings_df = pd.DataFrame(rh_rows)
    except Exception:
        pass


# ── Plan ──────────────────────────────────────────────────────────────────────

plan = monthly_buy_plan(target, holdings_df, monthly, prices)


# ── Tabs ──────────────────────────────────────────────────────────────────────

tabs = st.tabs(["📅 This Month", "📊 My Portfolio", "🔍 ETF Screener",
                "📈 Paper Trader", "🤖 AI Analyst"])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — THIS MONTH
# ════════════════════════════════════════════════════════════════════════════════

with tabs[0]:
    rh_note = " · Live from Robinhood" if st.session_state.get("rh_connected") else " · Paper portfolio"
    st.title(f"This Month's Investment Plan")
    st.caption(f"{risk} profile · ${monthly:,.0f}/month{rh_note}")

    total_current = plan["current_value"].sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Portfolio value today", f"${total_current:,.2f}")
    c2.metric("Adding this month",     f"${monthly:,.0f}")
    c3.metric("After contribution",    f"${total_current + monthly:,.2f}")

    st.divider()
    st.subheader(f"Buy this month with ${monthly:,.0f}:")

    for _, row in plan.iterrows():
        if row["buy_dollars"] < 0.01:
            continue
        drift_str = f"underweight {abs(row['drift_pct']):.1f}%" if row["drift_pct"] < -0.5 else \
                    f"overweight {row['drift_pct']:.1f}%" if row["drift_pct"] > 0.5 else "on target"
        col1, col2, col3, col4 = st.columns([2, 2, 2, 3])
        col1.markdown(f"### {row['ticker']}")
        col2.markdown(f"**${row['buy_dollars']:,.2f}**")
        col3.markdown(f"**{row['buy_shares']:.4f} shares**")
        col4.caption(f"Target {row['target_pct']:.1f}% · Currently {row['current_pct']:.1f}% · {drift_str}")

    st.divider()

    col_mark, col_rh = st.columns(2)
    with col_mark:
        if st.button("✅ Mark as invested (paper)", use_container_width=True, type="primary"):
            db = get_db()
            today = date.today().isoformat()
            for _, row in plan.iterrows():
                if row["buy_dollars"] < 0.01:
                    continue
                existing = db.execute("SELECT shares, avg_cost FROM paper_holdings WHERE ticker=?",
                                      [row["ticker"]]).fetchone()
                if existing:
                    old_shares, old_cost = existing
                    new_shares = old_shares + row["buy_shares"]
                    new_cost   = (old_shares*old_cost + row["buy_shares"]*row["current_price"]) / new_shares
                    db.execute("UPDATE paper_holdings SET shares=?, avg_cost=?, added_date=? WHERE ticker=?",
                               [new_shares, new_cost, today, row["ticker"]])
                else:
                    db.execute("INSERT INTO paper_holdings VALUES (?,?,?,?)",
                               [row["ticker"], row["buy_shares"], row["current_price"], today])
                db.execute("INSERT OR REPLACE INTO monthly_log VALUES (?,?,?,?)",
                           [today, row["ticker"], row["buy_dollars"], row["buy_shares"]])
            db.close()
            st.success("Recorded! Your paper portfolio has been updated.")
            st.rerun()

    with col_rh:
        if st.session_state.get("rh_connected"):
            if st.button("🤖 Execute on Robinhood", use_container_width=True):
                try:
                    import robin_stocks.robinhood as rh
                    for _, row in plan.iterrows():
                        if row["buy_dollars"] < 1:
                            continue
                        rh.order_buy_fractional_by_price(row["ticker"],
                                                         row["buy_dollars"],
                                                         timeInForce="gfd")
                    st.success("Orders placed on Robinhood!")
                except Exception as e:
                    st.error(f"Order failed: {e}")
        else:
            st.button("🤖 Execute on Robinhood (connect first)", disabled=True, use_container_width=True)

    # Allocation breakdown
    st.divider()
    st.subheader("Target allocation")
    col_pie, col_tbl = st.columns([1, 1])

    with col_pie:
        fig = px.pie(target, values="weight_pct", names="ticker", hole=0.45,
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(showlegend=False, margin=dict(t=0,b=0,l=0,r=0), height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col_tbl:
        tbl = target[["ticker","category","weight_pct","ret_1y","sharpe","drawdown"]].copy()
        tbl.columns = ["Ticker","Category","Target %","1Y Ret %","Sharpe","Max DD %"]
        st.dataframe(
            tbl.style
               .format({"Target %":"{:.1f}%","1Y Ret %":"{:+.1f}%","Max DD %":"{:+.1f}%"})
               .background_gradient(subset=["Sharpe"], cmap="Greens"),
            hide_index=True, use_container_width=True
        )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — MY PORTFOLIO
# ════════════════════════════════════════════════════════════════════════════════

with tabs[1]:
    st.title("My Portfolio")

    if holdings_df.empty:
        st.info("No holdings yet. Go to **This Month** and click **Mark as invested** after your first contribution.")
    else:
        # Enrich holdings with current prices and metrics
        hld = holdings_df.copy()
        hld["current_price"] = hld["ticker"].map(lambda t: latest_price(prices, t))
        hld["current_value"] = hld["shares"] * hld["current_price"]
        hld["cost_basis"]    = hld["shares"] * hld["avg_cost"]
        hld["gain_loss"]     = hld["current_value"] - hld["cost_basis"]
        hld["gain_pct"]      = (hld["gain_loss"] / hld["cost_basis"] * 100).round(2)
        hld = hld.merge(target[["ticker","weight_pct","category"]], on="ticker", how="left")

        total_val   = hld["current_value"].sum()
        total_cost  = hld["cost_basis"].sum()
        total_gain  = total_val - total_cost
        total_gain_pct = total_gain / total_cost * 100 if total_cost > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Portfolio Value", f"${total_val:,.2f}")
        c2.metric("Total Cost Basis", f"${total_cost:,.2f}")
        g_delta = f"{total_gain_pct:+.1f}%"
        c3.metric("Total Gain / Loss", f"${total_gain:+,.2f}", delta=g_delta)
        c4.metric("Positions", len(hld))

        st.divider()

        col_l, col_r = st.columns([1, 1])
        with col_l:
            hld["pct_of_port"] = (hld["current_value"] / total_val * 100).round(1)
            fig = px.pie(hld, values="pct_of_port", names="ticker", hole=0.45,
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(showlegend=False, margin=dict(t=0,b=0,l=0,r=0), height=300)
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            disp = hld[["ticker","shares","avg_cost","current_price","current_value","gain_loss","gain_pct"]].copy()
            disp.columns = ["Ticker","Shares","Avg Cost","Price","Value","Gain $","Gain %"]
            st.dataframe(
                disp.style
                    .format({"Shares":"{:.4f}","Avg Cost":"${:.2f}","Price":"${:.2f}",
                             "Value":"${:,.2f}","Gain $":"${:+,.2f}","Gain %":"{:+.2f}%"})
                    .applymap(lambda v: "color:#4ade80" if isinstance(v,str) and v.startswith("$+")
                              else ("color:#f87171" if isinstance(v,str) and v.startswith("$-") else ""),
                              subset=["Gain $"]),
                hide_index=True, use_container_width=True
            )

        # Drift vs target
        st.subheader("Drift from target allocation")
        drift_df = plan[plan["current_value"] > 0][["ticker","target_pct","current_pct","drift_pct"]].copy()
        drift_df.columns = ["Ticker","Target %","Current %","Drift %"]
        fig_drift = px.bar(drift_df, x="Ticker", y="Drift %",
                           color="Drift %", color_continuous_scale="RdYlGn",
                           color_continuous_midpoint=0)
        fig_drift.add_hline(y=0, line_dash="dash", line_color="white")
        fig_drift.update_layout(height=300, coloraxis_showscale=False)
        st.plotly_chart(fig_drift, use_container_width=True)
        st.caption("Negative = underweight (buy more). Positive = overweight (let it drift or trim).")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — ETF SCREENER
# ════════════════════════════════════════════════════════════════════════════════

with tabs[2]:
    st.title("ETF Screener")
    st.caption(f"Showing ETFs eligible for your **{risk}** profile — scored 50% Sharpe, 20% momentum, 15% return")

    if dropped_map:
        with st.expander(f"{len(dropped_map)} ETFs removed by correlation filter (>{int(CORR_THRESHOLD*100)}%)"):
            for t, (conflict, corr) in dropped_map.items():
                st.caption(f"{t} removed — {corr:.2f} correlation with {conflict}")

    view = metrics[["ticker","category","ret_1y","sharpe","vol","drawdown","momentum","price"]].copy()
    view.columns = ["Ticker","Category","1Y Ret %","Sharpe","Vol %","Max DD %","Momentum %","Price"]
    view.insert(0,"Rank", range(1, len(view)+1))
    view["In portfolio"] = view["Ticker"].isin(target["ticker"].tolist()).map({True:"✅", False:""})

    st.dataframe(
        view.style
            .format({"1Y Ret %":"{:+.1f}%","Sharpe":"{:.2f}",
                     "Vol %":"{:.1f}%","Max DD %":"{:+.1f}%",
                     "Momentum %":"{:+.2f}%","Price":"${:.2f}"})
            .background_gradient(subset=["Sharpe"], cmap="Greens"),
        use_container_width=True, hide_index=True, height=550
    )

    # Risk vs return chart
    st.subheader("Risk vs Return")
    scatter_df = metrics.copy()
    scatter_df["bubble_size"] = scatter_df["sharpe"].clip(lower=0.1)
    fig = px.scatter(scatter_df, x="vol", y="ret_1y", color="category",
                     hover_name="ticker", size="bubble_size",
                     labels={"vol":"Volatility %","ret_1y":"1Y Return %"},
                     color_discrete_sequence=px.colors.qualitative.Pastel)
    in_port = target["ticker"].tolist()
    port_pts = metrics[metrics["ticker"].isin(in_port)]
    fig.add_scatter(x=port_pts["vol"], y=port_pts["ret_1y"],
                    mode="markers+text", text=port_pts["ticker"],
                    textposition="top center",
                    marker=dict(size=14, color="white", line=dict(color="black",width=2)),
                    name="In your portfolio")
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap for selected portfolio
    st.subheader("Correlation Matrix — Your Portfolio")
    held_tickers = target["ticker"].tolist()
    pivot = (prices[prices["ticker"].isin(held_tickers)]
             .pivot(index="Date", columns="ticker", values="Close")
             .pct_change().dropna())
    corr = pivot.corr().round(2)
    fig_hm = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale="RdBu_r", zmin=-1, zmax=1,
        text=corr.values.round(2), texttemplate="%{text}", textfont={"size":10}))
    fig_hm.update_layout(height=450, margin=dict(t=20,b=0,l=0,r=0))
    st.plotly_chart(fig_hm, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — PAPER TRADER
# ════════════════════════════════════════════════════════════════════════════════

with tabs[3]:
    st.title("Paper Trader")
    st.caption("Tracks your portfolio value vs S&P 500 over time.")

    if st.button("📸 Snapshot today's value", type="primary"):
        if holdings_df.empty:
            st.warning("No holdings yet to snapshot.")
        else:
            hld_snap = holdings_df.copy()
            hld_snap["price"]  = hld_snap["ticker"].map(lambda t: latest_price(prices, t))
            hld_snap["value"]  = hld_snap["shares"] * hld_snap["price"]
            total = hld_snap["value"].sum()

            spy_price = latest_price(prices, "SPY")
            if spy_price is None:
                try:
                    spy_df = yf.download("SPY", period="1d", progress=False, auto_adjust=True)
                    spy_price = float(spy_df["Close"].iloc[-1])
                except Exception:
                    spy_price = 0

            db = get_db()
            db.execute("INSERT OR REPLACE INTO paper_snapshots VALUES (?,?,?)",
                       [date.today().isoformat(), round(total,2), round(spy_price,2)])
            db.close()
            st.success(f"Snapshot saved: portfolio ${total:,.2f}")

    db = get_db(read_only=True)
    snap_df = db.execute("SELECT * FROM paper_snapshots ORDER BY snap_date").fetchdf() \
              if "paper_snapshots" in [r[0] for r in db.execute("SHOW TABLES").fetchall()] \
              else pd.DataFrame()
    db.close()

    if snap_df.empty:
        st.info("No snapshots yet. Click above after you've invested your first month.")
    else:
        snap_df["snap_date"] = pd.to_datetime(snap_df["snap_date"])
        start_port = snap_df["portfolio_value"].iloc[0]
        start_spy  = snap_df["spy_value"].iloc[0]

        snap_df["port_indexed"] = (snap_df["portfolio_value"] / start_port * 100).round(2)
        snap_df["spy_indexed"]  = (snap_df["spy_value"] / start_spy * 100).round(2) if start_spy > 0 else 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=snap_df["snap_date"], y=snap_df["port_indexed"],
                                 name=f"Your portfolio ({risk})",
                                 line=dict(width=2, color="#60a5fa")))
        fig.add_trace(go.Scatter(x=snap_df["snap_date"], y=snap_df["spy_indexed"],
                                 name="S&P 500 (SPY)",
                                 line=dict(width=2, dash="dash", color="#94a3b8")))
        fig.update_layout(title="Performance vs S&P 500 (indexed to 100)",
                          yaxis_title="Value (indexed to 100)",
                          legend=dict(orientation="h", y=1.05), height=400)
        st.plotly_chart(fig, use_container_width=True)

        days = (snap_df["snap_date"].max() - snap_df["snap_date"].min()).days
        port_ret = (snap_df["portfolio_value"].iloc[-1] / start_port - 1) * 100
        spy_ret  = (snap_df["spy_value"].iloc[-1] / start_spy - 1) * 100 if start_spy > 0 else 0
        c1, c2, c3 = st.columns(3)
        c1.metric("Days tracked", days)
        c2.metric("Your return", f"{port_ret:+.1f}%")
        c3.metric("S&P 500 return", f"{spy_ret:+.1f}%",
                  delta=f"{port_ret-spy_ret:+.1f}% vs market")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — AI ANALYST
# ════════════════════════════════════════════════════════════════════════════════

def generate_quick_brief():
    lines = [f"# Portfolio Morning Brief — {date.today().strftime('%B %d, %Y')}\n"]
    lines.append(f"**Risk profile:** {risk}  |  **Monthly contribution:** ${monthly:,.0f}\n")

    avg_sharpe = target["sharpe"].mean()
    avg_ret    = target["ret_1y"].mean()
    avg_dd     = target["drawdown"].mean()
    best  = target.loc[target["ret_1y"].idxmax()]
    worst = target.loc[target["ret_1y"].idxmin()]

    lines.append("## Portfolio Metrics")
    lines.append(f"- **Avg Sharpe:** {avg_sharpe:.2f} — "
                 f"{'above' if avg_sharpe>=1 else 'below'} the 1.0 benchmark")
    lines.append(f"- **Avg 1Y Return:** {avg_ret:+.1f}%")
    lines.append(f"- **Avg Max Drawdown:** {avg_dd:+.1f}%")
    lines.append(f"- **Best performer:** {best['ticker']} ({best['ret_1y']:+.1f}%, Sharpe {best['sharpe']:.2f})")
    lines.append(f"- **Weakest performer:** {worst['ticker']} ({worst['ret_1y']:+.1f}%)\n")

    lines.append("## Holdings")
    for _, r in target.iterrows():
        lines.append(f"- **{r['ticker']}** ({r['category']}) — "
                     f"{r['weight_pct']:.1f}% weight, {r['ret_1y']:+.1f}% 1Y, Sharpe {r['sharpe']:.2f}")

    lines.append("\n## This Month")
    top_buy = plan.nlargest(3, "buy_dollars")
    lines.append(f"Largest buys: " +
                 ", ".join(f"{r['ticker']} (${r['buy_dollars']:.0f})" for _,r in top_buy.iterrows()))

    lines.append("\n## Key Notes")
    lines.append(f"- Correlation filter blocked pairs above {int(CORR_THRESHOLD*100)}% — holdings are genuinely diversified")
    lines.append(f"- Risk-free rate: {RISK_FREE*100:.1f}% — Sharpe below 1.0 means insufficient risk compensation")
    lines.append(f"- Inverse-volatility weighting: lower-vol ETFs receive higher allocations systematically")
    return "\n".join(lines)


with tabs[4]:
    st.title("AI Analyst")

    mode = st.radio("Mode", ["Quick Brief (free)", "Claude AI Chat (API key required)"], horizontal=True)
    st.divider()

    if mode == "Quick Brief (free)":
        if st.button("Generate Morning Brief", type="primary"):
            st.markdown(generate_quick_brief())

    else:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            st.warning("Enter your Anthropic API key in the sidebar.")
        else:
            try:
                import anthropic
            except ImportError:
                st.error("Run: pip3 install anthropic")
                st.stop()

            system = (f"You are a professional portfolio analyst. The user has a {risk} risk profile, "
                      f"invests ${monthly:,.0f}/month, and holds: " +
                      ", ".join(f"{r['ticker']} ({r['weight_pct']:.1f}%)" for _,r in target.iterrows()) +
                      f". Avg Sharpe: {target['sharpe'].mean():.2f}. "
                      "Answer concisely and professionally.")

            if "messages" not in st.session_state:
                st.session_state.messages = []

            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            if not st.session_state.messages:
                suggestions = ["Give me a morning brief", "Am I well diversified?",
                               "What macro risks should I watch?", "Explain my allocation"]
                cols = st.columns(len(suggestions))
                for col, s in zip(cols, suggestions):
                    if col.button(s, key=s):
                        st.session_state.messages.append({"role":"user","content":s})
                        st.rerun()

            if prompt := st.chat_input("Ask about your portfolio…"):
                st.session_state.messages.append({"role":"user","content":prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                    with st.spinner("Analyzing…"):
                        resp = client.messages.create(
                            model="claude-sonnet-4-6", max_tokens=1024,
                            system=system,
                            messages=[{"role":m["role"],"content":m["content"]}
                                      for m in st.session_state.messages])
                    reply = resp.content[0].text
                    st.markdown(reply)
                    st.session_state.messages.append({"role":"assistant","content":reply})
