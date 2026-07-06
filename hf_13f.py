import requests
import pandas as pd
from bs4 import BeautifulSoup
import streamlit as st

FUNDS = {
    "Citadel":    "0001423053",
    "Millennium": "0001273931",
    "Two Sigma":  "0001179392",
    "D.E. Shaw":  "0001016112",
    "Point72":    "0001603466",
}

HEADERS = {"User-Agent": "investment-toolkit rafaelcandeloro@gmail.com"}

# Maps company names as they appear in 13F filings → Yahoo Finance tickers
NAME_TO_TICKER = {
    "APPLE INC": "AAPL", "MICROSOFT CORP": "MSFT", "AMAZON COM INC": "AMZN",
    "NVIDIA CORP": "NVDA", "META PLATFORMS INC": "META", "ALPHABET INC": "GOOGL",
    "TESLA INC": "TSLA", "BROADCOM INC": "AVGO", "JPMORGAN CHASE & CO": "JPM",
    "UNITEDHEALTH GROUP INC": "UNH", "EXXON MOBIL CORP": "XOM", "VISA INC": "V",
    "MASTERCARD INC": "MA", "PROCTER & GAMBLE CO": "PG", "ELI LILLY & CO": "LLY",
    "HOME DEPOT INC": "HD", "ABBVIE INC": "ABBV", "COSTCO WHOLESALE CORP": "COST",
    "COCA COLA CO": "KO", "WALMART INC": "WMT", "CHEVRON CORP": "CVX",
    "MERCK & CO INC": "MRK", "ADOBE INC": "ADBE", "SALESFORCE INC": "CRM",
    "ADVANCED MICRO DEVICES INC": "AMD", "NETFLIX INC": "NFLX", "INTUIT INC": "INTU",
    "QUALCOMM INC": "QCOM", "TEXAS INSTRUMENTS INC": "TXN", "SERVICENOW INC": "NOW",
    "CATERPILLAR INC": "CAT", "GOLDMAN SACHS GROUP INC": "GS", "MORGAN STANLEY": "MS",
    "BANK OF AMERICA CORP": "BAC", "WELLS FARGO & CO": "WFC", "CITIGROUP INC": "C",
    "AMERICAN EXPRESS CO": "AXP", "INTUITIVE SURGICAL INC": "ISRG",
    "BOOKING HOLDINGS INC": "BKNG", "PALO ALTO NETWORKS INC": "PANW",
    "CROWDSTRIKE HOLDINGS INC": "CRWD", "SNOWFLAKE INC": "SNOW",
    "DATADOG INC": "DDOG", "CLOUDFLARE INC": "NET", "UBER TECHNOLOGIES INC": "UBER",
    "PALANTIR TECHNOLOGIES INC": "PLTR", "SHOPIFY INC": "SHOP", "PAYPAL HOLDINGS INC": "PYPL",
    "TAIWAN SEMICONDUCTOR MFG CO LTD": "TSM", "ASML HOLDING NV": "ASML",
    "DELL TECHNOLOGIES INC": "DELL", "ORACLE CORP": "ORCL", "INTEL CORP": "INTC",
    "MICRON TECHNOLOGY INC": "MU", "APPLIED MATERIALS INC": "AMAT",
    "LAM RESEARCH CORP": "LRCX", "MARVELL TECHNOLOGY INC": "MRVL",
    "ARISTA NETWORKS INC": "ANET", "FORTINET INC": "FTNT", "WORKDAY INC": "WDAY",
    "MONGODB INC": "MDB", "ATLASSIAN CORP": "TEAM", "REGENERON PHARMACEUTICALS INC": "REGN",
    "GILEAD SCIENCES INC": "GILD", "AMGEN INC": "AMGN", "VERTEX PHARMACEUTICALS INC": "VRTX",
    "MODERNA INC": "MRNA", "BOSTON SCIENTIFIC CORP": "BSX", "STRYKER CORP": "SYK",
    "MEDTRONIC PLC": "MDT", "THERMO FISHER SCIENTIFIC INC": "TMO",
    "DANAHER CORP": "DHR", "DEERE & CO": "DE", "UNION PACIFIC CORP": "UNP",
    "LOCKHEED MARTIN CORP": "LMT", "RAYTHEON TECHNOLOGIES CORP": "RTX",
    "NORTHROP GRUMMAN CORP": "NOC", "GENERAL ELECTRIC CO": "GE",
    "HONEYWELL INTERNATIONAL INC": "HON", "3M CO": "MMM",
    "SPDR S&P 500 ETF TR": "SPY", "INVESCO QQQ TR": "QQQ",
    "ISHARES RUSSELL 2000 ETF": "IWM", "VANGUARD SP 500 ETF": "VOO",
}


def _parse_infotable(xml_text: str) -> list:
    soup = BeautifulSoup(xml_text, "lxml-xml")
    entries = soup.find_all("infoTable")
    holdings = []
    for entry in entries:
        name  = entry.find("nameOfIssuer")
        value = entry.find("value")
        shrs  = entry.find("sshPrnamt")
        if name and value:
            try:
                holdings.append({
                    "name":             name.text.strip().upper(),
                    "value_thousands":  int(value.text.replace(",", "")),
                    "shares":           int(shrs.text.replace(",", "")) if shrs else 0,
                })
            except ValueError:
                continue
    return holdings


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_fund_holdings(fund_name: str, cik: str) -> tuple[pd.DataFrame, str]:
    """Returns (holdings_df, filing_date). holdings_df has columns: fund, name, ticker, value_thousands."""
    try:
        padded = cik.lstrip("0").zfill(10)
        meta   = requests.get(f"https://data.sec.gov/submissions/CIK{padded}.json",
                              headers=HEADERS, timeout=15)
        if meta.status_code != 200:
            return pd.DataFrame(), ""

        filings     = meta.json().get("filings", {}).get("recent", {})
        forms       = filings.get("form", [])
        accessions  = filings.get("accessionNumber", [])
        dates       = filings.get("filingDate", [])

        for i, form in enumerate(forms):
            if form != "13F-HR":
                continue

            acc_clean  = accessions[i].replace("-", "")
            acc_dashed = accessions[i]
            cik_int    = int(cik.lstrip("0"))
            filing_date = dates[i]

            idx = requests.get(
                f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_clean}/{acc_dashed}-index.json",
                headers=HEADERS, timeout=15
            )
            if idx.status_code != 200:
                continue

            for f in idx.json().get("directory", {}).get("item", []):
                fname = f.get("name", "").lower()
                if "infotable" in fname and fname.endswith(".xml"):
                    xml_r = requests.get(
                        f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_clean}/{f['name']}",
                        headers=HEADERS, timeout=15
                    )
                    if xml_r.status_code != 200:
                        continue
                    rows = _parse_infotable(xml_r.text)
                    if not rows:
                        continue
                    df = pd.DataFrame(rows)
                    df["fund"]   = fund_name
                    df["ticker"] = df["name"].map(NAME_TO_TICKER)
                    return df, filing_date

    except Exception:
        pass
    return pd.DataFrame(), ""


@st.cache_data(ttl=86400, show_spinner=False)
def get_aggregate_holdings() -> pd.DataFrame:
    """Aggregate top holdings across all funds; return ranked DataFrame."""
    frames = []
    for fund_name, cik in FUNDS.items():
        df, _ = fetch_fund_holdings(fund_name, cik)
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["ticker"].notna()].copy()

    agg = combined.groupby("ticker").agg(
        name        = ("name",            "first"),
        total_value = ("value_thousands", "sum"),
        fund_count  = ("fund",            "nunique"),
        funds       = ("fund",            lambda x: ", ".join(sorted(x.unique()))),
    ).reset_index()

    agg["value_rank"] = agg["total_value"].rank(pct=True)
    agg["count_rank"] = agg["fund_count"].rank(pct=True)
    agg["hf_score"]   = agg["value_rank"] * 0.6 + agg["count_rank"] * 0.4

    return agg.sort_values("hf_score", ascending=False).reset_index(drop=True)
