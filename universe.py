ETF_UNIVERSE = {
    "Broad Market": [
        "VOO",   # S&P 500
        "VTI",   # Total US market
        "VOOG",  # S&P 500 Growth
        "VOOV",  # S&P 500 Value
        "IWM",   # Russell 2000 small cap
        "IWB",   # Russell 1000 large cap
        "ITOT",  # iShares total market
        "SCHB",  # Schwab broad market
    ],
    "Growth & Tech": [
        "QQQ",   # Nasdaq 100
        "QQQM",  # Nasdaq 100 (lower cost)
        "VGT",   # Vanguard Information Technology
        "XLK",   # Technology Select Sector
        "SOXX",  # Semiconductors
        "SMH",   # VanEck Semiconductors
        "IGV",   # Software
        "ARKK",  # ARK Innovation
    ],
    "Value": [
        "VTV",   # Vanguard Value
        "VONV",  # Vanguard S&P 500 Value
        "IWD",   # iShares Russell 1000 Value
        "SCHV",  # Schwab US Large-Cap Value
        "RPV",   # Invesco S&P 500 Pure Value
    ],
    "Sector": [
        "XLF",   # Financials
        "XLE",   # Energy
        "XLV",   # Health Care
        "XLI",   # Industrials
        "XLY",   # Consumer Discretionary
        "XLP",   # Consumer Staples
        "XLU",   # Utilities
        "XLB",   # Materials
        "XLRE",  # Real Estate
        "XLC",   # Communication Services
    ],
    "International": [
        "EFA",   # Developed markets ex-US
        "EEM",   # Emerging markets
        "VEA",   # Vanguard developed markets
        "VWO",   # Vanguard emerging markets
        "IEMG",  # iShares Core emerging markets
        "EWJ",   # Japan
        "FXI",   # China large cap
    ],
    "Fixed Income": [
        "BND",   # Total bond market
        "AGG",   # iShares core US aggregate bond
        "TLT",   # 20+ year Treasury
        "IEF",   # 7-10 year Treasury
        "SHY",   # 1-3 year Treasury
        "HYG",   # High yield corporate bonds
        "LQD",   # Investment grade corporate bonds
        "VTIP",  # Inflation-protected (TIPS)
    ],
    "Real Assets": [
        "VNQ",   # Real estate (REITs)
        "GLD",   # Gold
        "IAU",   # iShares Gold
        "GSG",   # Commodities
        "PDBC",  # Diversified commodities
        "DBC",   # DB Commodity Index
    ],
    "Thematic / AI": [
        "BOTZ",  # AI & Robotics
        "AIQ",   # AI & Big Data
        "HACK",  # Cybersecurity
        "DRIV",  # Autonomous vehicles
        "ICLN",  # Clean energy
        "CARZ",  # Auto industry
    ],
}

TICKERS = [ticker for group in ETF_UNIVERSE.values() for ticker in group]
