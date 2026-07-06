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

# Risk profile pools — which ETFs are eligible per risk level
RISK_PROFILES = {
    "Conservative": [
        "GLD","IAU","VNQ",                                   # real assets
        "VTV","VONV","IWD","SCHV","VOOV",                   # value
        "XLP","XLU","XLV","XLF",                            # defensive sectors
        "VOO","VTI","ITOT","IWB",                           # broad market anchor
        "EFA","VEA",                                         # international developed
    ],
    "Moderate": [
        "VOO","VTI","VOOG","IWM","ITOT","SCHB",             # broad market
        "VTV","IWD","SCHV","VOOV",                           # value
        "QQQ","QQQM",                                        # growth (limited)
        "XLF","XLI","XLC","XLV","XLE",                      # sectors
        "EFA","VEA","IEMG",                                  # international
        "GLD",                                               # real asset hedge
    ],
    "Aggressive": [
        "QQQ","QQQM","VGT","XLK","SOXX","SMH","IGV","ARKK", # growth & tech
        "BOTZ","AIQ","HACK","DRIV","ICLN","CARZ",            # thematic
        "IWM","VOOG",                                        # growth-oriented broad
        "XLC","XLY",                                         # high-beta sectors
        "EEM","VWO","FXI",                                   # emerging markets
    ],
}

TICKERS = [t for group in ETF_UNIVERSE.values() for t in group]
