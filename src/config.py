# config.py

START_DATE = "2015-07-01"
END_DATE = "2025-07-31"

ASSETS = {
    "TSLA": "Tesla",
    "SPY": "S&P 500 ETF",
    "BND": "Total Bond Market ETF",
    "GC=F": "Gold Futures"
}

# Gold spot API config
ALPHA_VANTAGE_KEY = "VNRK6DHVEXNF2QOZ"  

RAW_DATA_PATH = "../data/raw/"
PROCESSED_DATA_PATH = "../data/processed/"

USE_GPU = False
