# config.py

# Date range for data collection and forecasting
START_DATE = "2015-07-01"
END_DATE = "2025-07-31"

# Assets to analyze
ASSETS = {
    "TSLA": "Tesla",
    "SPY": "S&P 500 ETF",
    "BND": "Total Bond Market ETF",
    "GLD": "Gold ETF (GLD)"

}

# Data paths
RAW_DATA_PATH = "../data/raw/"
PROCESSED_DATA_PATH = "../data/processed/"

# Hardware configuration
USE_GPU = False  
