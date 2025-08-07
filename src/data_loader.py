import os
import yfinance as yf
import pandas as pd
from src.config import ASSETS, START_DATE, END_DATE, RAW_DATA_PATH

def download_data(ticker: str, start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    try:
        df = yf.download(ticker, start=start, end=end)
        if df.empty:
            raise ValueError(f"No data returned for {ticker}")
        df.reset_index(inplace=True)
        df["Ticker"] = ticker
        return df
    except Exception as e:
        print(f"[ERROR] Failed to fetch data for {ticker}: {e}")
        return pd.DataFrame()

def save_data(df: pd.DataFrame, ticker: str, path: str = RAW_DATA_PATH):
    try:
        os.makedirs(path, exist_ok=True)
        df.to_csv(os.path.join(path, f"{ticker}.csv"), index=False)
        print(f"[INFO] Saved data for {ticker} to {path}")
    except Exception as e:
        print(f"[ERROR] Failed to save data for {ticker}: {e}")

def fetch_and_save_all():
    for ticker in ASSETS:
        df = download_data(ticker)
        if not df.empty:
            save_data(df, ticker)

def load_stock_data(ticker: str) -> pd.DataFrame:
    file_path = os.path.join(RAW_DATA_PATH, f"{ticker}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No saved data found for {ticker} at {file_path}")
    df = pd.read_csv(file_path, parse_dates=["Date"])
    return df
