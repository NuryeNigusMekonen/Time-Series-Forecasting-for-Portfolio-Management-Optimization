import os
import requests
import pandas as pd
import yfinance as yf
from src.config import ASSETS, START_DATE, END_DATE, RAW_DATA_PATH, ALPHA_VANTAGE_KEY

def download_data(ticker: str, start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    """Download historical data from Yahoo Finance."""
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
    """Save DataFrame to CSV in raw data folder."""
    try:
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, f"{ticker}.csv")
        df.to_csv(file_path, index=False)
        print(f"[INFO] Saved data for {ticker} to {file_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save data for {ticker}: {e}")

def fetch_latest_spot_gold_price():
    """Fetch latest spot gold price from Alpha Vantage realtime API."""
    try:
        url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency=XAU&to_currency=USD&apikey={ALPHA_VANTAGE_KEY}"
        r = requests.get(url).json()

        if "Realtime Currency Exchange Rate" not in r:
            raise ValueError(f"Alpha Vantage error: {r}")

        price = float(r["Realtime Currency Exchange Rate"]["5. Exchange Rate"])
        timestamp = r["Realtime Currency Exchange Rate"]["6. Last Refreshed"]

        return price, timestamp
    except Exception as e:
        print(f"[ERROR] Failed to fetch latest spot gold price: {e}")
        return None, None

def fetch_and_save_all():
    """Fetch and save all assets + update gold futures with spot price."""
    # Fetch stocks + gold futures from Yahoo Finance
    for ticker in ASSETS:
        df = download_data(ticker)
        if not df.empty:
            save_data(df, ticker)

    # Now get latest spot gold price from Alpha Vantage
    spot_price, spot_time = fetch_latest_spot_gold_price()
    if spot_price is None:
        print("[WARN] Could not get latest spot gold price, skipping update.")
        return

    # Load gold futures data (GC=F) saved earlier
    gold_path = os.path.join(RAW_DATA_PATH, "GC=F.csv")
    if not os.path.exists(gold_path):
        print("[WARN] Gold futures data not found, cannot update with spot price.")
        return

    gold_df = pd.read_csv(gold_path, parse_dates=["Date"])
    gold_df.sort_values("Date", inplace=True)

    # Update or append latest spot price with today's date (or spot_time date)
    spot_date = pd.to_datetime(spot_time).date()
    if gold_df["Date"].iloc[-1].date() == spot_date:
        # Replace last row's Close, Open, High, Low with spot price (approximate)
        gold_df.loc[gold_df.index[-1], ["Open", "High", "Low", "Close"]] = spot_price
    else:
        # Append new row for spot date with price in OHLC columns
        new_row = {
            "Date": spot_date,
            "Open": spot_price,
            "High": spot_price,
            "Low": spot_price,
            "Close": spot_price,
            "Adj Close": spot_price,
            "Volume": 0,
            "Ticker": "GC=F"
        }
        gold_df = pd.concat([gold_df, pd.DataFrame([new_row])], ignore_index=True)

    save_data(gold_df, "GC=F")
    print(f"[INFO] Updated GC=F.csv with latest spot gold price {spot_price} at {spot_time}")
