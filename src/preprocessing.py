import pandas as pd
import numpy as np
import os
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, ASSETS

def load_raw_data(ticker: str) -> pd.DataFrame:
    path = os.path.join(RAW_DATA_PATH, f"{ticker}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] File not found: {path}")
    
    # Use new pandas-compatible arg
    df = pd.read_csv(path, parse_dates=["Date"], skiprows=[1], on_bad_lines='skip')
    # Remove unnamed or empty columns if any
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Choose price column
    if "Adj Close" in df.columns:
        price_col = "Adj Close"
    elif "Close" in df.columns:
        price_col = "Close"
    else:
        raise ValueError("Neither 'Adj Close' nor 'Close' found in the data.")
    
    # Convert to numeric
    df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
    df = df.dropna(subset=[price_col])

    # Rename chosen price column to "Adj Close"
    if price_col != "Adj Close":
        df = df.rename(columns={price_col: "Adj Close"})

    # Then compute features using "Adj Close"
    df["Return"] = df["Adj Close"].pct_change()
    df["Log_Return"] = np.log1p(df["Return"])
    df["RollingVolatility_30"] = df["Return"].rolling(window=30).std()
    df["RollingMean_7"] = df["Adj Close"].rolling(window=7).mean()
    df["RollingMean_30"] = df["Adj Close"].rolling(window=30).mean()
    df["Momentum_7"] = df["Adj Close"] - df["RollingMean_7"]
    df["Momentum_30"] = df["Adj Close"] - df["RollingMean_30"]
    df["ROC_7"] = df["Adj Close"].pct_change(periods=7)
    df["EMA_12"] = df["Adj Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Adj Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]

    df = df.dropna().reset_index(drop=True)
    return df



def save_processed_data(df: pd.DataFrame, ticker: str):
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    out_path = os.path.join(PROCESSED_DATA_PATH, f"{ticker}_processed.csv")
    df.to_csv(out_path, index=False)
    print(f"[INFO] Processed data saved to {out_path}")

def process_all_assets():
    for ticker in ASSETS:
        try:
            print(f"[INFO] Processing {ticker} ...")
            raw_df = load_raw_data(ticker)
            processed_df = preprocess_data(raw_df)
            save_processed_data(processed_df, ticker)
        except Exception as e:
            print(f"[ERROR] Failed to process {ticker}: {e}")

if __name__ == "__main__":
    process_all_assets()
