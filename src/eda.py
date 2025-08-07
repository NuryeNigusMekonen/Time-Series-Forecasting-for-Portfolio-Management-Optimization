import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import PROCESSED_DATA_PATH, ASSETS

sns.set(style="whitegrid")

def load_processed_data(ticker: str) -> pd.DataFrame:
    return pd.read_csv(f"{PROCESSED_DATA_PATH}{ticker}_processed.csv", parse_dates=["Date"])

def plot_closing_prices(df: pd.DataFrame, ticker: str):
    plt.figure(figsize=(12, 5))
    plt.plot(df["Date"], df["Adj Close"])
    plt.title(f"{ticker} - Adjusted Closing Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.show()

def plot_returns_distribution(df: pd.DataFrame, ticker: str):
    plt.figure(figsize=(12, 5))
    sns.histplot(df["Return"], bins=50, kde=True)
    plt.title(f"{ticker} - Return Distribution")
    plt.xlabel("Return")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

def plot_rolling_volatility(df: pd.DataFrame, ticker: str):
    plt.figure(figsize=(12, 5))
    # Support either column name
    col = None
    for candidate in ["RollingVolatility_30", "RollingVolatility"]:
        if candidate in df.columns:
            col = candidate
            break

    if col is None:
        print(f"[WARN] No rolling volatility column found for {ticker}")
        return

    plt.plot(df["Date"], df[col])
    plt.title(f"{ticker} - 30-Day Rolling Volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.tight_layout()
    plt.show()


def run_eda():
    for ticker in ASSETS:
        print(f"[INFO] Running EDA for {ticker}")
        df = load_processed_data(ticker)
        plot_closing_prices(df, ticker)
        plot_returns_distribution(df, ticker)
        plot_rolling_volatility(df, ticker)
