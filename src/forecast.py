import os
import warnings
import pandas as pd
import numpy as np

from typing import Union, Dict, List
from datetime import datetime

# ARIMA
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA

# LSTM
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

warnings.filterwarnings("ignore")

# Paths
DATA_DIR = "../data/raw"
FORECASTS_DIR = "../forecasts"
os.makedirs(FORECASTS_DIR, exist_ok=True)


def load_data(asset: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{asset}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data for {asset} not found at {path}")
    
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df[["Date", "Close"]].dropna().sort_values("Date")
    df.set_index("Date", inplace=True)
    return df

# ARIMA FORECASTING

def forecast_arima(df: pd.DataFrame, periods: int = 30) -> pd.DataFrame:
    print("Training ARIMA model...")
    model = pm.auto_arima(df, seasonal=False, suppress_warnings=True, stepwise=True)
    forecast = model.predict(n_periods=periods)
    
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=periods, freq='B')  # business days
    return pd.DataFrame({"Date": future_dates, "Forecast_ARIMA": forecast}).set_index("Date")


# LSTM FORECASTING

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def forecast_lstm(df: pd.DataFrame, periods: int = 30, seq_len: int = 30, epochs: int = 20) -> pd.DataFrame:
    print("Training LSTM model...")
    series = df["Close"].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series)

    X, y = create_sequences(scaled, seq_len)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(50, activation='relu', input_shape=(seq_len, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, verbose=0)

    forecast_input = scaled[-seq_len:].reshape(1, seq_len, 1)
    forecasts = []
    for _ in range(periods):
        next_pred = model.predict(forecast_input)[0][0]
        forecasts.append(next_pred)
        forecast_input = np.append(forecast_input[:, 1:, :], [[[next_pred]]], axis=1)

    forecast = scaler.inverse_transform(np.array(forecasts).reshape(-1, 1)).flatten()
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=periods, freq='B')
    return pd.DataFrame({"Date": future_dates, "Forecast_LSTM": forecast}).set_index("Date")

# MAIN FORECAST FUNCTION

def generate_forecasts(asset: str, days_ahead: int = 30) -> pd.DataFrame:
    try:
        df = load_data(asset)
        print(f"Loaded data for {asset}: {df.shape[0]} rows")

        arima_df = forecast_arima(df, periods=days_ahead)
        lstm_df = forecast_lstm(df, periods=days_ahead)

        merged = pd.concat([arima_df, lstm_df], axis=1)
        merged.to_csv(os.path.join(FORECASTS_DIR, f"{asset}_forecast.csv"))
        print(f"Forecasts saved for {asset}")
        return merged

    except Exception as e:
        print(f"Error while forecasting {asset}: {str(e)}")
        return pd.DataFrame()

# New wrapper function to run forecasting on multiple assets

def run_forecasting(assets: List[str], days_ahead: int = 30) -> Dict[str, pd.DataFrame]:
    results = {}
    for asset in assets:
        print(f"\n==== Running forecast for {asset} ====")
        results[asset] = generate_forecasts(asset, days_ahead)
    return results


# ENTRY POINT

if __name__ == "__main__":
    assets = ["TSLA", "SPY", "BND", "XAUUSD"]
    run_forecasting(assets)
