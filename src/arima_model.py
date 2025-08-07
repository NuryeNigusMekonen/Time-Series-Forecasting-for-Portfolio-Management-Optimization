import pandas as pd
import numpy as np
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

def train_arima(series, order=(5, 1, 0)):
    # Split train-test (e.g. 80-20 split)
    split = int(len(series) * 0.8)
    train, test = series[:split], series[split:]

    model = ARIMA(train, order=order)
    model_fit = model.fit()

    # Forecast for test period
    forecast = model_fit.forecast(steps=len(test))
    rmse = np.sqrt(mean_squared_error(test, forecast))

    return model_fit, rmse


def evaluate_arima(model, train, test):
    predictions = model.predict(n_periods=len(test))
    mae = mean_absolute_error(test, predictions)
    rmse = np.sqrt(mean_squared_error(test, predictions))
    return predictions, mae, rmse

def forecast_future(model, periods=252):
    forecast, conf_int = model.predict(n_periods=periods, return_conf_int=True)
    return forecast, conf_int
