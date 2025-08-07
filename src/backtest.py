import pandas as pd
import numpy as np

def simulate_portfolio(df_dict, weights, start_date, end_date):
    price_df = pd.concat([df[["Date", "Adj Close"]].set_index("Date").rename(columns={"Adj Close": ticker}) 
                          for ticker, df in df_dict.items()], axis=1)

    price_df = price_df[start_date:end_date]
    returns = price_df.pct_change().dropna()

    weighted_returns = (returns * pd.Series(weights)).sum(axis=1)
    cumulative = (1 + weighted_returns).cumprod()
    return cumulative
