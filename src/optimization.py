import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns, plotting

def optimize_portfolio(df_dict):
    price_df = pd.concat([df[["Date", "Adj Close"]].set_index("Date").rename(columns={"Adj Close": ticker}) 
                          for ticker, df in df_dict.items()], axis=1)
    returns = price_df.pct_change().dropna()

    mu = expected_returns.mean_historical_return(price_df)
    S = risk_models.sample_cov(price_df)

    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()

    performance = ef.portfolio_performance(verbose=True)
    return cleaned_weights, performance
