import pandas as pd
import numpy as np

df = pd.read_csv("stock_prices.csv", header=[0,1], index_col=0)

tickers = ["ICICIBANK.NS", "INFY.NS", "LT.NS", "RELIANCE.NS"]
results = []

for t in tickers:
    close = df[("Close", t)]
    volume = df[("Volume", t)]
    daily_returns = close.pct_change().dropna()
    volatility = daily_returns.std()
    avg_volume = volume.mean()
    results.append({"ticker": t, "avg_daily_volume": avg_volume, "daily_volatility": volatility})

stats_df = pd.DataFrame(results)
print(stats_df)
stats_df.to_csv("liquidity_stats.csv", index=False)
print("Saved to liquidity_stats.csv")
