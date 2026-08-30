import pandas as pd
import numpy as np

df = pd.read_csv("stock_prices.csv", header=[0,1], index_col=0)

tickers = ["ICICIBANK.NS", "INFY.NS", "LT.NS", "RELIANCE.NS"]
closes = pd.DataFrame()
for t in tickers:
    closes[t.replace(".NS", "")] = df[("Close", t)]

returns = closes.pct_change().dropna()
corr_matrix = returns.corr()

print("Correlation Matrix (daily returns):")
print(corr_matrix.round(3))

corr_matrix.to_csv("correlation_matrix.csv")
print("\nSaved to correlation_matrix.csv")

pairs = []
for i in range(len(tickers)):
    for j in range(i+1, len(tickers)):
        t1, t2 = tickers[i].replace(".NS", ""), tickers[j].replace(".NS", "")
        pairs.append({"stock_1": t1, "stock_2": t2, "correlation": corr_matrix.loc[t1, t2]})

pairs_df = pd.DataFrame(pairs).sort_values("correlation", ascending=False)
print("\nMost correlated pairs:")
print(pairs_df)
pairs_df.to_csv("correlation_pairs.csv", index=False)
