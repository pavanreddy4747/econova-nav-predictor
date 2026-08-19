import pandas as pd
import numpy as np

df = pd.read_csv("hdfc_largecap_nav.csv")
df["date"] = pd.to_datetime(df["date"], dayfirst=True)
df = df.sort_values("date").reset_index(drop=True)

df["daily_return"] = df["nav"].pct_change()
df["rolling_volatility_30d"] = df["daily_return"].rolling(30).std()
df["rolling_avg_return_30d"] = df["daily_return"].rolling(30).mean()

print(df.tail(15)[["date", "nav", "daily_return", "rolling_volatility_30d", "rolling_avg_return_30d"]])
df.to_csv("nav_with_trends.csv", index=False)
print("Saved to nav_with_trends.csv")
