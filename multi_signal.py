import pandas as pd
import numpy as np

df = pd.read_csv("hdfc_largecap_nav.csv")
df["date"] = pd.to_datetime(df["date"], dayfirst=True)
df = df.sort_values("date").reset_index(drop=True)

df["daily_return"] = df["nav"].pct_change()

df["rolling_avg_return_5d"] = df["daily_return"].rolling(5).mean()
df["rolling_avg_return_10d"] = df["daily_return"].rolling(10).mean()
df["rolling_avg_return_30d"] = df["daily_return"].rolling(30).mean()

df["rolling_volatility_5d"] = df["daily_return"].rolling(5).std()
df["rolling_volatility_10d"] = df["daily_return"].rolling(10).std()
df["rolling_volatility_30d"] = df["daily_return"].rolling(30).std()

df["future_5d_return"] = df["nav"].shift(-5) / df["nav"] - 1

df.to_csv("nav_with_trends.csv", index=False)
print("Updated nav_with_trends.csv with multi-timeframe signals")

recent = df[df["date"] >= "2025-01-01"].dropna(subset=["future_5d_return"])
recent_worst = recent.sort_values("future_5d_return").head(10)
print(recent_worst[["date", "nav", "rolling_avg_return_5d", "rolling_avg_return_10d", "rolling_avg_return_30d", "future_5d_return"]])
