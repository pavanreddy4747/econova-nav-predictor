import pandas as pd
import numpy as np

df = pd.read_csv("nav_with_trends.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

df["future_5d_return"] = df["nav"].shift(-5) / df["nav"] - 1

recent = df[df["date"] >= "2025-01-01"].dropna(subset=["future_5d_return"])
recent_worst = recent.sort_values("future_5d_return").head(10)

print("Worst 10 5-day NAV drops since Jan 2025 (recent period):")
print(recent_worst[["date", "nav", "rolling_avg_return_30d", "rolling_volatility_30d", "future_5d_return"]])

recent_worst.to_csv("recent_backtest.csv", index=False)
print("Saved to recent_backtest.csv")
