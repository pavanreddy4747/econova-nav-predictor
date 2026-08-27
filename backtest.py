import pandas as pd
import numpy as np

df = pd.read_csv("nav_with_trends.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

df["future_5d_return"] = df["nav"].shift(-5) / df["nav"] - 1

worst_drops = df.dropna(subset=["future_5d_return"]).sort_values("future_5d_return").head(10)

print("Top 10 worst 5-day NAV drops in history:")
print(worst_drops[["date", "nav", "rolling_avg_return_30d", "rolling_volatility_30d", "future_5d_return"]])

worst_drops.to_csv("worst_drops_backtest.csv", index=False)
print("Saved to worst_drops_backtest.csv")
