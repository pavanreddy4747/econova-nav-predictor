import pandas as pd
import numpy as np

df = pd.read_csv("nav_with_trends.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

recent = df[df["date"] >= "2025-01-01"].dropna(subset=["future_5d_return"])
recent_worst = recent.sort_values("future_5d_return").head(10)

recent_worst.to_csv("recent_backtest.csv", index=False)
print("Regenerated recent_backtest.csv with updated columns")
print(recent_worst.columns.tolist())
