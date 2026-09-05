import pandas as pd
import numpy as np

nifty = pd.read_csv("nifty_prices.csv", header=[0,1], index_col=0)
nifty_close = nifty[("Close", "^NSEI")]
nifty_close.index = pd.to_datetime(nifty_close.index)
nifty_close = nifty_close.sort_index()

nifty_returns = nifty_close.pct_change()
nifty_df = pd.DataFrame({"date": nifty_close.index, "nifty_close": nifty_close.values})
nifty_df["nifty_return"] = nifty_returns.values
nifty_df["nifty_rolling_5d"] = nifty_df["nifty_return"].rolling(5).mean()
nifty_df["nifty_rolling_10d"] = nifty_df["nifty_return"].rolling(10).mean()

fund_df = pd.read_csv("nav_with_trends.csv")
fund_df["date"] = pd.to_datetime(fund_df["date"])

merged = pd.merge(fund_df, nifty_df, on="date", how="inner")
print(f"Merged dataset: {len(merged)} overlapping trading days")

corr_fund_nifty = merged["daily_return"].corr(merged["nifty_return"])
print(f"\nCorrelation between fund daily return and Nifty daily return: {corr_fund_nifty:.4f}")

merged["future_5d_return"] = merged["nav"].shift(-5) / merged["nav"] - 1

corr_nifty5d_future = merged["nifty_rolling_5d"].corr(merged["future_5d_return"])
corr_fund5d_future = merged["rolling_avg_return_5d"].corr(merged["future_5d_return"])

print(f"\nPredictive power (correlation with future 5-day fund return):")
print(f"  Nifty 5-day trend -> future fund return: {corr_nifty5d_future:.4f}")
print(f"  Fund's own 5-day trend -> future fund return: {corr_fund5d_future:.4f}")

merged.to_csv("merged_with_nifty.csv", index=False)
print("\nSaved merged_with_nifty.csv")
