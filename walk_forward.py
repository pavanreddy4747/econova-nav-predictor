import pandas as pd
import numpy as np

df = pd.read_csv("nav_with_trends.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

df["future_5d_return"] = df["nav"].shift(-5) / df["nav"] - 1
df = df.dropna(subset=["future_5d_return", "rolling_avg_return_5d", "rolling_avg_return_10d", "rolling_avg_return_30d"])

train_size = int(len(df) * 0.7)
train_df = df.iloc[:train_size].copy()
test_df = df.iloc[train_size:].copy()

print(f"Train period: {train_df['date'].min().date()} to {train_df['date'].max().date()} ({len(train_df)} days)")
print(f"Test period (out-of-sample): {test_df['date'].min().date()} to {test_df['date'].max().date()} ({len(test_df)} days)")

def signal_hit_rate(data, threshold_percentile=10):
    worst_n = int(len(data) * threshold_percentile / 100)
    worst_days = data.nsmallest(worst_n, "future_5d_return")
    flagged = worst_days["rolling_avg_return_5d"] < 0
    return flagged.sum(), len(worst_days), flagged.mean()

train_hits, train_total, train_rate = signal_hit_rate(train_df)
print(f"\nTRAIN (in-sample) hit rate: {train_hits}/{train_total} = {train_rate:.1%}")

test_hits, test_total, test_rate = signal_hit_rate(test_df)
print(f"TEST (out-of-sample, walk-forward) hit rate: {test_hits}/{test_total} = {test_rate:.1%}")

print(f"\nDegradation from train to test: {(train_rate - test_rate)*100:.1f} percentage points")
if test_rate < train_rate * 0.7:
    print("WARNING: Significant performance drop out-of-sample. Signal may be overfit to training period.")
else:
    print("Signal holds up reasonably well out-of-sample.")

results = pd.DataFrame({
    "period": ["train (in-sample)", "test (out-of-sample)"],
    "hit_rate": [train_rate, test_rate],
    "n_worst_days_tested": [train_total, test_total]
})
results.to_csv("walk_forward_results.csv", index=False)
print("\nSaved walk_forward_results.csv")
