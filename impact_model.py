import pandas as pd
import numpy as np

stats_df = pd.read_csv("liquidity_stats.csv")

def predict_impact(trade_value_rupees, stock_price, avg_daily_volume, volatility, k=1.0):
    trade_shares = trade_value_rupees / stock_price
    participation_rate = trade_shares / avg_daily_volume
    impact_pct = k * volatility * np.sqrt(participation_rate)
    return impact_pct * 100

test_prices = {"ICICIBANK.NS": 1250, "INFY.NS": 1780, "LT.NS": 3600, "RELIANCE.NS": 1400}
trade_size = 50000000

print("Simulating a Rs 5 crore trade in each stock:")
print("")
for _, row in stats_df.iterrows():
    ticker = row["ticker"]
    price = test_prices[ticker]
    impact = predict_impact(trade_size, price, row["avg_daily_volume"], row["daily_volatility"])
    print(f"{ticker}: predicted NAV impact = {impact:.4f}%%")
