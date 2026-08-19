import pandas as pd
import numpy as np

nav_df = pd.read_csv("nav_with_trends.csv")
latest = nav_df.iloc[-1]
trend_score = 1 if latest["rolling_avg_return_30d"] < 0 else 0
vol_score = latest["rolling_volatility_30d"]

stats_df = pd.read_csv("liquidity_stats.csv")

def predict_impact(trade_value_rupees, stock_price, avg_daily_volume, volatility, k=1.0):
    trade_shares = trade_value_rupees / stock_price
    participation_rate = trade_shares / avg_daily_volume
    impact_pct = k * volatility * np.sqrt(participation_rate)
    return impact_pct * 100

test_prices = {"ICICIBANK.NS": 1250, "INFY.NS": 1780, "LT.NS": 3600, "RELIANCE.NS": 1400}
trade_size = 50000000

results = []
for _, row in stats_df.iterrows():
    ticker = row["ticker"]
    price = test_prices[ticker]
    impact = predict_impact(trade_size, price, row["avg_daily_volume"], row["daily_volatility"])
    risk_score = impact * (1 + trend_score * 0.5) * (1 + vol_score * 10)
    results.append({"ticker": ticker, "predicted_impact_pct": round(impact, 4), "redemption_trend_risk": trend_score, "final_risk_score": round(risk_score, 4)})

risk_df = pd.DataFrame(results).sort_values("final_risk_score", ascending=False)
print(risk_df)
risk_df.to_csv("final_risk_scores.csv", index=False)
print("Saved to final_risk_scores.csv")
