import pandas as pd
import numpy as np
from arch import arch_model

df = pd.read_csv("nav_with_trends.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

returns = df["daily_return"].dropna() * 100

model = arch_model(returns, vol="Garch", p=1, q=1, dist="normal")
fitted = model.fit(disp="off")

print(fitted.summary())

forecast = fitted.forecast(horizon=5)
predicted_variance = forecast.variance.values[-1]
predicted_volatility = np.sqrt(predicted_variance) / 100

print("\nGARCH-forecasted daily volatility for next 5 days:")
for i, v in enumerate(predicted_volatility, 1):
    print(f"Day {i}: {v*100:.4f}%")

current_vol = df["rolling_volatility_10d"].iloc[-1]
print(f"\nCurrent rolling 10-day volatility (old method): {current_vol*100:.4f}%")
print(f"GARCH-forecasted next-day volatility (new method): {predicted_volatility[0]*100:.4f}%")

forecast_df = pd.DataFrame({
    "day_ahead": range(1, 6),
    "garch_forecasted_volatility": predicted_volatility
})
forecast_df.to_csv("garch_forecast.csv", index=False)
print("\nSaved to garch_forecast.csv")
