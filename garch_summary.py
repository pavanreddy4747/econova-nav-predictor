import pandas as pd
import numpy as np
from arch import arch_model
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("nav_with_trends.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

returns = df["daily_return"].dropna() * 100

model = arch_model(returns, vol="Garch", p=1, q=1, dist="normal")
fitted = model.fit(disp="off")

forecast = fitted.forecast(horizon=5)
predicted_variance = forecast.variance.values[-1]
garch_volatility = np.sqrt(predicted_variance) / 100

current_10d_vol = df["rolling_volatility_10d"].iloc[-1]
current_30d_vol = df["rolling_volatility_30d"].iloc[-1]

persistence = fitted.params["alpha[1]"] + fitted.params["beta[1]"]

summary = {
    "rolling_10d_volatility": float(current_10d_vol),
    "rolling_30d_volatility": float(current_30d_vol),
    "garch_next_day_forecast": float(garch_volatility[0]),
    "garch_5day_avg_forecast": float(garch_volatility.mean()),
    "garch_persistence": float(persistence),
    "garch_day1": float(garch_volatility[0]),
    "garch_day2": float(garch_volatility[1]),
    "garch_day3": float(garch_volatility[2]),
    "garch_day4": float(garch_volatility[3]),
    "garch_day5": float(garch_volatility[4]),
}

pd.DataFrame([summary]).to_csv("garch_summary.csv", index=False)
print("Saved garch_summary.csv")
print(summary)
