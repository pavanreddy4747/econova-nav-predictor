import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

np.random.seed(42)

def true_impact(trade_value, price, avg_volume, volatility, k=1.0):
    trade_shares = trade_value / price
    participation_rate = trade_shares / avg_volume
    return k * volatility * np.sqrt(participation_rate) * 100

n_samples = 5000
trade_values = np.random.uniform(1_000_000, 500_000_000, n_samples)
prices = np.random.uniform(200, 4000, n_samples)
avg_volumes = np.random.uniform(500_000, 15_000_000, n_samples)
volatilities = np.random.uniform(0.008, 0.03, n_samples)

true_impacts = true_impact(trade_values, prices, avg_volumes, volatilities)
noise = np.random.normal(0, 0.05 * true_impacts.std(), n_samples)
observed_impacts = np.clip(true_impacts + noise, 0, None)

X = pd.DataFrame({
    "trade_value": trade_values,
    "price": prices,
    "avg_volume": avg_volumes,
    "volatility": volatilities,
    "participation_rate": (trade_values / prices) / avg_volumes
})
y = observed_impacts

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Random Forest Model Performance:")
print(f"Mean Absolute Error: {mae:.5f}")
print(f"R-squared: {r2:.5f}")

importances = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)
print("\nFeature Importance:")
print(importances)

joblib.dump(model, "ml_impact_model.pkl")
print("\nModel saved to ml_impact_model.pkl")
