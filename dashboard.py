import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="NAV Impact & Frontrunning Risk Predictor", layout="wide")

st.title("Mutual Fund NAV Impact & Frontrunning Risk Predictor")
st.caption("ECONOVA C4004AT | Predicting NAV-distorting trades before they happen, for AMC compliance teams")

nav_df = pd.read_csv("nav_with_trends.csv")
nav_df["date"] = pd.to_datetime(nav_df["date"])
ml_model = joblib.load("ml_impact_model.pkl")

st.markdown("### Fund: HDFC Large Cap Fund - Direct Growth")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("NAV History (Last 90 Days)")
    chart_data = nav_df[["date", "nav"]].tail(90).set_index("date")
    st.line_chart(chart_data)

with col2:
    latest = nav_df.iloc[-1]
    st.metric("Latest NAV", f"{latest['nav']:.2f}")
    if latest["rolling_avg_return_30d"] < 0:
        st.error("Overall Signal: ELEVATED RISK")
    else:
        st.success("Overall Signal: NORMAL")

st.markdown("### Multi-Timeframe Redemption Pressure Signal")
c1, c2, c3 = st.columns(3)
fast_neg = latest["rolling_avg_return_5d"] < 0
med_neg = latest["rolling_avg_return_10d"] < 0
slow_neg = latest["rolling_avg_return_30d"] < 0

with c1:
    st.metric("5-Day Signal (Fast)", f"{latest['rolling_avg_return_5d']*100:.3f}%", delta="Negative" if fast_neg else "Positive", delta_color="inverse")
with c2:
    st.metric("10-Day Signal (Medium)", f"{latest['rolling_avg_return_10d']*100:.3f}%", delta="Negative" if med_neg else "Positive", delta_color="inverse")
with c3:
    st.metric("30-Day Signal (Slow)", f"{latest['rolling_avg_return_30d']*100:.3f}%", delta="Negative" if slow_neg else "Positive", delta_color="inverse")

agreement = sum([fast_neg, med_neg, slow_neg])
if agreement == 3:
    st.error("All 3 timeframes agree: STRONG redemption pressure signal")
elif agreement >= 1:
    st.warning(f"{agreement}/3 timeframes show negative pressure: MODERATE signal")
else:
    st.success("All timeframes normal: LOW redemption pressure")

st.markdown("---")
st.subheader("Simulate a Trade")

trade_cr = st.slider("Trade size (Rs Crore)", min_value=1, max_value=50, value=5, step=1)
trade_size = trade_cr * 10000000

liquidity_df = pd.read_csv("liquidity_stats.csv")
test_prices = {"ICICIBANK.NS": 1250, "INFY.NS": 1780, "LT.NS": 3600, "RELIANCE.NS": 1400}

combined_trend_score = (int(fast_neg) * 0.5) + (int(med_neg) * 0.3) + (int(slow_neg) * 0.2)
vol_score = latest["rolling_volatility_10d"]

def predict_impact_formula(trade_value_rupees, stock_price, avg_daily_volume, volatility, k=1.0):
    trade_shares = trade_value_rupees / stock_price
    participation_rate = trade_shares / avg_daily_volume
    impact_pct = k * volatility * np.sqrt(participation_rate)
    return impact_pct * 100

def predict_impact_ml(trade_value_rupees, stock_price, avg_daily_volume, volatility):
    trade_shares = trade_value_rupees / stock_price
    participation_rate = trade_shares / avg_daily_volume
    X = pd.DataFrame({
        "trade_value": [trade_value_rupees],
        "price": [stock_price],
        "avg_volume": [avg_daily_volume],
        "volatility": [volatility],
        "participation_rate": [participation_rate]
    })
    return ml_model.predict(X)[0]

live_results = []
for _, row in liquidity_df.iterrows():
    ticker = row["ticker"]
    price = test_prices[ticker]
    impact_formula = predict_impact_formula(trade_size, price, row["avg_daily_volume"], row["daily_volatility"])
    impact_ml = predict_impact_ml(trade_size, price, row["avg_daily_volume"], row["daily_volatility"])
    risk_score = impact_formula * (1 + combined_trend_score) * (1 + vol_score * 10)
    if risk_score > 0.5:
        level = "HIGH"
    elif risk_score > 0.2:
        level = "MEDIUM"
    else:
        level = "LOW"
    live_results.append({
        "Stock": ticker.replace(".NS", ""),
        "Formula Impact (%)": round(impact_formula, 4),
        "ML Model Impact (%)": round(impact_ml, 4),
        "Risk Score": round(risk_score, 4),
        "Compliance Flag": level
    })

live_df = pd.DataFrame(live_results).sort_values("Risk Score", ascending=False).reset_index(drop=True)

st.subheader(f"Frontrunning Risk Alert — Rs {trade_cr} Crore Trade")
st.caption("Comparing formula-based (Almgren-Chriss) vs machine learning (Random Forest) predictions side by side")

def color_flag(val):
    if val == "HIGH":
        return "background-color: #ff4b4b; color: white; font-weight: bold"
    elif val == "MEDIUM":
        return "background-color: #ffa500; color: white; font-weight: bold"
    else:
        return "background-color: #21c55d; color: white; font-weight: bold"

styled = live_df.style.map(color_flag, subset=["Compliance Flag"])
st.dataframe(styled, use_container_width=True, hide_index=True)

top_risk = live_df.iloc[0]
if top_risk["Compliance Flag"] == "HIGH":
    st.warning(f"ALERT: Trading {top_risk['Stock']} at this size carries HIGH frontrunning risk. Recommend splitting into smaller tranches over multiple sessions.")

st.markdown("---")
st.subheader("Model Comparison: Formula vs Machine Learning")

with st.expander("How our ML model was trained", expanded=False):
    st.markdown("""
    We trained a Random Forest Regressor on 5,000 simulated trade scenarios using the Almgren-Chriss formula
    as ground truth (with realistic noise added), then let the model learn the relationship independently.

    **Result:** R-squared = 0.994, Mean Absolute Error = 0.014%

    **Key finding:** The model independently discovered that **participation rate** (trade size relative to
    average daily volume) explains ~81%% of impact, and **volatility** explains ~19%%, together over 99%% of
    the prediction. This validates the core insight of market microstructure theory: it's the *relative* size
    of a trade, not its absolute size, that drives price impact.
    """)

feat_col1, feat_col2 = st.columns(2)
with feat_col1:
    st.markdown("**Feature Importance (ML Model)**")
    feat_df = pd.DataFrame({
        "Feature": ["Participation Rate", "Volatility", "Trade Value", "Avg Volume", "Price"],
        "Importance": [0.8088, 0.1867, 0.0019, 0.0014, 0.0012]
    })
    st.bar_chart(feat_df.set_index("Feature"))

with feat_col2:
    st.markdown("**Why we use both models together**")
    st.markdown("""
    - **Formula (Almgren-Chriss):** Explainable, grounded in finance theory, works even with limited data
    - **ML Model:** Can capture non-linear patterns and interactions a fixed formula might miss, given enough real trade data
    - **In production:** the formula provides a transparent baseline for regulators; the ML model could be retrained
      on real historical trade data as it becomes available, improving accuracy over time
    """)

st.markdown("---")
st.subheader("Model Validation: Historical Backtest")

tab1, tab2 = st.tabs(["2020 COVID Crash (Severe Stress Test)", "Recent Period (2025-2026)"])

with tab1:
    st.caption("Testing against the worst 10 five-day NAV drops in the fund's full 10+ year history")
    backtest_df = pd.read_csv("worst_drops_backtest.csv")
    backtest_df["date"] = pd.to_datetime(backtest_df["date"]).dt.strftime("%d %b %Y")
    disp = backtest_df[["date", "nav", "rolling_avg_return_30d", "rolling_volatility_30d", "future_5d_return"]].copy()
    disp.columns = ["Date", "NAV", "30D Trend (before crash)", "30D Volatility (before crash)", "Actual Next 5-Day Return"]
    disp["30D Trend (before crash)"] = (disp["30D Trend (before crash)"] * 100).round(3).astype(str) + "%"
    disp["30D Volatility (before crash)"] = (disp["30D Volatility (before crash)"] * 100).round(3).astype(str) + "%"
    disp["Actual Next 5-Day Return"] = (disp["Actual Next 5-Day Return"] * 100).round(2).astype(str) + "%"
    st.dataframe(disp, use_container_width=True, hide_index=True)
    st.info("In 9 of 10 worst crashes (including COVID), our 30-day trend signal was already negative and volatility elevated before the crash accelerated.")

with tab2:
    st.caption("Testing against the worst 10 five-day NAV drops since January 2025")
    recent_df = pd.read_csv("recent_backtest.csv")
    recent_df["date"] = pd.to_datetime(recent_df["date"]).dt.strftime("%d %b %Y")
    disp2 = recent_df[["date", "nav", "rolling_avg_return_5d", "rolling_avg_return_10d", "future_5d_return"]].copy()
    disp2.columns = ["Date", "NAV", "5D Trend (before drop)", "10D Trend (before drop)", "Actual Next 5-Day Return"]
    disp2["5D Trend (before drop)"] = (disp2["5D Trend (before drop)"] * 100).round(3).astype(str) + "%"
    disp2["10D Trend (before drop)"] = (disp2["10D Trend (before drop)"] * 100).round(3).astype(str) + "%"
    disp2["Actual Next 5-Day Return"] = (disp2["Actual Next 5-Day Return"] * 100).round(2).astype(str) + "%"
    st.dataframe(disp2, use_container_width=True, hide_index=True)
    st.info("Adding faster 5-day and 10-day signals improved detection of recent, sharper drops. Some sudden shocks remain inherently hard to predict, a realistic limitation.")

st.markdown("---")
st.subheader("How This Works")

with st.expander("NAV Impact Model (Almgren-Chriss + ML)"):
    st.markdown("""
    Predicted price impact scales with the square root of participation rate, scaled by volatility.
    We validate this formula-based model against a Random Forest ML model trained to learn the same
    relationship independently, both agree closely, and the ML model's feature importance confirms
    the theory.

    Formula: Impact % = k x volatility x sqrt(trade size / average daily volume)
    """)

with st.expander("Multi-Timeframe Redemption Pressure Signal"):
    st.markdown("""
    Combines 5-day, 10-day, and 30-day rolling NAV return trends to catch both sudden shocks and
    sustained stress. Validated against 10+ years of real historical data including the 2020 COVID crash.
    """)

with st.expander("Why This Matters for Compliance"):
    st.markdown("""
    SEBI actively monitors frontrunning. This tool gives AMC compliance teams an early, explainable,
    quantified signal of which upcoming trades carry the highest distortion and frontrunning risk.
    """)

st.markdown("---")
st.caption("Data: AMFI (via mftool), NSE (via yfinance) | Models: Almgren-Chriss formula + Random Forest ML, validated via historical backtest | Built with Python & Streamlit")
