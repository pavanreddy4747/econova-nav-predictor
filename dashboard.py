import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="NAV Impact & Frontrunning Risk Predictor", layout="wide")

st.title("Mutual Fund NAV Impact & Frontrunning Risk Predictor")
st.caption("ECONOVA C4004AT | Predicting NAV-distorting trades before they happen, for AMC compliance teams")

nav_df = pd.read_csv("nav_with_trends.csv")
nav_df["date"] = pd.to_datetime(nav_df["date"])

st.markdown("### Fund: HDFC Large Cap Fund - Direct Growth")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("NAV History (Last 90 Days)")
    chart_data = nav_df[["date", "nav"]].tail(90).set_index("date")
    st.line_chart(chart_data)

with col2:
    latest = nav_df.iloc[-1]
    st.metric("Latest NAV", f"{latest['nav']:.2f}")
    st.metric("30-Day Avg Return", f"{latest['rolling_avg_return_30d']*100:.3f}%")
    st.metric("30-Day Volatility", f"{latest['rolling_volatility_30d']*100:.3f}%")
    if latest["rolling_avg_return_30d"] < 0:
        st.error("Redemption Pressure Signal: ELEVATED")
    else:
        st.success("Redemption Pressure Signal: NORMAL")

st.markdown("---")
st.subheader("Simulate a Trade")
st.caption("Adjust the trade size to see live-updated frontrunning risk across the fund's holdings")

trade_cr = st.slider("Trade size (Rs Crore)", min_value=1, max_value=50, value=5, step=1)
trade_size = trade_cr * 10000000

liquidity_df = pd.read_csv("liquidity_stats.csv")
test_prices = {"ICICIBANK.NS": 1250, "INFY.NS": 1780, "LT.NS": 3600, "RELIANCE.NS": 1400}
trend_score = 1 if latest["rolling_avg_return_30d"] < 0 else 0
vol_score = latest["rolling_volatility_30d"]

def predict_impact(trade_value_rupees, stock_price, avg_daily_volume, volatility, k=1.0):
    trade_shares = trade_value_rupees / stock_price
    participation_rate = trade_shares / avg_daily_volume
    impact_pct = k * volatility * np.sqrt(participation_rate)
    return impact_pct * 100

live_results = []
for _, row in liquidity_df.iterrows():
    ticker = row["ticker"]
    price = test_prices[ticker]
    impact = predict_impact(trade_size, price, row["avg_daily_volume"], row["daily_volatility"])
    risk_score = impact * (1 + trend_score * 0.5) * (1 + vol_score * 10)
    if risk_score > 0.5:
        level = "HIGH"
    elif risk_score > 0.2:
        level = "MEDIUM"
    else:
        level = "LOW"
    live_results.append({
        "Stock": ticker.replace(".NS", ""),
        "Predicted NAV Impact (%)": round(impact, 4),
        "Redemption Pressure": "Elevated" if trend_score == 1 else "Normal",
        "Risk Score": round(risk_score, 4),
        "Compliance Flag": level
    })

live_df = pd.DataFrame(live_results).sort_values("Risk Score", ascending=False).reset_index(drop=True)

st.subheader(f"Frontrunning Risk Alert — Rs {trade_cr} Crore Trade")

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
st.subheader("How This Works")

with st.expander("NAV Impact Model (Almgren-Chriss)"):
    st.markdown("""
    Predicted price impact scales with the square root of participation rate (trade size relative to average daily volume),
    scaled by the stock's historical volatility. This is a widely used institutional market-impact model,
    not a black-box prediction, every number here is explainable to a compliance officer or regulator.

    Formula: Impact % = k x volatility x sqrt(trade size / average daily volume)
    """)

with st.expander("Redemption Pressure Signal"):
    st.markdown("""
    Tracks 30-day rolling NAV returns and volatility. A sustained negative return trend combined with rising volatility
    signals the fund may be under redemption pressure, meaning it may need forced selling soon.
    """)

with st.expander("Why This Matters for Compliance"):
    st.markdown("""
    SEBI actively monitors frontrunning, where a party trades ahead of a known large institutional order.
    This tool gives AMC compliance teams an early, explainable, quantified signal of which upcoming trades
    carry the highest distortion and frontrunning risk, so they can adjust execution strategy before risk materializes.
    """)

st.markdown("---")
st.caption("Data: AMFI (via mftool), NSE (via yfinance) | Model: Almgren-Chriss market impact + custom redemption trend signal | Built with Python & Streamlit")
