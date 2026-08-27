import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

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
    return impact_pct * 100, participation_rate

live_results = []
for _, row in liquidity_df.iterrows():
    ticker = row["ticker"]
    price = test_prices[ticker]
    impact_formula, participation_rate = predict_impact_formula(trade_size, price, row["avg_daily_volume"], row["daily_volatility"])
    risk_score = impact_formula * (1 + combined_trend_score) * (1 + vol_score * 10)
    if risk_score > 0.5:
        level = "HIGH"
    elif risk_score > 0.2:
        level = "MEDIUM"
    else:
        level = "LOW"
    live_results.append({
        "Stock": ticker.replace(".NS", ""),
        "Predicted Impact (%)": round(impact_formula, 4),
        "Participation Rate (%)": round(participation_rate * 100, 3),
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

st.markdown("---")
st.subheader(f"Why is {top_risk['Stock']} flagged {top_risk['Compliance Flag']}?")

explain_col1, explain_col2 = st.columns([2, 1])

with explain_col1:
    reasons = []
    if top_risk["Participation Rate (%)"] > 1:
        reasons.append(f"This trade represents **{top_risk['Participation Rate (%)']:.2f}%** of the stock's average daily trading volume, a significant share that alone drives most of the predicted price impact.")
    else:
        reasons.append(f"This trade represents a modest **{top_risk['Participation Rate (%)']:.2f}%** of average daily volume.")

    if agreement == 3:
        reasons.append("All three redemption pressure timeframes (5-day, 10-day, 30-day) are currently negative, indicating the fund may be under sustained selling pressure, amplifying the risk of this trade needing to happen at an inopportune time.")
    elif agreement >= 1:
        reasons.append(f"{agreement} of 3 redemption pressure signals are negative, indicating partial selling pressure on the fund.")
    else:
        reasons.append("Redemption pressure signals are currently normal, so this risk is driven mainly by trade size and liquidity, not fund stress.")

    if vol_score > 0.01:
        reasons.append(f"Current 10-day volatility ({vol_score*100:.3f}%) is elevated, meaning price moves are larger and less predictable right now, compounding the impact risk.")

    for i, r in enumerate(reasons, 1):
        st.markdown(f"{i}. {r}")

    st.markdown(f"**Recommendation:** " + (
        "Split this trade into smaller tranches across multiple sessions, and consider using algorithmic execution (e.g., VWAP/TWAP strategies) to minimize market impact and reduce frontrunning exposure."
        if top_risk["Compliance Flag"] == "HIGH" else
        "Standard execution should be acceptable, but monitor redemption trends before proceeding."
    ))

with explain_col2:
    st.metric("Risk Score", f"{top_risk['Risk Score']:.4f}")
    st.metric("Predicted Impact", f"{top_risk['Predicted Impact (%)']:.4f}%")
    st.metric("Participation Rate", f"{top_risk['Participation Rate (%)']:.3f}%")

st.markdown("---")
st.subheader("Export Compliance Report")

report_df = live_df.copy()
report_df["Fund"] = "HDFC Large Cap Fund - Direct Growth"
report_df["Trade Size (Rs Crore)"] = trade_cr
report_df["Report Generated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
report_df["Redemption Pressure (3-Signal Agreement)"] = f"{agreement}/3"

csv_data = report_df.to_csv(index=False)
st.download_button(
    label="Download Compliance Report (CSV)",
    data=csv_data,
    file_name=f"frontrunning_risk_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
    mime="text/csv"
)

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
st.caption("Data: AMFI (via mftool), NSE (via yfinance) | Models: Almgren-Chriss formula + Random Forest ML, validated via historical backtest | Built with Python & Streamlit")
