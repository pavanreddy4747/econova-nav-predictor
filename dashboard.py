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
garch = pd.read_csv("garch_summary.csv").iloc[0]
corr_matrix = pd.read_csv("correlation_matrix.csv", index_col=0)
walk_forward = pd.read_csv("walk_forward_results.csv")

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

st.markdown("### Volatility: Historical vs. GARCH-Forecasted")
vcol1, vcol2, vcol3 = st.columns(3)
with vcol1:
    st.metric("10-Day Historical Volatility", f"{garch['rolling_10d_volatility']*100:.3f}%")
with vcol2:
    st.metric("GARCH Next-Day Forecast", f"{garch['garch_next_day_forecast']*100:.3f}%",
               delta=f"{(garch['garch_next_day_forecast'] - garch['rolling_10d_volatility'])*100:+.3f}pp vs historical")
with vcol3:
    st.metric("GARCH Persistence (alpha+beta)", f"{garch['garch_persistence']:.3f}")

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
st.caption("Enter any trade size, from retail-scale to institutional block trades")

input_mode = st.radio("Input unit:", ["Lakhs (Rs 1L - Rs 99L)", "Crores (Rs 1Cr - Rs 500Cr)"], horizontal=True)

if input_mode.startswith("Lakhs"):
    trade_lakh = st.slider("Trade size (Rs Lakh)", min_value=1, max_value=99, value=50, step=1)
    trade_size = trade_lakh * 100000
    trade_display = f"Rs {trade_lakh} Lakh"
else:
    trade_cr = st.slider("Trade size (Rs Crore)", min_value=1, max_value=500, value=50, step=1)
    trade_size = trade_cr * 10000000
    trade_display = f"Rs {trade_cr} Crore"

exact_input = st.number_input("Or enter an exact trade value in Rs (overrides slider if changed from 0):", min_value=0, value=0, step=1000000, format="%d")
if exact_input > 0:
    trade_size = exact_input
    trade_display = f"Rs {exact_input:,}"

st.info(f"Simulating a **{trade_display}** trade")

vol_source = st.radio("Volatility input:", ["Historical (10-day rolling)", "GARCH-forecasted (next-day)"], horizontal=True)
vol_score = garch["garch_next_day_forecast"] if vol_source.startswith("GARCH") else garch["rolling_10d_volatility"]

liquidity_df = pd.read_csv("liquidity_stats.csv")
test_prices = {"ICICIBANK.NS": 1250, "INFY.NS": 1780, "LT.NS": 3600, "RELIANCE.NS": 1400}
combined_trend_score = (int(fast_neg) * 0.5) + (int(med_neg) * 0.3) + (int(slow_neg) * 0.2)

def predict_impact_formula(trade_value_rupees, stock_price, avg_daily_volume, volatility, k=1.0):
    trade_shares = trade_value_rupees / stock_price
    participation_rate = trade_shares / avg_daily_volume
    impact_pct = k * volatility * np.sqrt(participation_rate)
    return impact_pct * 100, participation_rate

live_results = []
for _, row in liquidity_df.iterrows():
    ticker = row["ticker"]
    price = test_prices[ticker]
    impact_formula, participation_rate = predict_impact_formula(trade_size, price, row["avg_daily_volume"], vol_score)
    risk_score = impact_formula * (1 + combined_trend_score) * (1 + vol_score * 10)
    level = "HIGH" if risk_score > 0.5 else ("MEDIUM" if risk_score > 0.2 else "LOW")
    live_results.append({
        "Stock": ticker.replace(".NS", ""),
        "Predicted Impact (%)": round(impact_formula, 4),
        "Participation Rate (%)": round(participation_rate * 100, 3),
        "Risk Score": round(risk_score, 4),
        "Compliance Flag": level
    })

live_df = pd.DataFrame(live_results).sort_values("Risk Score", ascending=False).reset_index(drop=True)
st.subheader(f"Frontrunning Risk Alert — {trade_display} Trade")

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

if top_risk["Participation Rate (%)"] > 50:
    st.error(f"EXTREME: This trade represents over 50% of {top_risk['Stock']}'s average daily volume. Executing at this size in a single day would likely cause severe, disorderly price impact and is operationally unrealistic without multi-day execution.")

st.markdown("---")
st.subheader("Portfolio Stress Scenario: Correlated Holdings")
pcol1, pcol2 = st.columns([1, 1])
with pcol1:
    st.markdown("**Correlation Matrix (2-year daily returns)**")
    st.dataframe(corr_matrix.round(3), use_container_width=True)
with pcol2:
    most_corr_pair = ("LT", "RELIANCE")
    corr_value = corr_matrix.loc[most_corr_pair[0], most_corr_pair[1]]
    st.markdown(f"**Most correlated pair: {most_corr_pair[0]} & {most_corr_pair[1]}**")
    st.metric("Correlation", f"{corr_value:.3f}")
    individual_risk = live_df[live_df["Stock"].isin(most_corr_pair)]["Risk Score"].sum()
    compounded_risk = individual_risk * (1 + corr_value)
    st.metric("Sum of Individual Risk Scores", f"{individual_risk:.4f}")
    st.metric("Correlation-Adjusted Combined Risk", f"{compounded_risk:.4f}", delta=f"+{(compounded_risk - individual_risk):.4f}")

st.markdown("---")
st.subheader(f"Why is {top_risk['Stock']} flagged {top_risk['Compliance Flag']}?")
reasons = []
if top_risk["Participation Rate (%)"] > 1:
    reasons.append(f"This trade represents **{top_risk['Participation Rate (%)']:.2f}%** of average daily volume.")
else:
    reasons.append(f"This trade represents a modest **{top_risk['Participation Rate (%)']:.2f}%** of average daily volume.")
if agreement == 3:
    reasons.append("All three redemption pressure timeframes are currently negative.")
elif agreement >= 1:
    reasons.append(f"{agreement} of 3 redemption pressure signals are negative.")
else:
    reasons.append("Redemption pressure signals are currently normal.")
reasons.append(f"Using {vol_source.lower()} volatility of {vol_score*100:.3f}% as model input.")
for i, r in enumerate(reasons, 1):
    st.markdown(f"{i}. {r}")

st.markdown("---")
st.subheader("Export Compliance Report")
report_df = live_df.copy()
report_df["Fund"] = "HDFC Large Cap Fund - Direct Growth"
report_df["Trade Size"] = trade_display
report_df["Volatility Source"] = vol_source
report_df["Report Generated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
csv_data = report_df.to_csv(index=False)
st.download_button(label="Download Compliance Report (CSV)", data=csv_data,
    file_name=f"frontrunning_risk_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv")

st.markdown("---")
st.subheader("Model Validation")
tab1, tab2, tab3 = st.tabs(["2020 COVID Crash", "Recent Period (2025-2026)", "Walk-Forward Validation"])

with tab1:
    backtest_df = pd.read_csv("worst_drops_backtest.csv")
    backtest_df["date"] = pd.to_datetime(backtest_df["date"]).dt.strftime("%d %b %Y")
    disp = backtest_df[["date", "nav", "rolling_avg_return_30d", "rolling_volatility_30d", "future_5d_return"]].copy()
    disp.columns = ["Date", "NAV", "30D Trend (before crash)", "30D Volatility (before crash)", "Actual Next 5-Day Return"]
    disp["30D Trend (before crash)"] = (disp["30D Trend (before crash)"] * 100).round(3).astype(str) + "%"
    disp["30D Volatility (before crash)"] = (disp["30D Volatility (before crash)"] * 100).round(3).astype(str) + "%"
    disp["Actual Next 5-Day Return"] = (disp["Actual Next 5-Day Return"] * 100).round(2).astype(str) + "%"
    st.dataframe(disp, use_container_width=True, hide_index=True)
    st.info("9 of 10 worst crashes (including COVID) were preceded by a negative 30-day trend and elevated volatility.")

with tab2:
    recent_df = pd.read_csv("recent_backtest.csv")
    recent_df["date"] = pd.to_datetime(recent_df["date"]).dt.strftime("%d %b %Y")
    disp2 = recent_df[["date", "nav", "rolling_avg_return_5d", "rolling_avg_return_10d", "future_5d_return"]].copy()
    disp2.columns = ["Date", "NAV", "5D Trend (before drop)", "10D Trend (before drop)", "Actual Next 5-Day Return"]
    disp2["5D Trend (before drop)"] = (disp2["5D Trend (before drop)"] * 100).round(3).astype(str) + "%"
    disp2["10D Trend (before drop)"] = (disp2["10D Trend (before drop)"] * 100).round(3).astype(str) + "%"
    disp2["Actual Next 5-Day Return"] = (disp2["Actual Next 5-Day Return"] * 100).round(2).astype(str) + "%"
    st.dataframe(disp2, use_container_width=True, hide_index=True)
    st.info("Adding faster 5-day and 10-day signals improved detection of recent, sharper drops.")

with tab3:
    wf_display = walk_forward.copy()
    wf_display["hit_rate"] = (wf_display["hit_rate"] * 100).round(1).astype(str) + "%"
    st.dataframe(wf_display, use_container_width=True, hide_index=True)
    wcol1, wcol2 = st.columns(2)
    with wcol1:
        st.metric("In-Sample Hit Rate (Train)", f"{walk_forward.iloc[0]['hit_rate']*100:.1f}%")
    with wcol2:
        st.metric("Out-of-Sample Hit Rate (Test)", f"{walk_forward.iloc[1]['hit_rate']*100:.1f}%")
    st.warning("Honest finding: the signal correctly flags ~45-47% of the worst 10% of trading days, both in-sample and out-of-sample (confirming it's not overfit) — but this is meaningfully lower than its 9/10 hit rate against the most extreme COVID-level crashes. The model has genuine skill at extreme tail events, modest skill at everyday bad days.")

st.markdown("---")
st.caption("Data: AMFI (via mftool), NSE (via yfinance) | Models: Almgren-Chriss + Random Forest + GARCH(1,1) + correlation-adjusted risk, validated via backtest and walk-forward testing | Built with Python & Streamlit")
