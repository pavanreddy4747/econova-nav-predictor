import yfinance as yf
import pandas as pd
tickers = ["HDFCBANK.NS", "ICICIBANK.NS", "RELIANCE.NS", "INFY.NS", "LT.NS"]
data = yf.download(tickers, period="2y")
print(data.head())
data.to_csv("stock_prices.csv")
print("Saved to stock_prices.csv")
