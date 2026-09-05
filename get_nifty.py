import yfinance as yf
import pandas as pd
import numpy as np

nifty = yf.download("^NSEI", period="2y")
nifty.to_csv("nifty_prices.csv")
print("Nifty data saved")
print(nifty.tail())
