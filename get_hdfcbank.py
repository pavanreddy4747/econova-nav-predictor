import yfinance as yf
df = yf.download("HDFCBANK.NS", period="2y")
print(df.head())
df.to_csv("hdfcbank_prices.csv")
print("Saved separately")
