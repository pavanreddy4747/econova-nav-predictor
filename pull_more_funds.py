from mftool import Mftool
import pandas as pd

mf = Mftool()

funds = {
    "SBI_Large_Cap": "119598",
    "ICICI_Large_Cap": "120586"
}

for name, code in funds.items():
    data = mf.get_scheme_historical_nav(code)
    df = pd.DataFrame(data["data"])
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df["nav"] = df["nav"].astype(float)
    df = df.sort_values("date").reset_index(drop=True)

    df["daily_return"] = df["nav"].pct_change()
    df["rolling_avg_return_5d"] = df["daily_return"].rolling(5).mean()
    df["rolling_avg_return_10d"] = df["daily_return"].rolling(10).mean()
    df["rolling_avg_return_30d"] = df["daily_return"].rolling(30).mean()
    df["rolling_volatility_5d"] = df["daily_return"].rolling(5).std()
    df["rolling_volatility_10d"] = df["daily_return"].rolling(10).std()
    df["rolling_volatility_30d"] = df["daily_return"].rolling(30).std()

    filename = f"nav_{name}.csv"
    df.to_csv(filename, index=False)
    print(f"{name}: {len(df)} records saved to {filename}")
    print(df.tail(3)[["date", "nav", "rolling_avg_return_30d"]])
    print()
