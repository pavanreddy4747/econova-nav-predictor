from mftool import Mftool
import pandas as pd
mf = Mftool()
data = mf.get_scheme_historical_nav("119018")
df = pd.DataFrame(data["data"])
print(df.head(10))
print("Total records:", len(df))
df.to_csv("hdfc_largecap_nav.csv", index=False)
print("Saved to hdfc_largecap_nav.csv")
