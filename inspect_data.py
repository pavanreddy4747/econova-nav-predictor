import pandas as pd
import numpy as np
df = pd.read_csv("stock_prices.csv", header=[0,1], index_col=0)
print(df.columns.tolist())
