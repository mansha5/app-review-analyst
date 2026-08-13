import pandas as pd
import os

raw_dir = "data/raw"

for file in sorted(os.listdir(raw_dir)):
    if file.endswith(".csv"):
        df = pd.read_csv(f"{raw_dir}/{file}")
        print(f"{file:<25} {len(df):>5} reviews | columns: {list(df.columns)}")