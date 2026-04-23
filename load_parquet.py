import pandas as pd
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "mytest/test.parquet"

df = pd.read_parquet(path)
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nFirst row:")
print(df.iloc[0].to_dict())
