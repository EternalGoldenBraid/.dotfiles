import pandas as pd
import os


data_path = "data/lappi_data.csv"
print("Cleaning data")
df = pd.read_csv("data/vaalit_2019.csv")

# Remove rows with empty strings
lappi_column_names = [col for col in df.columns if col[:6] == 'Lappi.']
df = df.dropna(subset=lappi_column_names)
# df = df[df.apply(lambda x: x.str.len().gt(0).all(), axis=1)]

# df.to_csv(data_path, index=False)
df.to_csv(data_path)
    
print(df)