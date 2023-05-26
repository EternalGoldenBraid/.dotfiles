import pandas as pd
import os


# Store a subset of data in a new csv file
data_path = "data/lappi_data.csv"

df = pd.read_csv("data/vaalit_2019.csv")
# Remove rows with empty strings
lappi_column_names = [col for col in df.columns if col[:6] == 'Lappi.']
df = df.dropna(subset=lappi_column_names)

df.to_csv(data_path)
print(df)