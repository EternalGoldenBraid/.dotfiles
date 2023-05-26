import pandas as pd
import os


# Store a subset of data in a new csv file
data_path = "data/lappi_data.csv"

df = pd.read_csv("data/vaalit_2019.csv")
# Remove rows with empty strings
lappi_column_names = [col for col in df.columns if col[:6] == 'Lappi.']
df = df.dropna(subset=lappi_column_names)
lappi_column_names = [col for col in self.data.columns if col[:6] == 'Lappi.']


# Rename the columns Ln_1, Ln_2, ... , Ln_5, Lt_1, Lt_2, ... , Lt_5
# Data comprises of 5 columns of numbers and 5 columns of text
self.data = self.data[lappi_column_names]
self.data.columns = [f"Ln_{i}" for i in range(1, 6)] + [f"Lt_{i}" for i in range(1, 6)]

df.to_csv(data_path)
print(df)

# Create a new dataframe where each row contains
# (question_id, question, numerical answer to the question, comment to the question) in the form
# (L_i, column_name, numerical answer, comment)
df_new = pd.DataFrame(columns=["question_id", "question", "label", "comment"])
for i in range(1, 6):
    df_new = df_new.append(
        df[[f"Ln_{i}", f"Lt_{i}"]].rename(
            columns={f"Ln_{i}": "question_id", f"Lt_{i}": "comment"}), ignore_index=True)
        
print(df_new)

# # Drop rows where no numerical answer
# df_new = df_new[df_new[df_new.columns[2:]] != "-"]
# df_new.iloc[:, 2:] = df_new.iloc[:, 2:].replace("-", "Tyhjä")
# df_new = df_new[df_new.apply(lambda x: x.str.len().gt(0).all(), axis=1)]

