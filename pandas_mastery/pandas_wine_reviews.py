import os
import pandas as pd

# Display all columns
pd.set_option('display.max_columns', None)

# DataFrame
path = os.getcwd()
file_path = "/home/gusi-and-pilili2/PycharmProjects/dataScience/data/winemag-data-130k-v2.csv"

df = pd.read_csv(file_path, index_col=0)
print(f"Dataset Shape: {df.shape}")
print(f"Dataset Columns: {df.columns}")
print(df.info())
print("Dataset first 5 rows:\n", df.head())
print("Dataset last 5 rows:\n", df.tail())

# First row of dataframe
first_row = df.iloc[0]
print("First row:\n",first_row)

# First row of description
first_descriptions = df.description.iloc[:10]
print("First ten description:\n", first_descriptions)

# Data Samples with index label 1, 2, 3, 5, 8
sample_data = df.loc[[1, 2, 3, 5, 8], :]  # iloc and loc:  First columns, second rows
print("FSample Data:\n", sample_data)

# Create a variable df containing 'country', 'province', 'region_1', 'region_2' for i: 0, 1, 10 , 100
df_var = df.loc[[0,1,10,100], ['country', 'province', 'region_1', 'region_2']]
print("Show Case:\n", df_var)

# Create a variable df containing country and variety for the first 100 records
# The difference between loc and iloc is in indexing
df_var2 = df.loc[: 99, ['country', 'variety']]
df_var3 = df.iloc[: 100, ['country', 'variety']]
print("Show first 100 countries with their wine variety:\n", df_var2)