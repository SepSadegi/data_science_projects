"""
This script demonstrates various methods for inspecting and indexing a DataFrame after
reading it from a CSV file.
"""

import pandas as pd

# Reading the DataFrame from the CSV file
df = pd.read_csv('lotr_characters.csv')

# Inspection
print("# Inspection Methods")

# Get a brief summary of the DataFrame
print("\ndf.info() output:")
df.info()  # Note: info() prints directly and doesn't need 'print()'

# Get the dimensions of the DataFrame
print("\ndf.shape output (Rows, Columns):", df.shape)

# Get the list of column labels
print("\ndf.columns output (Column Labels):", df.columns)

# Get the row and column axis labels (df.index for rows, df.columns for columns)
print("\ndf.axes output (Row and Column axis labels):", df.axes)

# Get a summary of statistics for numerical columns
print("\ndf.describe() output (Summary statistics):")
print(df.describe())

# View the first 5 rows of the DataFrame
print("\ndf.head() output (First 5 rows):")
print(df.head())

# View the last 5 rows of the DataFrame
print("\ndf.tail() output (Last 5 rows):")
print(df.tail())

# Get the number of elements in the DataFrame
print("\ndf.size output (Total elements):", df.size)

# Check for missing values
print("\nMissing values in DataFrame (True = missing):")
print(df.isnull().sum())

# Indexing
print("\n# Indexing Methods")

# Select the 'Name' column
names = df['Name']
print("\nColumn 'Name':\n", names)

# Select the first row using iloc (position-based indexing)
first_row = df.iloc[0]
print("\nFirst row using iloc (position-based):\n", first_row)

# Select the row where the Name is 'Gandalf' (label-based indexing)
gandalf_row = df.loc[df['Name'] == 'Gandalf']
print("\nRow where Name is 'Gandalf' (label-based indexing):\n", gandalf_row)

# Select rows where Age is greater than 100 (Boolean indexing)
age_filter = df[df['Age'] > 100]
print("\nRows where Age is greater than 100 (Boolean indexing):\n", age_filter)

# Set the 'Name' column as the index
df.set_index('Name', inplace=True)
print("\nDataFrame with 'Name' set as the index:\n", df)

# Reset the index to default
df.reset_index(inplace=True)
print("\nDataFrame after resetting the index:\n", df)

# Slicing: Select specific rows (2 to 4) and columns ('Name', 'Race')
sliced_data = df.loc[2:4, ['Name', 'Race']]
print("\nSliced Data (Rows 2 to 4, Columns 'Name' and 'Race'):\n", sliced_data)
