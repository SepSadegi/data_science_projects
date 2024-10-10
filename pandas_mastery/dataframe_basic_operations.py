"""
This script demonstrates basic operations on a pandas DataFrame, including:
1. Inserting new columns
2. Updating existing values
3. Removing columns
4. Renaming columns
5. Modifying column names directly
"""

import pandas as pd

# Reading the DataFrame from the CSV file
df = pd.read_csv('lotr_characters.csv')

# Display the original DataFrame
print("Original DataFrame:\n", df)

# 1. Inserting a new column 'Home'
df['Home'] = ['Shire', 'Shire', 'Shire', 'Shire', 'Middle-earth', 'Rivendell', 'Mirkwood', 'Moria', 'Gondor']
print("\nDataFrame after inserting 'Home' column:\n", df)

# 2. Updating Values
# Let's update Frodo's age to 51 where Name is 'Frodo'
df.loc[df['Name'] == 'Frodo', 'Age'] = 51
print("\nDataFrame after updating Frodo's Age to 51:\n", df)

# 3. Removing a column ('Home')
df = df.drop(columns=['Home'])
print("\nDataFrame after removing 'Home' column:\n", df)

# 4. Renaming Columns
# Renaming 'Role' to 'Occupation'
df = df.rename(columns={'Role': 'Occupation'})
print("\nDataFrame after renaming 'Role' to 'Occupation':\n", df)

# 5. Modifying the column names directly
df.columns = ['Character Name', 'Species', 'Years', 'Occupation']
print("\nDataFrame after modifying column names directly:\n", df)

# Optional: Displaying a summary of the DataFrame structure after operations
print("\nSummary of the DataFrame:\n")
df.info()
