"""
This script demonstrates various methods to create a DataFrame and save it to a CSV file.
"""

import pandas as pd

# Creating a DataFrame from a dictionary
data_dict = {
    'Name': ['Frodo', 'Samwise', 'Merry', 'Pippin', 'Gandalf', 'Aragorn', 'Legolas', 'Gimli', 'Boromir'],
    'Race': ['Hobbit', 'Hobbit', 'Hobbit', 'Hobbit', 'Wizard', 'Human', 'Elf', 'Dwarf', 'Human'],
    'Age': [50, 38, 36, 28, 2019, 87, 2931, 140, 41],
    'Role': ['Ring-bearer', 'Companion', 'Companion', 'Companion', 'Wizard', 'King', 'Archer', 'Warrior', 'Captain']
}

df_dict = pd.DataFrame(data_dict)
print("DataFrame created from a dictionary:\n", df_dict)

# Creating a DataFrame from a list of lists
data_list = [
    ['Frodo', 'Hobbit', 50, 'Ring-bearer'],
    ['Samwise', 'Hobbit', 38, 'Companion'],
    ['Merry', 'Hobbit', 36, 'Companion'],
    ['Pippin', 'Hobbit', 28, 'Companion'],
    ['Gandalf', 'Wizard', 2019, 'Wizard'],
    ['Aragorn', 'Human', 87, 'King'],
    ['Legolas', 'Elf', 2931, 'Archer'],
    ['Gimli', 'Dwarf', 140, 'Warrior'],
    ['Boromir', 'Human', 41, 'Captain']
]

df_list = pd.DataFrame(data_list, columns=['Name', 'Race', 'Age', 'Role'])
print("\nDataFrame created from a list of lists:\n", df_list)

# Creating a DataFrame from a list of dictionaries
data_list_dicts = [
    {'Name': 'Frodo', 'Race': 'Hobbit', 'Age': 50, 'Role': 'Ring-bearer'},
    {'Name': 'Samwise', 'Race': 'Hobbit', 'Age': 38, 'Role': 'Companion'},
    {'Name': 'Merry', 'Race': 'Hobbit', 'Age': 36, 'Role': 'Companion'},
    {'Name': 'Pippin', 'Race': 'Hobbit', 'Age': 28, 'Role': 'Companion'},
    {'Name': 'Gandalf', 'Race': 'Wizard', 'Age': 2019, 'Role': 'Wizard'},
    {'Name': 'Aragorn', 'Race': 'Human', 'Age': 87, 'Role': 'King'},
    {'Name': 'Legolas', 'Race': 'Elf', 'Age': 2931, 'Role': 'Archer'},
    {'Name': 'Gimli', 'Race': 'Dwarf', 'Age': 140, 'Role': 'Warrior'},
    {'Name': 'Boromir', 'Race': 'Human', 'Age': 41, 'Role': 'Captain'},
]

df_list_dicts = pd.DataFrame(data_list_dicts)
print("\nDataFrame created from a list of dictionaries:\n", df_list_dicts)

# Creating a DataFrame from another DataFrame (copying)
df_copy = df_dict.copy()
print("\nDataFrame created by copying another DataFrame:\n", df_copy)

# Creating a DataFrame using from_records
data_record = [
    ('Frodo', 'Hobbit', 50, 'Ring-bearer'),
    ('Samwise', 'Hobbit', 38, 'Companion'),
    ('Merry', 'Hobbit', 36, 'Companion'),
    ('Pippin', 'Hobbit', 28, 'Companion'),
    ('Gandalf', 'Wizard', 2019, 'Wizard'),
    ('Aragorn', 'Human', 87, 'King'),
    ('Legolas', 'Elf', 2931, 'Archer'),
    ('Gimli', 'Dwarf', 140, 'Warrior'),
    ('Boromir', 'Human', 41, 'Captain')
]

df_record = pd.DataFrame.from_records(data_record, columns=['Name', 'Race', 'Age', 'Role'])
print("\nDataFrame created using pd.DataFrame.from_records():\n", df_record)

# Saving the DataFrame to a CSV file
df_dict.to_csv('lotr_characters.csv', index=False)
print("\nDataFrame saved to 'lotr_characters.csv'.")