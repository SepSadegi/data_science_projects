'''
This script demonstrates basic Pandas operations using a medical dataset.
The dataset contains the following columns:
    sex (binary): 1 if Male, 0 otherwise
    age (int): Age of the patient at the start of the study
    obstruct (binary): Obstruction of colon by tumor (1 if obstructed, 0 otherwise)
    outcome (binary): 1 if the patient died within 5 years, 0 otherwise
    TRTMT (binary): 1 if the patient was treated, 0 otherwise
'''

import os
import pandas as pd

# Define the path to the dataset
data_directory = '../treatment_effect_estimation'

# Load the dataset, setting the 0th column as the row index
data = pd.read_csv(os.path.join(data_directory, 'dummy_data.csv'), index_col=0)
print(f'Dataset contains {data.shape[0]} rows and {data.shape[1]} columns.')
print("Columns in the dataset:", data.columns)
# print(data.head())

# Display the type of the dataset
# A DataFrame is a two-dimensional, labeled data structure with columns of potentially different data types.
print("Type of data:", type(data))

# Display the first few entries in the 'TRTMT' column
# Series are similar to lists in Python, with one important difference: each Series can only contain one type of data.
print("First few entries in 'TRTMT':", data['TRTMT'].head())
print("Type of 'TRTMT' column:", type(data.TRTMT))

# Accessing columns using different methods
dot_notation_column = data.TRTMT
loc_notation_column = data.loc[:, 'TRTMT']
bracket_notation_column = data['TRTMT']

# Verify that different methods of accessing the same column are equivalent
print("Dot notation equals .loc notation:", dot_notation_column.equals(loc_notation_column))
print("Dot notation equals bracket notation:", dot_notation_column.equals(bracket_notation_column))

# Slicing the DataFrame to filter rows
age_below_50 = data[data['age'] <= 50]
print("Patients aged 50 or below:")
print(age_below_50)

# Filtering rows based on multiple conditions
below_50_and_treated = data[(data['age'] <= 50) & (data['TRTMT'] == True)]
print("Patients aged 50 or below and treated:")
print(below_50_and_treated)

# Display type of the filtered DataFrame
print("Type of filtered DataFrame:", type(below_50_and_treated))

# More Advanced Operations
# Calculate the number of rows, shape, and size
print(f"Number of rows in 'age_below_50': {len(age_below_50)}")
print(f"Shape of 'age_below_50': {age_below_50.shape}") # tuple form (rows, cols)
print(f"Size of 'age_below_50': {age_below_50.size}")

# Calculate the number of rows, shape, and size for 'treated_patient'
treated_patient = data['TRTMT']
print(f"Number of rows in 'treated_patient': {len(treated_patient)}")
print(f"Shape of 'treated_patient': {treated_patient.shape}")
print(f"Size of 'treated_patient': {treated_patient.size}")

# Calculate and display the proportion of male patients
proportion_male_patients = len(data[data['sex'] == 1]) / data.shape[0]
print(f"Calculated proportion of male patients: {proportion_male_patients:.2f}, Expected: {21/50}")

# Calculate the proportion of male patients using the mean method
proportion_male_patients_mean = data['sex'].mean()
print(f"Proportion of male patients using mean method: {proportion_male_patients_mean:.2f}")

# Update values using .loc[row,col]
# Note: Indexing starts at 0 by default in Pandas unless explicitly set otherwise
# In this script, since the dataframe's index is defined, the first row is at index 1 and not 0
print(f"Original value at index 2 for 'TRTMT': {data.loc[2, 'TRTMT']}")
data.loc[2, 'TRTMT'] = True
print(f"Updated value at index 2 for 'TRTMT': {data.loc[2, 'TRTMT']}")

# For a specific scenario, if the study only includes females, set the 'sex' column to 0 for all patients
data.loc[:, 'sex'] = 0
print("Updated 'sex' column (set to 0 for all patients):")
print(data['sex'].head())

# Access patients by specifying a range of indices: `start:end`, where `end` is included.
print("Data for patients from index 3 to 4:")
print(data.loc[3:4, :])