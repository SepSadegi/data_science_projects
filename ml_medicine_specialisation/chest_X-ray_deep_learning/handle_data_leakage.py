'''
This script identifies and addresses patient overlap between the
training and validation datasets.

Patient overlap is a common issue related to data leakage in machine learning,
where the same patient's data appears in both the training and validation sets.
This can lead to biased evaluation of the model's performance.

The following steps are performed:
1. Load the training and validation datasets.
2. Extract patient IDs and identify overlapping patients.
3. Remove rows from the validation set that have overlapping patient IDs.
'''

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Configure matplotlib settings
mpl.rc('xtick', labelsize=16)
mpl.rc('ytick', labelsize=16)
mpl.rcParams.update({'font.size': 15})
plt.ioff()

# Define the path to the dataset
data_directory = '/home/ssadegi/PycharmProjects/ml_medicine_specialisation/chest_X-ray_deep_learning'

# Load training and validation datasets
train_df = pd.read_csv(os.path.join(data_directory, 'train-small.csv'))
valid_df = pd.read_csv(os.path.join(data_directory, 'valid-small.csv'))

print(f'Training dataset: {train_df.shape[0]} rows, {train_df.shape[1]} columns')
print(f'Validation dataset: {valid_df.shape[0]} rows, {valid_df.shape[1]} columns')

# Extract Patient IDs from both datasets
train_patient_ids = train_df.PatientId.values
valid_patient_ids = valid_df.PatientId.values

# Convert arrays of Patient IDs to sets for easier comparison
train_patient_ids_set = set(train_patient_ids)
valid_patient_ids_set = set(valid_patient_ids)

print(f'Unique Patient IDs in training set: {len(train_patient_ids_set)}')
print(f'Unique Patient IDs in validation set: {len(valid_patient_ids_set)}')

# Identify overlapping Patient IDs between training and validation sets
overlapping_patient_ids = list(train_patient_ids_set.intersection(valid_patient_ids_set))
num_overlapping_patients = len(overlapping_patient_ids)
print(f'Number of overlapping Patient IDs: {num_overlapping_patients}')
print(f'Overlapping Patient IDs: {overlapping_patient_ids}')

# Find indices of overlapping patients in both datasets
train_overlap_indices = [index for patient_id in overlapping_patient_ids
                         for index in train_df.index[train_df['PatientId'] == patient_id].tolist()]
valid_overlap_indices = [index for patient_id in overlapping_patient_ids
                         for index in valid_df.index[valid_df['PatientId'] == patient_id].tolist()]

print(f'Indices of overlapping patients in training set: {train_overlap_indices}')
print(f'Indices of overlapping patients in validation set: {valid_overlap_indices}')

# Remove overlapping patients from the validation set
if valid_overlap_indices:
    valid_df.drop(valid_overlap_indices, inplace=True)
else:
    print("No overlapping indices found in the validation DataFrame.")

# Verify the results by checking for any remaining overlaps
new_valid_patient_ids = valid_df.PatientId.values
new_valid_patient_ids_set = set(new_valid_patient_ids)

print(f'Unique Patient IDs in the cleaned validation set: {len(new_valid_patient_ids_set)}')

new_overlapping_patient_ids = list(train_patient_ids_set.intersection(new_valid_patient_ids_set))
num_new_overlapping_patients = len(new_overlapping_patient_ids)
print(f'Number of overlapping Patient IDs in the cleaned validation set: {num_new_overlapping_patients}')


