'''
Taxi Tip Prediction using Scikit-Learn and Snap ML

This code uses a real NYC taxi trip dataset to train a decision tree regression model for
predicting taxi tip amount.

The code explores two libraries:

Scikit-Learn: A popular Python library for machine learning.
Snap ML: A high-performance IBM library for machine learning, offering efficient CPU/GPU
implementations and novel algorithms.
For more on Snap ML, visit https://www.zurich.ibm.com/snapml/.

Objectives:

* Data preprocessing: Performing basic cleaning and preparation of data using Scikit-Learn.
* Regression modeling: Building regression models with both Scikit-Learn and Snap ML APIs.
* Decision Tree Regressor: Training a Decision Tree model for regression tasks using both libraries.
* Model evaluation: Running predictions and assessing the performance of the trained models.

Target variable:

The target variable is "tip_amount," representing the amount of tip paid for each taxi trip.
'''

## Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeRegressor as SklearnDTR
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import mean_squared_error

import time
import gc, sys

# Read in data
raw_data = pd.read_csv('yellow_tripdata_2019-06.csv')
print(f'There are {len(raw_data)} observations in the dataset.')
print(f'There are {len(raw_data.columns)} variables in the dataset.')

print(list(raw_data.columns))
# Display first rows in the dataset
print(raw_data.head())

# some trips report 0 tip. it is assumed that these tips were paid in cash.
# for this study we drop all these rows
raw_data = raw_data[raw_data.tip_amount > 0]

# we also remove some outliers, namely those where the tip was larger than the fare cost
raw_data = raw_data[raw_data.tip_amount <= raw_data.fare_amount]

# we remove trips with very large fare cost
raw_data = raw_data[((raw_data.fare_amount >= 2) & (raw_data.fare_amount < 200))]

# we drop variables that include the target variable in it, namely the total_amount
clean_data = raw_data.drop(columns=['total_amount'])

# release memory occupied by raw_data as we do not need it anymore
# we are dealing with a large dataset, thus we need to make sure we do not run out of memory
del raw_data
gc.collect()

# print the number of trips left in the dataset
print(f'There are {len(clean_data)} observations in the dataset.')
print(f'There are {len(clean_data.columns)} variables in the dataset.')

plt.hist(clean_data.tip_amount.values, 16, histtype= 'bar', facecolor='g')
plt.title('Histogram of tip amounts')
plt.show()

print(f'Minimum amount value is {np.min(clean_data.tip_amount.values)} $')
print(f'Maximum amount value is {np.max(clean_data.tip_amount.values)} $')
print(f'90% of the transactions have an amount less or equal  than {np.percentile(clean_data.tip_amount.values, 90)} $')

# Display first rows in the dataset
print(clean_data.head())

## Dataset Preprocessing
# This data contains various data types like dates, locations, and categorical features.
# We will need to transform these features into a suitable format for model training.
# This may involve encoding categorical features and handling date/time formats.

# Convert 'tpep_dropoff_datetime' and 'tpep_pickup_datetime' columns to datetime objects
clean_data.tpep_dropoff_datetime = pd.to_datetime(clean_data.tpep_dropoff_datetime)
clean_data.tpep_pickup_datetime = pd.to_datetime(clean_data.tpep_pickup_datetime)

# Extract pickup and dropoff hour
clean_data['pickup_hour'] = clean_data.tpep_dropoff_datetime.dt.hour
clean_data['dropoff_hour'] = clean_data.tpep_dropoff_datetime.dt.hour

# Extract pickup and dropoff day of the week (0 = Monday, 6 = Sunday)
clean_data['pickup_day'] = clean_data.tpep_pickup_datetime.dt.weekday
clean_data['dropoff_day'] = clean_data.tpep_dropoff_datetime.dt.weekday

# Calculate trip time in seconds
clean_data['trip_time'] = (clean_data.tpep_dropoff_datetime - clean_data.tpep_pickup_datetime).dt.total_seconds()
print(f'Min trip time is {np.min(clean_data.trip_time)} and max trip time is {np.max(clean_data.trip_time)} seconds')

'''
Reduce dataset size (optional)

Ideally, the entire dataset should be used for training.
However, if you encounter memory limitations, reducing the data size can be helpful.
Adjust the value of 'first_n_rows' based on the available memory resources.
'''
# This code snippet utilizes only the first 200,000 rows of the clean data.
first_n_rows = 200000
clean_data = clean_data.head(first_n_rows)

# Excluding the pickup and drop off datetime
clean_data = clean_data.drop(columns=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])

# Encode categorical features using one-hot encoding from the Pandas package
categorical_features = ["VendorID", "RatecodeID", "store_and_fwd_flag", "PULocationID", "DOLocationID", "payment_type", "pickup_hour", "dropoff_hour", "pickup_day", "dropoff_day"]
proc_data = pd.get_dummies(clean_data, columns=categorical_features)
# proc_data now contains the one-hot encoded features for these columns

# release memory occupied by raw_data as we do not need it anymore
# we are dealing with a large dataset, thus we need to make sure we do not run out of memory
del clean_data
gc.collect()

# Extract labels (target variable)
y = proc_data[['tip_amount']].values.astype('float32')

# Drop the target variable from the feature matrix
proc_data = proc_data.drop(columns=['tip_amount'])

# Extract feature matrix used for training (independent variables)
X = proc_data.values

# Normalize features (optional)
X = normalize(X, axis=1, norm='l1', copy=False) # L1 Norm (Manhattan distance)
# The L1 norm of a vector is the sum of the absolute values of its elements.

# Print the shapes of the features matrix and the labels vector
print(f'Features shape: {X.shape}, Target variable shape: {y.shape}')

## Dataset Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
'''
random_state (default=None): This argument controls the randomness used when splitting the data.  

Without random_state: If we don't set a value for random_state, the function will use a random seed 
for shuffling the data before splitting. This means that every time we run our code, we might get 
a different split of the data into training and testing sets.
 
With random_state: Setting a value for random_state (like random_state=42 in your code) ensures that 
the data is shuffled and split in a deterministic way based on that specific seed. This guarantees that 
we get the same split of data for training and testing whenever we run our code with the same random_state 
value. This reproducibility is helpful for debugging, comparing different models, or ensuring consistency 
in our experiments.
'''

print(f'X_train.shape = {X_train.shape}, & y_train.shape = {y_train.shape}')
print(f'X_test.shape = {X_test.shape}, & y_test.shape = {y_test.shape}')

# Option 1: Build a Decision Tree Regressor model with Scikit-Learn
# For reproducible output across multiple function calls, set random_state to a given integer value
sklearn_dt = SklearnDTR(max_depth=8, random_state=35)

# Train a Decision Tree Regressor model using scikit_learn
t0 = time.time()
sklearn_dt.fit(X_train, y_train)
sklearn_time = time.time() - t0
print(f'[Scikit-Learn] Training time (s): {sklearn_time:.5f}')

# Option 2: Build a Decision Tree Regressor model with Snap ML
from snapml import DecisionTreeRegressor as SnapDTR

# In contrast to sklearn's Decision Tree, Snap ML offers multi-threaded CPU/GPU training
# To use the GPU, one needs to set the use_gpu parameter to True
# snapml_dt = DecisionTreeRegressor(max_depth=4, random_state=45, use_gpu=True)

# To set the number of CPU threads used at training time, we need to set the n_jobs parameter
# Here n_jobs=4 specifies that we want to use 4 CPU threads for training
# for reproducible output across multiple function calls, set random_state to a given integer value
snapml_dt = SnapDTR(max_depth=8, random_state=45, n_jobs=4)

# Train a Decision Tree Regressor model using Snap ML
t0 = time.time()
snapml_dt.fit(X_train, y_train)
snapml_time = time.time() - t0
print(f'[Snap ML] Training time (s): {snapml_time:.5f}')

# Evaluate the Scikit-Learn and Snap ML Decision Tree Regressor Models
# Snap ML vs Scikit-Learn training speedup
training_speedup = sklearn_time/snapml_time
print(f'[Decision Tree Regressor] Snap ML vs. Scikit-Learn speedup: {training_speedup: .2f}x')

# Inference and Evaluation using Scikit-Learn
# Run inference
sklearn_pred = sklearn_dt.predict(X_test)
# Evaluate mean squared error on the test dataset
sklearn_mse = mean_squared_error(y_test, sklearn_pred)
print(f'[Scikit-Learn] MSE score : {sklearn_mse: .3f}')

# Inference and Evaluation using Snap ML
# Run inference
snapml_pred = snapml_dt.predict(X_test)
# Evaluate mean squared error on the test dataset
snapml_mse = mean_squared_error(y_test, snapml_pred)
print(f'[Snap ML] RMSE score : {snapml_mse: .3f}')

# Train a new SnapML Decision Tree Regressor with different parameters
snapml_dt_new = SnapDTR(max_depth=12, random_state=45, n_jobs=4)
snapml_dt_new.fit(X_train, y_train) # Train the new model

# Make predictions on the test set using the new model (inference)
snapml_pred_new = snapml_dt_new.predict(X_test)

# Evaluate mean squared error (MSE) on the test dataset for the new model
snapml_mse_new = mean_squared_error(y_test, snapml_pred_new)

# Print the MSE results for comparison
print("==================================================")
print(f'[Snap ML - Previous Model] MSE score: {snapml_mse: .3f}')
print(f'[Snap ML - New Model (max_depth=12)] MSE score: {snapml_mse_new: .3f}')
print("Higher MSE, potentially indicating overfitting.")
print("==================================================")
# We learned that increasing the max_depth parameter to 12 increases the MSE,
# suggesting that a simpler model might be preferable in this case.

