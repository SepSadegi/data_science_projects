"""
Credit Card Fraud Detection using Scikit-Learn and Snap ML

This Python code focuses on utilizing machine learning (ML) techniques, particularly
Decision Tree and Support Vector Machine (SVM) models, to detect fraudulent credit card
transactions. It leverages a real dataset containing credit card transaction information
from September 2013 by European cardholders. The primary goal is to train these models to
classify transactions as either legitimate or fraudulent.

Additionally, this exercise introduces the Snap Machine Learning (Snap ML) library, an
IBM tool designed for high-performance ML modeling. Snap ML offers efficient CPU/GPU
implementations for linear and tree-based models, enhancing algorithm speed and accuracy.
Through system awareness and advanced algorithms, Snap ML provides accelerated ML capabilities.
For further details, users are encouraged to visit the Snap ML information page.

Objectives:

* Perform basic data preprocessing in Python
* Model a classification task using the Scikit-Learn and Snap ML Python APIs
* Train Suppport Vector Machine and Decision Tree models using Scikit-Learn and Snap ML
* Run inference and assess the quality of the trained models

Problem Statement:

As a member of a financial institution, the task is to build a model for predicting whether a
credit card transaction is fraudulent or not. This is treated as a binary classification problem,
where a transaction is labelled as fraudulent (1) if it is a fraud, and non-fraudulent (0) otherwise.

Dataset Information:
The dataset contains transactions over a certain period, with the majority being legitimate and
only a small fraction being fraudulent. It is highly unbalanced, with only 492 out of 284,807
transactions being fraudulent (0.172%).

Approach:
1. Load the dataset.
2. Preprocess the data if required.
3. Split the dataset into training and testing sets.
4. Train a binary classification model using machine learning algorithms.
5. Evaluate the model's performance using appropriate metrics.
6. Adjust the model or experiment with different algorithms as needed.

"""

## Import libraries
from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.tree import DecisionTreeClassifier as SklearnDTC
from sklearn.metrics import roc_auc_score

import time


## Read the input data
raw_data = pd.read_csv('creditcard.csv')
print(f'There are {len(raw_data)} observations in the credit card fraud dataset.')
print(f'There are {len(raw_data.columns)} variables in the dataset.')

## Display the first rows in the dataset
print(raw_data.head())

## Data inflation
n_replicas = 10

## inflate the original dataset
big_raw_data = pd.DataFrame(np.repeat(raw_data.values, n_replicas, axis=0), columns=raw_data.columns)
print(f'There are {len(big_raw_data)} observations in the inflated credit card fraud dataset.')
print(f'There are {len(big_raw_data.columns)} variables in the dataset.')

# Display the first rows in the dataset
print(big_raw_data.head())

# Each row in the dataset represents a credit card transaction.
# There are 31 variables in each row. One variable, 'Class', represents the target variable.
# The goal is to train a model to predict the 'Class' variable using the other variables.

# Note: Feature names are anonymized as V1, V2, ..., V28 for confidentiality.
# These features are numerical values resulting from a PCA transformation.
# The 'Class' feature is the target variable, taking values 1 for fraud and 0 otherwise.

# Get the set of distinct classes
labels = big_raw_data.Class.unique()

# Get the count of each class
sizes = big_raw_data.Class.value_counts().values

# Plot the class value counts
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.3f%%')
ax.set_title('Target Variable Value Counts')
plt.show()

# The 'Class' variable has two values: 0 (legitimate transaction) and 1 (fraudulent transaction),
# making this a binary classification problem.

# Note: The dataset is highly unbalanced, with unequal representation of target variable classes.
# Special attention is required when training or evaluating the model.
# One approach to handle this is by biasing the model to pay more attention to minority class samples.
# The models in this study will be configured to consider class weights of the samples during training.

# Plotting histogram of transaction amounts
plt.hist(big_raw_data.Amount.values, 6, histtype= 'bar', facecolor='g')
plt.title('Histogram of transaction amounts')
plt.show()

print("Minimum amount value is ", np.min(big_raw_data.Amount.values))
print("Maximum amount value is ", np.max(big_raw_data.Amount.values))
print("90% of the transactions have an amount less or equal  than", np.percentile(big_raw_data.Amount.values, 90))

## Dataset Preprocessing
# data preprocessing such as scaling/normalization is typically useful for
# linear models to accelerate the training convergence

# Standardize features by removing the mean and scaling to unit variance
# This line selects a subset of the DataFrame (all rows, and columns from index 1 to index 30)
big_raw_data.iloc[:, 1:30] = StandardScaler().fit_transform(big_raw_data.iloc[:, 1:30])
data_matrix = big_raw_data.values
'''
The StandardScaler class from scikit-learn standardize the selected subset of the data. 
The fit_transform method is applied to this subset, which fits the scaler to the data 
and then transforms it. Standardization involves scaling the features so that they have 
a mean of 0 and a standard deviation of 1. 
This is a common preprocessing step in machine learning to ensure that features are on 
a similar scale, which can improve the performance of certain algorithms.
'''

# X: feature matrix (for this analysis, we exclude the Time variable from the dataset)
X = data_matrix[:, 1:30]

# y: label vector
y = data_matrix[:, 30]

## Data normalization
# The L1 norm, also known as the Manhattan norm or taxicab norm,
# is a way of measuring the size of a vector in a vector space.
X = normalize(X, norm="l1") # L1 Norm (Manhattan distance)
# The L1 norm of a vector is the sum of the absolute values of its elements.

# Print the shape of the features matrix and the labels vector
print(f'X.shape = {X.shape}, & y.shape = {y.shape}')

## Dataset Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
'''
* test_size=0.3: This parameter specifies the proportion of the dataset to include in 
the test split. Here, it indicates that 30% of the data will be used for testing, while 
the remaining 70% will be used for training.

* random_state=42: This parameter sets the random seed used by the random number generator 
during the splitting process. Setting a random seed ensures reproducibility, meaning that 
running the code multiple times with the same seed will produce the same results. In this 
case, random_state=42 ensures that the data is split in a consistent manner each time the code is executed.

* stratify=y: This parameter ensures that the class distribution in the original dataset is 
preserved in the train-test split. When stratify=y is specified, the train_test_split function 
will ensure that the proportion of samples in each class of the target variable y is the same 
in both the training and testing subsets as it is in the original dataset. This is particularly 
useful for imbalanced datasets where one class may be significantly underrepresented. 
By stratifying the split based on the target variable, it helps ensure that both the training 
and testing sets are representative of the overall class distribution, leading to more reliable model evaluation.
'''

print(f'X_train.shape = {X_train.shape}, & y_train.shape = {y_train.shape}')
print(f'X_test.shape = {X_test.shape}, & y_test.shape = {y_test.shape}')

## Build a Decision Tree Classifier model with Scikit-Learn
# Compute the sample weights to be used as input to the train routine so that
# it takes it into account the class imbalance presents in this dataset
w_train = compute_sample_weight('balanced', y_train)

# For reproducible output across multiple function calls, set random_state to a given integer value
sklearn_dt = SklearnDTC(max_depth=4, random_state=35)

# Train a Decision Tree Classifier using scikit_learn
t0 = time.time()
sklearn_dt.fit(X_train, y_train, sample_weight=w_train)
sklearn_time = time.time() - t0
print(f'[Scikit-Learn] Training time (s): {sklearn_time:.5f}')

## Build a Decision Tree Classifier model with Snap ML
from snapml import DecisionTreeClassifier as SnapDTC
# Snap ML offers multi-threaded CPU/GPU training of decision trees, unlike scikit-learn
# to use the GPU, set the use_gpu parameter to True
# snapml_dt = SnapDTC(max_depth=4, random_state=45, use_gpu=True)

# to set the number of CPU threads used at training time, set the n_jobs parameter
# for reproducible output across multiple function calls, set random_state to a given integer value
snapml_dt = SnapDTC(max_depth=4, random_state=35, n_jobs=4)

# Train a Decision Tree Classifier using Snap ML
t0 = time.time()
snapml_dt.fit(X_train, y_train, sample_weight=w_train)
snapml_time = time.time() - t0
print(f'[Snap ML] Training time (s): {snapml_time:.5f}')

## Evaluate the Scikit-Learn and Snap ML Decision Tree Clssifier Models
# Snap ML vs Scikit-Learn training speedup
training_speedup = sklearn_time/snapml_time
print(f'[Decision Tree Classifier] Snap ML vs. Scikit-Learn speedup: {training_speedup: .2f}x')

# Run inference and compute the probabilities of the test sample to belong
# to the class of fraudulent transactions
sklearn_pred = sklearn_dt.predict_proba(X_test)[:,1]

# Evaluate the Compute Area Under the Receiver Operating Characteristic Curve (ROC-AUC)
# score from the predictions
sklearn_roc_auc = roc_auc_score(y_test, sklearn_pred)
print(f'[Scikit-Learn] ROC_AUC score : {sklearn_roc_auc: .3f}')

# Run inference and compute the probabilities of the test sample to belong
# to the class of fraudulent transactions
snapml_pred = snapml_dt.predict_proba(X_test)[:,1]

# Evaluate the Compute Area Under the Receiver Operating Characteristic Curve (ROC-AUC)
# score from the predictions
snapml_roc_auc = roc_auc_score(y_test, snapml_pred)
print(f'[Snap ML] ROC_AUC score : {snapml_roc_auc: .3f}')

# Both decision tree models yield the same score on the test dataset, as observed earlier.
# However, Snap ML completes the training routine 12 times faster than Scikit-Learn.
# Additional examples and demonstrations of Snap ML's capabilities: https://github.com/IBM/snapml-examples

## Build a Support Vector Machine model with Scikit-Learn
from sklearn.svm import LinearSVC

# Instantiate a scikit-lean SVM model
# to indicate the class imbalance at fit time, set class_weight='balanced'
# for reproducile output across multiple function calls, set random_state to a given integer value
sklearn_svm = LinearSVC(class_weight='balanced', random_state=31, loss="hinge", fit_intercept=False)

# Train a linear Support Vector Machine model using Scikit-Learn
t0 = time.time()
sklearn_svm.fit(X_train, y_train)
sklearn_time =  time.time() - t0
print(f'[Scikit-Learn] Training time (s): {sklearn_time:.2f}')

## Build a Support Vector Machine model with Snap ML
from snapml import SupportVectorMachine

# In contrast to scikit-learn's LinearSVC, Snap ML offers multi-threaded CPU/GPU training of SVMs
# to use the GPU, set the use_gpu parameter to True
# snapml_svm = SupportVectorMachine(class_weight='balanced', random_state=25, use_gpu=True, fit_intercept=False)

# to set the number of threads used at training time, one needs to set the n_jobs parameter
snapml_svm = SupportVectorMachine(class_weight='balanced', random_state=25, n_jobs=4, fit_intercept=False)
print(snapml_svm.get_params())

# Train a linear Support Vector Machine model using Snap ML
t0 = time.time()
snapml_svm.fit(X_train, y_train)
snapml_time = time.time() - t0
print(f'[Snap ML] Training time (s): {snapml_time:.2f}')

## Evaluate the Scikit-Learn and Snap ML Support Vector Machine Models
# Compute the Snap ML vs Scikit-Learn training speedup
training_speedup = sklearn_time/snapml_time
print(f'[Support Vector Machine] Snap ML vs. Scikit-Learn speedup: {training_speedup: .2f}x')

# Run inference using the Scikit-Learn model
# Get the confidence scores for the test samples
sklearn_pred = sklearn_svm.decision_function(X_test)

# Evaluate accuracy on test set
acc_sklearn = roc_auc_score(y_test, sklearn_pred)
print(f'[Scikit-Learn] ROC_AUC score : {acc_sklearn: .3f}')

# Run inference using the Snap ML model
# Get the confidence scores for the test samples
snapml_pred = snapml_svm.decision_function(X_test)

# Evaluate accuracy on test set
acc_snapml = roc_auc_score(y_test, snapml_pred)
print(f'[Snap ML] ROC_AUC score : {acc_snapml: .3f}')

# In this section, the quality of the SVM models trained earlier will be evaluated using the hinge loss metric.
# The hinge loss metric is a measure of the SVM model's performance.

# To compute the hinge loss, inference will be performed on the test set using both Scikit-Learn and Snap ML models.
# The predictions from both sets will then be used to calculate the hinge loss metric.
# (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.hinge_loss.html)

# Compute and print the hinge losses of both Scikit-Learn and Snap ML models.
from sklearn.metrics import hinge_loss

# Evaluate the hinge loss from the prediction
loss_sklearn = hinge_loss(y_test, sklearn_pred)
print(f'[Scikit-Learn] Hinge Loss : {loss_sklearn: .3f}')

loss_snapml = hinge_loss(y_test, snapml_pred)
print(f'[Snap ML] Hinge Loss : {loss_snapml: .3f}')
