'''
This code demonstrates the implementation of regression trees using Scikit-Learn.
It covers important parameters, training of a regression tree, and evaluation of accuracy.

Objectives:

* Train a Regression Tree
* Evaluate a Regression Tree Performance

About the Dataset:

This dataset contains information about various areas of Boston and their respective features,
which can be used to predict the median value of owner-occupied homes. The features include:

CRIM: Crime per capita
ZN: Proportion of residential land zoned for lots over 25,000 sq.ft.
INDUS: Proportion of non-retail business acres per town
CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
NOX: Nitric oxides concentration (parts per 10 million)
RM: Average number of rooms per dwelling
AGE: Proportion of owner-occupied units built prior to 1940
DIS: Weighted distances to five Boston employment centers
RAD: Index of accessibility to radial highways
TAX: Full-value property-tax rate per $10,000
PTRATIO: Pupil-teacher ratio by town
LSTAT: Percent lower status of the population

The target variable to predict is:

MEDV: Median value of owner-occupied homes in $1000s
This information can be used to create a predictive model in Python for estimating the
median housing prices in different areas of Boston.
'''

import pandas as pd # Pandas will allow us to create a dataframe of the date to use and manipulate
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# Read the Data
data = pd.read_csv('real_estate_data.csv')
print(data.head())
print(data.shape)

# Checking for invalid data, such as rows with missing values, which will be addresses during pre-processing
print(data.isna().sum())

## Data Pre-Processing
# Dropping rows with missing values because we have enough data in our dataset
data.dropna(inplace=True) # This modifies data directly, removing rows with NaN values
print(data.isna().sum()) # Checking again for missing values

# Splitting the dataset into features and target variables
X = data.drop(columns=["MEDV"])
y = data["MEDV"]

print(f'X.shape = {X.shape}, & y.shape = {y.shape}')
print(X.head())
print(y.head())

## Dataset Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# 20% of the data will be used for testing

## Creating a DecisionTreeRegressor object with the MSE criterion
'''
The important parameters of DecisionTreeRegressor are:
- criterion: {"squared_error", "friedman_mse", "absolute_error", "poisson"} - The function used to measure error
- max_depth: The maximum depth the tree can be
- min_samples_split: The minimum number of samples required to split a node
- min_samples_leaf: The minimum number of samples that a leaf can contain
- max_features: {"auto", "sqrt", "log2"} - The number of features we examine looking for the best one, used to speed up training
'''
regression_tree_mse = DecisionTreeRegressor(criterion="squared_error") # mse stands for Mean Squared Error

# Training the regression tree with the training data
regression_tree_mse.fit(X_train, y_train)

# Calculating the R^2 score for the predictions made by the MSE criterion regression tree
'''
To evaluate our model, we will use the score method of the DecisionTreeRegressor object
by providing our testing data. The resulting number is the R^2 value, which indicates
the coefficient of determination, providing insight into the goodness of fit of the model.

Additionally, we can calculate the average error in our testing set, which represents
the average error in median home value prediction.
'''
r2_score_mse = regression_tree_mse.score(X_test, y_test)
print(f'R^2 Score (MSE Criterion): {r2_score_mse: .3f}')

# Making predictions on the testing set using the trained regression tree model
prediction_mse = regression_tree_mse.predict(X_test)

# Calculating the mean absolute error (MAE) for the predictions
MAE_mse = (prediction_mse - y_test).abs().mean()*1000
print(f'Mean Absolute Error (MSE Criterion): {MAE_mse: .3f} $')

###
# Creating a new DecisionTreeRegressor object with the MAE criterion
regression_tree_mae = DecisionTreeRegressor(criterion="absolute_error")

# Training the regression tree with the training data
regression_tree_mae.fit(X_train, y_train)

# Calculating the R^2 score for the predictions made by the MAE criterion regression tree
r2_score_mae = regression_tree_mae.score(X_test, y_test)
print(f'R^2 Score (MAE Criterion): {r2_score_mae: .3f}')

# Making predictions on the testing set using the newly trained regression tree model
prediction_mae = regression_tree_mae.predict(X_test)

# Calculating the mean absolute error (MAE) for the predictions
MAE_mae = (prediction_mae - y_test).abs().mean()*1000
print(f'Mean Absolute Error (MSE Criterion): {MAE_mae: .3f} $')
