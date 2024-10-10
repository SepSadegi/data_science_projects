# Objectives:
# Use scikit-learn to implement simple Linear Regression
# Create a model, train it, test it and use the model

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# Load dataset
df = pd.read_csv("FuelConsumptionCo2.csv")

# Set display options to show all columns
pd.set_option('display.max.columns', None)

# Display the first few rows of the DataFrame
print(df.head())

# Display all the column headers
print(df.columns)

# Display summary statistics of the data
print(df.describe())

# Select some features to explore more
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']] # Select specific columns using double brackets
print(cdf.head(9)) # Display the first 9 rows

# Plotting histograms for selected features
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
features = ['CYLINDERS', 'ENGINESIZE', 'CO2EMISSIONS', 'FUELCONSUMPTION_COMB']
colors = ['yellowgreen', 'crimson', 'grey', 'blue']
# Plot histograms
for i, feature in enumerate(features):
    row, col = divmod(i, 2)
    sns.histplot(cdf[feature], bins=10, ax=axes[row, col], color=colors[i], kde=True)
    axes[row, col].set_title(f'Distribution of {feature}')
    axes[row, col].set_xlabel(feature)
    axes[row, col].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('feature_histograms.png', format='png', dpi=300)
plt.show()


# Create scatter plots for all selected features against CO2EMISSIONS
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# Plotting scatter plots for selected features vs the Emission, to see how linear their relationship is
sns.scatterplot(x='ENGINESIZE', y='CO2EMISSIONS', data=cdf, ax=axes[0], color='crimson')
axes[0].set_title('Engine Size vs CO2 Emissions')

sns.scatterplot(x='CYLINDERS', y='CO2EMISSIONS', data=cdf, ax=axes[1], color='yellowgreen')
axes[1].set_title('Cylinders vs CO2 Emissions')

sns.scatterplot(x='FUELCONSUMPTION_COMB', y='CO2EMISSIONS', data=cdf, ax=axes[2], color='blue')
axes[2].set_title('Fuel Consumption vs CO2 Emissions')

plt.tight_layout()
plt.savefig('scatter_comparison.png', format='png', dpi=300)
plt.show()

# Train/test split
# 80% of the entire dataset will be used for training and 20% for testing
# We create a mask to select random rows using np.random.rand() function
msk = np.random.rand(len(df)) < 0.8  # creates boolean array (probability of 0.8 being True and 0.2 being False)
train = cdf[msk]
test = cdf[~msk]  # ~msk return a new boolean array where each 'True' value in msk becomes 'False' and vice versa

# Training model with ENGINESIZE
regr = linear_model.LinearRegression()
# Convert train_x and train_y to 2D arrays with one column
train_x = np.asanyarray(train['ENGINESIZE']).reshape(-1, 1)
train_y = np.asanyarray(train['CO2EMISSIONS']).reshape(-1, 1)
# OR
x_train = np.asanyarray(train[['ENGINESIZE']])
y_train = np.asanyarray(train[['CO2EMISSIONS']])
# Train a Linear Regression Model
regr.fit(x_train, y_train)

# The coefficients
print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)

# Coefficients:  [[39.44568829]]
# Intercept:  [124.62553792]

# Plot fitted line on training data
plt.figure(figsize=(8, 6))
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='crimson', label='Training Data')
plt.plot(x_train, regr.predict(x_train), '-k', label='Fitted Line')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.title("Linear Regression Model for Engine Size vs CO2 Emissions")
plt.legend()
plt.tight_layout()
plt.savefig('engine_size_vs_emissions.png', format='png', dpi=300)
plt.show()

# Predict on test data
x_test = np.asanyarray(test[['ENGINESIZE']])
y_test = np.asanyarray(test[['CO2EMISSIONS']])
y_pred = regr.predict(x_test)

# Evaluation metrics: MAE, MSE, RMSE
# Comparing actual values and predicted values to calculate the accuracy of the model.
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Evaluation Metrics for ENGINESIZE:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

# Evaluation Metrics for ENGINESIZE:
# Mean Absolute Error (MAE): 22.55
# Mean Squared Error (MSE): 889.48
# R2 Score: 0.78

# Plotting residuals
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.scatterplot(x=x_test.flatten(), y=residuals.flatten(), color='crimson')
plt.axhline(0, color='k', linestyle='--')
plt.xlabel("Engine Size")
plt.ylabel("Residuals")
plt.title("Residuals Plot for Engine Size vs CO2 Emissions")
plt.tight_layout()
plt.savefig('residuals_plot.png', format='png', dpi=300)
plt.show()

# Train/test with FUELCONSUMPTION_COMB
regr_fuel = linear_model.LinearRegression()
xf_train = np.asanyarray(train[['FUELCONSUMPTION_COMB']])
# Train a Linear Regression Model
regr_fuel.fit(xf_train, y_train)

# The coefficients
print('Coefficients: ', regr_fuel.coef_)
print('Intercept: ', regr_fuel.intercept_)

# Coefficients:  [[15.76687088]]
# Intercept:  [72.67796862]

# Evaluation on test data for FUELCONSUMPTION_COMB
xf_test = np.asanyarray(test[['FUELCONSUMPTION_COMB']])
# Predictions
yf_pred = regr_fuel.predict(xf_test)

mae_fuel = mean_absolute_error(y_test, yf_pred)
mse_fuel = mean_squared_error(y_test, yf_pred)
r2_fuel = r2_score(y_test, yf_pred)

print("\nEvaluation Metrics for FUELCONSUMPTION_COMB:")
print(f"Mean Absolute Error (MAE): {mae_fuel:.2f}")
print(f"Mean Squared Error (MSE): {mse_fuel:.2f}")
print(f"R2 Score: {r2_fuel:.2f}")

# Mean Absolute Error (MAE): 19.10
# Mean Squared Error (MSE): 654.14
# R2 Score: 0.84

# Visualizing comparison of evaluation metrics
metrics = ['MAE', 'MSE', 'R2 Score']
engine_size_metrics = [mae, mse, r2]
fuel_consumption_metrics = [mae_fuel, mse_fuel, r2_fuel]

fig, ax = plt.subplots(figsize=(10, 6))
index = np.arange(len(metrics))
bar_width = 0.35
bar1 = plt.bar(index, engine_size_metrics, bar_width, color='crimson', label='Engine Size')
bar2 = plt.bar(index + bar_width, fuel_consumption_metrics, bar_width, color='blue', label='Fuel Consumption')

# Add the text labels on top of the bars
for bar in bar1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., height,
            f'{height:.2f}', ha='center', va='bottom')

for bar in bar2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., height,
            f'{height:.2f}', ha='center', va='bottom')

plt.xlabel('Evaluation Metric')
plt.ylabel('Score')
plt.title('Comparison of Evaluation Metrics for Engine Size and Fuel Consumption Models')
plt.xticks(index + bar_width / 2, metrics)
plt.legend()
plt.tight_layout()
plt.savefig('evaluation_metrics_comparison.png', format='png', dpi=300)
plt.show()

# Final result: The model trained using FUELCONSUMPTION_COMB worked better than the model
# using ENGINESIZE based on all three evaluation metrics (MAE, MSE, and R² Score).
# This model makes more accurate predictions, has fewer large errors, and better explains
# the variance in the target variable (CO2EMISSIONS).