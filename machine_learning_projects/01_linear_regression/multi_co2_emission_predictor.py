# Objectives:
# Use scikit-learn to implement Multiple Linear Regression
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

# Select some features to use for regression
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
print(cdf.head(9))

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

# Train/test split (80/20)
# 80% of the entire dataset will be used for training and 20% for testing
# We create a mask to select random rows using np.random.rand() function
msk = np.random.rand(len(df)) < 0.8  # creates boolean array (probability of 0.8 being True and 0.2 being False)
train = cdf[msk]
test = cdf[~msk]  # ~msk return a new boolean array where each 'True' value in msk becomes 'False' and vice versa

# Multiple Regression Model means that more than one independant variable is present
# Here, we are predicting the co2 emission using the FUELCONSUMPTION_COMB, EngineSize and Cylinders of cars.
regr = linear_model.LinearRegression()
x_train = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
y_train = np.asanyarray(train[['CO2EMISSIONS']])
# Train a Multiple Regression Model
regr.fit(x_train, y_train)

# The coefficients of the hyperplane
print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)

# Predict on test data
x_test = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
y_test = np.asanyarray(test[['CO2EMISSIONS']])
y_pred = regr.predict(x_test)

# Evaluation metrics: MAE, MSE, RMSE
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Evaluation Metrics for ENGINESIZE:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R2 Score: {r2:.2f}")  # Variance score
print('Variance score: %.2f' % regr.score(x_test, y_test))

# Residuals plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test.flatten(), y=residuals.flatten(), color='purple')
plt.axhline(0, color='k', linestyle='--')
plt.title("Residuals vs CO2 Emissions (Predicted)")
plt.xlabel("CO2 Emissions (True)")
plt.ylabel("Residuals")
plt.savefig('residuals_plot_multiple_regression.png', format='png', dpi=300)
plt.show()


# Multiple Regression with FUELCONSUMPTION_CITY and FUELCONSUMPTION_HWY
regr2 = linear_model.LinearRegression()
x2_train = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']])
regr2.fit(x2_train, y_train)

# Predict on test data (coefficients of the new hyperplane)
x2_test = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']])
y2_pred = regr2.predict(x2_test)

# Evaluation metrics for the second model
mse2 = mean_squared_error(y_test, y2_pred)
mae2 = mean_absolute_error(y_test, y2_pred)
r2_2 = r2_score(y_test, y2_pred)

print(f"Mean Squared Error (MSE): {mse2:.2f}")
print(f"Mean Absolute Error (MAE): {mae2:.2f}")
print(f"R² Score: {r2_2:.2f}")  # Variance score
# Explained variance score: 1 is perfect prediction

# Comparison Bar Plot of Evaluation Metrics for both models
comparison_metrics = ['MAE', 'MSE', 'R²']
model_1 = [mae, mse, r2]
model_2 = [mae2, mse2, r2_2]

fig, ax = plt.subplots(figsize=(10.5, 6))
index = np.arange(len(comparison_metrics))
bar_width = 0.35

bar1 = plt.bar(index, model_1, bar_width, label='Model 1 (FUELCONSUMPTION_COMB)', color='mediumslateblue')
bar2 = plt.bar(index + bar_width, model_2, bar_width, label='Model 2 (FUELCONSUMPTION_CITY & HWY)', color='yellowgreen')

# Add the text labels on top of the bars
for bar in bar1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.2f}', ha='center', va='bottom')
for bar in bar2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.2f}', ha='center', va='bottom')

ax.set_xlabel('Evaluation Metrics')
ax.set_ylabel('Values')
ax.set_title('Model Comparison: FUELCONSUMPTION_COMB vs CITY & HWY')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(comparison_metrics)
ax.legend(loc='best')
plt.tight_layout()
plt.savefig('model_comparison_multiple_regression.png', format='png', dpi=300)
plt.show()