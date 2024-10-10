'''
This Python code uses the linear regression technique to model and analyze store
sales over time based on data from Corporación Favorita, a large Ecuadorian grocery retailer.
The dataset consists of daily sales data for multiple stores and product families, stored in
a CSV file within a ZIP archive.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Read the CSV file from ZIP and parse the 'date' column
df = pd.read_csv('data/train.csv.zip', compression='zip', parse_dates=['date'])
print(f"Dataset Shape: {df.shape}")
print(f"Dataset Columns: {list(df.columns)}")
df.info()

# Prepare the data: set date index, group by date, and calculate average sales
df = df.set_index('date').to_period('D')
df = df.set_index(['store_nbr', 'family'], append=True)
df = df.groupby('date').mean()['sales']
df = df.to_frame()
print(df.head())

# Create a time feature for time-based model
df['time'] = np.arange(len(df.index))

# Train the time-based model
X_time = df[['time']]
y_time = df['sales']
model_time = LinearRegression()
model_time.fit(X_time, y_time)
print('Time Model Coefficient:', model_time.coef_)
print('Time Model Intercept:', model_time.intercept_)

# Predict sales using the time model
y_pred_time = pd.Series(model_time.predict(X_time), index=X_time.index)

# Plot actual sales and predicted sales for time-based model
fig, ax = plt.subplots(figsize=(11, 4))
plt.style.use("seaborn-v0_8-whitegrid")
plt.rc(
    "figure",
    autolayout=True,
    titlesize=18,
    titleweight='bold',
)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)
plot_params = {
    'color': '0.75',
    'marker': '.',
    'linestyle': '-',
    'markeredgecolor': '0.25',
    'markerfacecolor': '0.25'
}
# Plotting time-based model
ax = y_time.plot(ax=ax, **plot_params)
ax = y_pred_time.plot(color='crimson', linewidth=3)
plt.title('Time Plot of Total Product Sale')
plt.xlabel('Date')
plt.ylabel('Number of Sales')
plt.tight_layout()
plt.savefig('e3_product_sales_time_plot.png', format='png', dpi=300)
plt.show()

# Create lag feature for lag-based model
df['lag_1'] = df['sales'].shift(1)
print(df.head())

# Train the lag-based model
X_lag = df[['lag_1']].dropna()
y_lag = df['sales'].loc[X_lag.index]
y_lag, X_lag = y_lag.align(X_lag, join='inner')  # Drop corresponding values in target
model_lag = LinearRegression()
model_lag.fit(X_lag, y_lag)
print('Lag Model Coefficient:', model_lag.coef_)
print('Lag Model Intercept:', model_lag.intercept_)

# Predict sales using the lag model
y_pred_lag = pd.Series(model_lag.predict(X_lag), index=X_lag.index)

# Plot actual sales and predicted sales for lag-based model
fig, ax = plt.subplots()
ax.plot(X_lag['lag_1'], y_lag, '.', color='0.25')
ax.plot(X_lag['lag_1'], y_pred_lag, linewidth=3, color='crimson')
ax.set_aspect('equal')
plt.title('Lag Plot of Product Sales')
plt.xlabel('Lag_1')
plt.ylabel('Number of Sales')
plt.tight_layout()
plt.savefig('e3_product_sales_lag_plot.png', format='png', dpi=300)
plt.show()