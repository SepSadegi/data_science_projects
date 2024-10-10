'''
This code analyzes the daily traffic data from the Baregg Tunnel in Switzerland, utilizing
linear regression to explore the relationship between time-based and lag-based features
in time series forecasting. The dataset shows the number of vehicles passing through
the tunnel from November 2003 to November 2005.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Flags to control individual plotting sections
plot_time_model = False
plot_lag_model = False
plot_ma = True

# Load the dataset and parse dates
tunnel_df = pd.read_csv('data/tunnel.csv', parse_dates=['Day'])
print(f"Dataset Shape: {tunnel_df.shape}")
print(f"Dataset Columns: {list(tunnel_df.columns)}")
tunnel_df.info()

# Create a time series by setting the index to the 'Day' column
df = tunnel_df.set_index('Day').to_period()

# Check for missing values
missing_count = df.isnull().sum()
print("Number of missing values per column:")
print(missing_count[missing_count > 0])  # Only show columns with missing values

# Create a time-step feature
df['Time'] = np.arange(len(df.index))
print(df.head())

# Training data for time-based model
X_time = df[['Time']]
y_time = df['NumVehicles']

# Train the time-based model
model_time = LinearRegression()
model_time.fit(X_time, y_time)

print('Time Model Coefficient:', model_time.coef_)
print('Time Model Intercept:', model_time.intercept_)

# Store the fitted values to a time series with the same index
y_pred_time = pd.Series(model_time.predict(X_time), index=X_time.index)

# Set Matplotlib style and parameters
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

if plot_time_model:
    # Plotting time-based model
    fig, ax = plt.subplots(figsize=(11, 4))
    # Plotting time-based model
    plt.plot(df.index.to_timestamp(), df.NumVehicles, **plot_params)
    plt.plot(df.index.to_timestamp(), y_pred_time, linewidth=3, color='royalblue')
    plt.title('Time Plot of Tunnel Traffic')
    plt.xlabel('Date')
    plt.ylabel('Number of Vehicles')
    plt.tight_layout()
    plt.savefig('e2_tunnel_traffic_time_plot.png', format='png', dpi=300)
    plt.show()

# Create Lag Features
df['Lag_1'] = df['NumVehicles'].shift(1)
print(df.head())

# Training data for lag-based model
X_lag = df[['Lag_1']].dropna()
y_lag = df['NumVehicles'].loc[X_lag.index]
y_lag, X_lag = y_lag.align(X_lag, join='inner')  # Drop corresponding values in target

# Train the lag-based model
model_lag = LinearRegression()
model_lag.fit(X_lag, y_lag)

print('Lag Model Coefficient:', model_lag.coef_)
print('Lag Model Intercept:', model_lag.intercept_)

# Store the fitted values to a time series with the same index
y_pred_lag = pd.Series(model_lag.predict(X_lag), index=X_lag.index)

if plot_lag_model:
    # Plotting Lag-based model
    fig, ax = plt.subplots()
    ax.plot(X_lag['Lag_1'], y_lag, '.', color='0.25')
    ax.plot(X_lag['Lag_1'], y_pred_lag, linewidth=3, color='royalblue')
    ax.set_aspect('equal')
    plt.title('Lag Plot of Tunnel Traffic')
    plt.xlabel('Lag_1')
    plt.ylabel('Number of Vehicles')
    plt.tight_layout()
    plt.savefig('e2_tunnel_traffic_lag_plot.png', format='png', dpi=300)
    plt.show()

# Modelling Trend
moving_average = df.rolling(
    window=365,
    center=True,
    min_periods=183,
).mean()

if plot_ma:
    # Plotting One-Year Moving Average
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df.index.to_timestamp(), df.NumVehicles, **plot_params)
    ax.plot(moving_average.index.to_timestamp(), moving_average.NumVehicles, linewidth=3, color='green')
    plt.title('Tunnel Traffic - 365-Day Moving Average')
    plt.xlabel('Date')
    plt.ylabel('Number of Vehicles')
    plt.tight_layout()
    plt.savefig('e2_tunnel_traffic_ma_plot.png', format='png', dpi=300)
    plt.show()