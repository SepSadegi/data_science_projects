'''
Time Series Analysis: Exploring Lag Features in Hardcover Sales

Goal:
This code explores the relationship between the current day's Hardcover book sales
and the sales from the previous day (Lag_1). We aim to identify whether there is a
significant serial dependence between consecutive days, which would make lag features
useful for forecasting.

Conclusion:
We observe that there is a strong positive correlation between sales on consecutive
days. When sales are high on one day, they are likely to be high the next day as well.
This indicates that lag features can be valuable in time series forecasting models.
'''

# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

# Checking for available styles in Matplotlib
# print(plt.style.available)

# Step 1: Load and Inspect the Dataset
# Read the CSV file into a DataFrame and display basic information
df = pd.read_csv('data/book_sales.csv')
df.info()

# Step 2: Feature Engineering
# Create a time-step feature to track the order of the data points
df['Time'] = np.arange(len(df.index))

# Create a lag feature: 'Lag_1' represents Hardcover sales from the previous day
df['Lag_1'] = df['Hardcover'].shift(1)

# Display the first few rows of the updated DataFrame
print(df.head())


# Step 3: Time Plot
plt.figure(figsize=(11, 4))
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
    titlesize=16,
    titlepad=10,
)

plt.plot('Time', 'Hardcover', data=df, color='0.75')
ax = sns.regplot(x='Time', y='Hardcover', data=df, ci=None, scatter_kws=dict(color='0.25'))
ax.set_title('Time Plot of Hardcover Sales')
ax.set_xlabel('Time')
ax.set_ylabel('Hardcover Sales')
plt.savefig('e1_hardcover_sales_time_plot.png', format='png', dpi=300)
plt.show()

# Step 4: Lag Plot
plt.figure(figsize=(6, 6))
ax = sns.regplot(x='Lag_1', y='Hardcover', data=df, ci=None, scatter_kws=dict(color='0.25'))
ax.set_title('Lag Plot of Hardcover Sales')
ax.set_xlabel('Sales on Previous Day (Lag_1)')
ax.set_ylabel('Hardcover Sales')
plt.savefig('e1_hardcover_sales_lag_plot.png', format='png', dpi=300)
plt.show()