"""
This script applies differencing and moving average filtering on CO2 concentration data.
The dataset contains CO2 concentrations in parts per million, collected at the Mauna Loa Observatory, Hawaii.

Techniques:
1. First differencing (co2_1stdiff) - used to remove the trend from the time series.
   - Note: The time series is adjusted as `time_series[1:]` after differencing,
     since differencing reduces the data length by one.
2. Moving average smoothing (co2_ma) - used to filter out seasonality with a window size of 12.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage import uniform_filter1d

# Configure matplotlib for better visual appearance
mpl.rc('xtick', labelsize=16)
mpl.rc('ytick', labelsize=16)
mpl.rcParams.update({'font.size': 15})
plt.ioff()

# Path to the Excel file
file_path = "data/co2-atmospheric-mlo-monthly-scripps.xls"
# List all available sheets in the Excel file
xls = pd.ExcelFile(file_path)
print("Available sheets:", xls.sheet_names)

# Load the CO2 dataset, skip the first 6 rows to reach the data
df = pd.read_excel(file_path, sheet_name= 'Scripps CO2 Dataset', header=6)
print("Column names:", df.columns)

# Extract the time and CO2 concentration series
time_series = df['Decimal Date']
co2_series = df['CO2 Filled']

# Apply first differences to remove trend
co2_1stdiff = co2_series.diff().dropna()

# Apply a moving average filter (window size = 12) to remove seasonality
co2_ma = uniform_filter1d(co2_series, size=12, mode='nearest')

plt.figure(figsize=(14.4, 13.35))
# Plot the original data
plt.subplot(3, 1, 1)
plt.plot(time_series, co2_series, label='Original CO2 Series', color='blue')
plt.xlabel('Time')
plt.ylabel('CO2 Levels')
plt.legend(loc='best')

# Plot the first differences (to remove trend)
plt.subplot(3, 1, 2)
plt.plot(time_series[1:], co2_1stdiff, label='First Differences', color='orange')
plt.xlabel('Time')
plt.ylabel('CO2 Diff')
plt.legend(loc='best')

# Plot the filtered series via moving averages (to remove seasonality)
plt.subplot(3,1,3)
plt.plot(time_series, co2_ma, label='Filtered (Moving Average)', color='green')
plt.xlabel('Time')
plt.ylabel('CO2 MA')

plt.tight_layout()
plt.legend(loc='best')
plt.savefig('co2_emission_diff_ma.jpg', format='jpg', dpi=300)
plt.show()