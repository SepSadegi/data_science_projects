'''
This script simulates white noise, and plots:
1. A time series plot of the white noise.
2. The Autocorrelation Function (ACF).
3. The Partial Autocorrelation Function (PACF).

White noise refers to a sequence of random data points with a mean of zero and no temporal structure.
'''

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Set seed for reproducibility
np.random.seed(2021)
T = 200  # number of time points
t = np.arange(1, T+1)

# Simulate white noise: data with no temporal structure
# Mean (loc) = 0, Standard Deviation (scale) = 1
y_white_noise = np.random.normal(loc=0, scale=1, size=T)

plt.figure(figsize=(15, 5))

# Plot time series
plt.subplot(1, 3, 1)
plt.plot(t, y_white_noise, color='red')
plt.title("White Noise Time Series")
plt.xlabel("Time (t)")
plt.ylabel("Y(t)")

# Plot sample ACF
plt.subplot(1, 3, 2)
plot_acf(y_white_noise, lags=20, ax=plt.gca())
plt.title("ACF")
plt.xlabel("Lag")
plt.ylabel("Sample ACF")

# Plot sample PACF
plt.subplot(1,3,3)
plot_pacf(y_white_noise, lags=20, ax=plt.gca())
plt.title("PACF")
plt.xlabel("Lag")
plt.ylabel("Sample PACF")

plt.tight_layout()
plt.savefig('01_white_noise_acf_pacf.jpg', format='jpg', dpi=300)
plt.show()