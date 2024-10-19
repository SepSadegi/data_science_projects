"""
This script simulates two AR(1) processes with different AR coefficients
(high positive and high negative) and visualizes their time series, ACF, and PACF.

1. For a positive AR(1) coefficient (phi > 0):
   - The ACF decays exponentially as a function of lag.
   - ACF values remain positive at all lags.
   - The PACF shows a strong coefficient at lag 1, with all coefficients
      for lags greater than 1 being approximately zero.

2. For a negative AR(1) coefficient (phi < 0):
   - The ACF also decays exponentially as a function of lag.
   - The ACF oscillates between negative and positive values as it decays,
     reflecting the alternating behaviour typical of processes with negative coefficients.
   - Similar to the positive phi case, the PACF shows a non-zero value at lag 1
     and near-zero coefficients for all lags greater than 1.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Configure matplotlib for better visual appearance
mpl.rc('xtick', labelsize=16)
mpl.rc('ytick', labelsize=16)
mpl.rcParams.update({'font.size': 15})
plt.ioff()

# Set seed for reproducibility
np.random.seed(2021)
T = 500  # number of time points

# Parameters for AR(1) process with phi = 0.9 and variance = 1
v = 1.0  # Innovation variance
sd = np.sqrt(v)  # Innovation standard deviation
phi1 = 0.9  # AR(1) coefficient for first process

# Simulate AR(1) process with phi = 0.9
ar1_process = ArmaProcess(ar=[1, -phi1], ma=[1])  # AR(1) process defined in statsmodels
yt1 = ar1_process.generate_sample(nsample=T, scale=sd)  # Generate time series

# Parameters for AR(1) process with phi = -0.9 and variance = 1
phi2 = -0.9  # AR(1) coefficient for second process

# Simulate AR(1) process with phi = -0.9
ar2_process = ArmaProcess(ar=[1, -phi2], ma=[1])
yt2 = ar2_process.generate_sample(nsample=T, scale=sd)

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(yt1)
plt.title(r'Time Series for $\phi = 0.9$', fontsize=14)

plt.subplot(2, 1, 2)
plt.plot(yt2)
plt.title(r'Time Series for $\phi = -0.9$', fontsize=14)

plt.tight_layout()
plt.savefig('02_ar(1)_pos_neg_phi.jpg', format='jpg', dpi=300)
plt.show()

# Set up for ACF and PACF plots
lag_max = 50  # Maximum lag to compute ACF and PACF

# Plot the theoretical ACF for both AR(1) processes
plt.figure(figsize=(10, 8))

# Theoretical ACF for phi = 0.9
cov_0 = sd**2 / (1 - phi1**2)  # Auto-covariance at lag 0
cov_h = [phi1**h * cov_0 for h in range(lag_max + 1)]  # Auto-covariance at lag h
plt.subplot(3, 2, 1)
plt.stem(range(lag_max + 1), np.array(cov_h) / cov_0, linefmt='r-', markerfmt='ro', basefmt='k')
plt.title(r'$\phi = 0.9$', fontsize=14)
plt.ylim(-1, 1)
plt.xlabel('Lag')
plt.ylabel('True ACF')

# Theoretical ACF for phi = -0.9
cov_0 = sd**2 / (1 - phi2**2)  # Auto-covariance at lag 0
cov_h = [phi2**h * cov_0 for h in range(lag_max + 1)]  # Auto-covariance at lag h
plt.subplot(3, 2, 2)
plt.stem(range(lag_max + 1), np.array(cov_h) / cov_0, linefmt='r-', markerfmt='ro', basefmt='k')
plt.title(r'$\phi = -0.9$', fontsize=14)
plt.ylim(-1, 1)
plt.xlabel("Lag")
plt.ylabel("True ACF")

# Plot the sample ACF for both AR(1) processes
plot_acf(yt1, lags=lag_max, ax=plt.subplot(3, 2, 3), title=" ")
plt.xlabel("Lag")
plt.ylabel("Sample ACF")
plot_acf(yt2, lags=lag_max, ax=plt.subplot(3, 2, 4), title=" ")
plt.xlabel("Lag")
plt.ylabel("Sample ACF")

# Plot the sample PACF for both AR(1) processes
plot_pacf(yt1, lags=lag_max, ax=plt.subplot(3, 2, 5), title=" ")
plt.xlabel("Lag")
plt.ylabel("Sample PACF")
plot_pacf(yt2, lags=lag_max, ax=plt.subplot(3, 2, 6), title=" ")
plt.xlabel("Lag")
plt.ylabel("Sample PACF")

plt.tight_layout()
plt.savefig('02_ar(1)_acf_pacf.jpg', format='jpg', dpi=300)
plt.show()

