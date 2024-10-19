"""
This code manually computes the Maximum Likelihood Estimate (MLE) of the autoregressive (AR)
coefficient ϕ, the unbiased estimator of the innovation variance v (denoted as s2), and the
MLE of v based on a dataset simulated from an AR(1) process.

The estimation is performed using the conditional likelihood, which conditions on the first
observation. This approach is useful as it simplifies the likelihood function, allowing for
closed-form solutions for ϕ and v.
"""
import numpy as np

# Set seed for reproducibility
np.random.seed(2021)
T = 500  # number of time points

# Parameters for AR(1) process with phi = 0.9 and variance = 1
phi = 0.9  # AR(1) coefficient
v = 1.0  # Variance of the white noise (innovations)
sd = np.sqrt(v)  # Standard deviation of innovations

# Simulate AR(1) time series
yt = np.zeros(T)
yt[0] = np.random.normal(0, sd)  # Initial value
for t in range(1, T):
    yt[t] = phi * yt[t-1] + np.random.normal(0, sd)

## Case 1: Conditional likelihood
# Define response vector (y) and design matrix (X)
y = yt[1:T].reshape(-1, 1)  # Response vector (y from 2 to T)
X = yt[0:T-1].reshape(-1, 1)  # Design matrix (X from 1 to T-1)

# MLE for AR(1) coefficient (phi)
phi_MLE = np.dot(X.T, y) / np.sum(X**2)

# Unbiased estimate for variance s^2
s2 = np.sum((y - phi_MLE * X)**2) / (len(y) - 1)

# MLE for variance v
v_MLE = s2 * (len(y) - 1) / len(y)

# Output the results
print(f"MLE of conditional likelihood for phi: {phi_MLE[0][0]:.4f}")
print(f"MLE for the variance v: {v_MLE:.4f}")
print(f"Estimate s^2 for the variance v: {s2:.4f}")

# MLE of conditional likelihood for phi: 0.8914
# MLE for the variance v: 0.9594
# Estimate s2 for the variance v: 0.9613