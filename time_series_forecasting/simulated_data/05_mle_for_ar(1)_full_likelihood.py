"""
This code computes the MLE for ϕ using the full likelihood
"""

import numpy as np
from scipy.optimize import minimize_scalar
from statsmodels.tsa.arima_process import ArmaProcess

# Set seed for reproducibility
np.random.seed(2021)
T = 500  # number of time points

# Parameters for AR(1) process with phi = 0.9 and variance = 1
phi = 0.9  # AR(1) coefficient
v = 1.0  # Variance of the white noise (innovations)
sd = np.sqrt(v)  # Standard deviation of innovations

# Simulate AR(1) process
ar = np.array([1, -phi])  # AR(1) model specification: yt = phi * y_{t-1} + eps
ma = np.array([1])  # No MA part
arma_process = ArmaProcess(ar, ma)
yt = arma_process.generate_sample(nsample=T, scale=sd)

# Log likelihood function for the AR(1) process
def log_p(phi, yt):
    return 0.5 * (np.log(1 - phi**2) - np.sum((yt[1:] - phi * yt[:-1])**2) - yt[0]**2 * (1 - phi**2))

# Use a built-in optimization method to find the MLE of phi
result = minimize_scalar(lambda phi: -log_p(phi, yt), bounds=(-1, 1), method='bounded', options={'xatol': 1e-4})

# Output the MLE for phi
print(f"MLE of full likelihood for phi:{result.x: .4f}")

