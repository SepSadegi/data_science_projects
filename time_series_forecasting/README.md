Time series are datasets where each data point is linked to a specific timestamp. Typically, these 
points are arranged in chronological order and recorded at regular intervals (e.g., every minute or 
every day).

The goal of time series forecasting is to learn patterns from past data to predict future values. 
Time series forecasting is a crucial aspect of predictive analytics, especially when dealing with 
time-dependent data, such as stock prices, weather conditions, and sales trends. This repository is 
for practising time series analysis and forecasting. 

## Important Concepts for Time Series Forecasting

Below are the key concepts and necessary techniques:
- **Basic Techniques**: ACF, PACF, stationarity, seasonality, trend.
- **Statistical Models**: ARIMA, SARIMA, Holt-Winters, exponential smoothing.
- **Machine Learning/Deep Learning**: LSTM, neural networks, feature engineering.
- **Evaluation Metrics**: MAE, MSE, RMSE, MAPE.
- **Advanced Tools**: STL decomposition, Prophet, Granger causality.


## 1. Stationarity
- **Definition**: A time series is stationary if its statistical properties (mean, variance, 
autocorrelation) do not change over time.
- **Importance**: Many forecasting methods, such as ARIMA, assume that the time series is stationary. 
If our data is not stationary, we might need to apply transformations, such as differencing or 
logarithmic scaling, to make it stationary.

## 2. Autocorrelation and Partial Autocorrelation
- **ACF (Autocorrelation Function)**: Measures how a time series is correlated with its past values 
at different lags. A high autocorrelation at lag $k$ means that values $k$ time steps apart are similar.
- **PACF (Partial Autocorrelation Function)**: Like ACF but removes the effects of shorter lags. It 
shows the direct correlation between a time series and its lagged values, excluding intermediate lags.
- **Importance**: ACF and PACF plots help identify the order of AR and MA components in models like 
ARIMA.

## 3. ARIMA (Autoregressive Integrated Moving Average)
- **AR (Autoregressive)**: The model uses the relationship between an observation and a number of lagged 
observations (previous time points).
- **I (Integrated)**: This represents the differencing step to make the series stationary.
- **MA (Moving Average)**: The model uses the relationship between an observation and a residual error 
from a moving average model applied to lagged observations.
- **Importance**: ARIMA is one of the most widely used methods for univariate time series forecasting, 
especially for short-term forecasts. It's effective for time series with trends and seasonal patterns.

## 4. Seasonality and Trend
- **Trend**: The long-term increase or decrease in the data.
- **Seasonality**: Repeating patterns or cycles in the data that occur at regular intervals 
(e.g., daily, monthly, yearly).
- **Importance**:  Capturing and modeling trends and seasonality correctly is critical for 
accurate forecasting. Techniques like seasonal decomposition (e.g., STL decomposition) help 
separate these components.

## 5. Exponential Smoothing Methods
- **Simple Exponential Smoothing (SES)**: Suitable for data without trend or seasonality. It assigns 
exponentially decreasing weights to past observations.
- **Holt’s Linear Trend Model**: Adds support for trend in SES.
- **Holt-Winters (Triple Exponential Smoothing)**: Adds seasonality to the trend and smoothing 
components.
- **Importance**: Useful for smoothing data and producing forecasts for various types of time 
series (e.g., with trend or seasonality).

## 6. Seasonal Decomposition (STL Decomposition)
- **Additive and Multiplicative Models**: Break the time series into trend, seasonal, and residual 
components. In an additive model, the time series is expressed as the sum of these components. 
In a multiplicative model, the components are multiplied.
- **Importance**: Helps understand and model underlying patterns, allowing better pre-processing 
for forecasting models.

## 7. SARIMA (Seasonal ARIMA)
- **SARIMA**: Extends ARIMA by adding seasonal autoregressive, differencing, and moving average 
components.
- **Importance**: Useful when the data has seasonal patterns, which ARIMA may not handle well.

## 8. Cross-Validation for Time Series
- **Walk-forward validation**: Involves training the model on past data and testing on future data, 
preserving the temporal order.
- **Importance**: Ensures models are evaluated in a realistic way that mirrors real-world forecasting 
scenarios.

## 9. Lag Features and Feature Engineering
- **Lag Features**: These are features created by shifting the original time series by one or more 
time steps. They allow machine learning models (e.g., regression models, neural networks) to learn 
relationships between past values and the current value.
- **Rolling Statistics**: Rolling mean, rolling variance, and rolling median help capture trends and 
volatility over time.
- **Importance**: In machine learning-based forecasting (like LSTM or random forests), creating 
effective lag features can significantly improve the model’s performance.

## 10. Error Metrics for Forecasting
- **MAE (Mean Absolute Error)**: Average of absolute differences between predicted and actual values.
- **MSE (Mean Squared Error)**: Average of squared differences between predicted and actual values.
- **RMSE (Root Mean Squared Error)**: Square root of MSE, penalizing larger errors more than MAE.
- **MAPE (Mean Absolute Percentage Error)**: Average of absolute percentage differences between 
forecasted and actual values.
- **Importance**: These metrics are essential for evaluating forecasting model performance and 
comparing different models.

## 11. Long Short-Term Memory (LSTM) and Deep Learning Models
- **LSTM**: A type of recurrent neural network (RNN) that can learn long-term dependencies in time 
series data.
- **Other Neural Networks**: Emerging models, such as CNNs for time series and Transformer-based 
architectures, are increasingly used.
- **Importance**: Deep learning models, especially LSTMs, are effective at capturing complex patterns 
and relationships in time series.

## 12. Prophet
- **Prophet**: A forecasting tool developed by Facebook, particularly effective for time series with 
strong seasonal trends and missing data.
- **Importance**: Prophet is simple to use and flexible, allowing for the inclusion of holidays, 
trends, and custom seasonality components.

## 13. Granger Causality
- **Definition**: A statistical test to determine if one time series can predict another.
- **Importance**: Useful in multivariate time series analysis where relationships between multiple variables are of interest.





