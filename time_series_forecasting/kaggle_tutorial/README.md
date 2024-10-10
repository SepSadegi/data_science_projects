# Time Series Forecasting Basics

This repository documents my learning journey and exercises on time series forecasting, based on the 
[Kaggle Time Series Forecasting tutorial](https://www.kaggle.com/learn/time-series).

## Topics Covered

- **Time Series Basics**: Visualizing and analyzing time-dependent data.
- **Trend Analysis**: Identifying trends in time series data and using them to make predictions.
- **Seasonality**: Detecting repeating patterns that occur at regular intervals.
- **Lag Features**: Utilizing past values (lags) to make future predictions.
- **Autoregressive Models**: Capturing trends and seasonal effects using models like ARIMA.
- **Forecasting Methods**: Applying various models to generate time-based predictions.

## Linear Regression with Time Series

The **linear regression** algorithm learns to create a weighted sum from input features:

```plaintext
target = weight_1 * feature_1 + weight_2 * feature_2 + bias
```

During training, the regression algorithm estimates the parameters `weight_1`, `weight_2`​, and bias that best fit 
the target values, minimizing the squared error between predictions and actual values. 
The weights are known as **regression coefficients**, and the bias is called the intercept because it indicates where 
the graph crosses the y-axis.

## Types of Features in Time Series

There are two feature types unique to time series: **time-step features** and **lag features**.

### Time-step Features

Time-step features are derived directly from the time index. The most basic time-step feature is the **time dummy**, 
which counts time steps in the series:

```plaintext
target = weight * time + bias
```

Time-step features allow us to model **time dependence**.

### Lag Features

Lag features are created by shifting the observations of the target series so they appear to occur later in time. 
A 1-step lag feature can be created by shifting the observations of the target series by one time step, effectively 
moving each value to align with the value that occurs in the subsequent time period; however, it's also possible to 
shift by multiple steps to capture longer-term dependencies.

Linear regression with a lag feature produces the model:

```plaintext
target = weight * lag + bias
```

Lag features enable us to model **serial dependence**, which occurs when an observation can be predicted from previous 
observations.

## Autoregressive Models

 ### What is AR(1)?
An **AR(1) model** (first-order autoregressive) means that the current value of the time series depends only on the value 
from the previous time step. This can be expressed mathematically as:

$$
y_t = \phi y_{t-1} + \epsilon_t
$$

- $y_t$ is the value at time $t$.
- $\phi$ is the **autoregressive coefficient**, which determines the influence of the previous value $y_{t-1}$ 
on the current value.
- $\epsilon_t$ is a random error term (white noise).

Interpreting the regression coefficients can help us recognize serial dependence in a time plot.

### Range of the Autoregressive Coefficient:

The value of $\phi$ (the autoregressive coefficient) determines key properties of the time series:

- **If $|\phi| < 1$**, the process is **stationary**. This means the time series tends to revert to its mean over time 
and has a constant variance.
- **If $\phi = 1$**, the process becomes a **random walk** (non-stationary). Here, each observation is highly persistent, 
and the series "drifts" over time without reverting to a long-term mean.
- **If $|\phi| > 1$**, the process is **explosive**, meaning that the values in the series will grow without bound, 
leading to instability.
- **If $\phi = 0$**, the process is just **white noise**, meaning no dependence on past values at all.

### Oscillatory Behavior

The sign of $\phi$ affects the nature of the series' movement:

- **When $\phi > 0$**: The process tends to move in the **same direction** as the previous time step. 
Positive values of $\phi$ mean the time series has **positive autocorrelation**.
- **When $\phi < 0$**: The process tends to **oscillate**, meaning that the current value is negatively related to 
the previous one. The series alternates around the mean, creating a back-and-forth, or wave-like, behavior. This 
implies **negative autocorrelation**.

In general, the closer $\phi$ is to 1 (in either direction), the stronger the correlation between successive values, 
and the more persistent the behavior of the series.

### Note:

In time series modeling, the role of the coefficient ($\phi$ or weight) varies depending on whether we're dealing with 
a time-based trend or an autoregressive lag.

#### Time-Based Trend Model

In a **time-based** trend model, the target is modeled as:

$$
target = \phi_{time} * Time + \epsilon_t
$$

- A positive $\phi_{time}$ indicates an upward trend over time (e.g., sales growing steadily).
- A negative $\phi_{time}$ suggests a downward trend.
- If $|\phi_{time}|$ is large, the trend is steep.

#### Lag-Based Autoregressive Model

In a **lag-based** autoregressive model like AR(1):

$$
target = \phi_{lag} * Lag_1 + \epsilon_t
$$

- Here, $\phi_{lag}$ captures the relationship between the current value and the previous value of the same series. 
- **Positive** $\phi_{lag}$ (close to 1) indicates **persistence**, meaning high values are often followed by high values, and low 
values are followed by low values. If $\phi_{lag}=1$, the series is a **random walk** and becomes non-stationary 
(values drift over time).
- **Negative** $\phi_{lag}$ results in **oscillation**, where high values are followed by low values, and vice versa.
- If **$|\phi_{lag}| < 1$**, the series is **stationary**, reverting to its mean.
- If **$|\phi_{lag}| > 1$**, the series becomes **explosive**, growing or shrinking without bounds. 

### Key Differences between $\phi_{lag}$ and $\phi_{time}$:

1. **Nature of Dependence**:
 - $\phi_{lag}$ reflects serial dependence (one value depends on the previous one).
 - $\phi_{time}$ reflects linear time-based trends (the target changes consistently over time).

2. **Conditions for Stability**:
 - $\phi_{lag}$ must be between -1 and 1 for stationarity.
 - $\phi_{time}$ does not have the same restriction because it does not influence stability. 
 Instead, it indicates whether the series has an upward or downward trend.

3. **Non-stationarity and Explosiveness**:
 - When $|\phi_{lag}| > 1$, the series becomes explosive.
 - In a time-based trend model, even a large $\phi_{time}$ leads to linear growth or decline, not exponential.
 
## Trend in the Time Series

The trend component of a time series represents a persistent, long-term change in the mean of the series. The trend 
is the slowest-moving part of a series, the part representing the largest time scale of importance. In a time series 
of product sales, an increasing trend might be the effect of a market expansion as more people become aware of the 
product year by year.

### Moving Average Plots

To see what kind of trend a time series might have, we can use a moving average plot. To compute a moving average of 
a time series, we compute the average of the values within a sliding window of some defined width. Each point on the 
graph represents the average of all the values in the series that fall within the window on either side. The idea is 
to smooth out any short-term fluctuations in the series so that only long-term changes remain.

For a change to be a part of the trend, it should occur over a longer period than any seasonal changes. To visualize 
a trend, therefore, we take an average over a period longer than any seasonal period in the series.

To create a moving average, first use the `rolling` method to begin a windowed computation. Follow this by the `mean` 
method to compute the average over the window. 

### Engineering Trend

Once we've identified the shape of the trend, we can attempt to model it using a time-step feature by scikit-learn's 
LinearRegression.

---

## Example 1: Exploring Lag Features in Hardcover Sales

This example illustrates how to create and analyze lag features in time series data using `pandas` and `seaborn`. 
I use a dataset of daily Hardcover book sales to understand the serial dependence between consecutive days. 
Specifically, I compare sales on one day with the sales from the previous day (`Lag_1`).

- **Code:** `e1_hardcover_book_sales.py`
- **Training Data:** `book_sales.csv`
- **Plots:** `e1_hardcover_sales_time_plot.png` and `e1_hardcover_sales_lag_plot.png`

### **Key Learnings:**
- **Lag Features**: Lag features capture the relationship between the current observation and past observations, 
making them useful for forecasting models.
- **Serial Dependence**: We see a strong positive correlation between `Hardcover` sales on one day and the 
previous day (`Lag_1`), indicating that high sales on one day are often followed by high sales the next day.
- **Visual Analysis**: The lag plot shows a clear trend, which suggests that including lag 
features in time series models can improve prediction accuracy.

The `Time Plot` helps visualize trends over time, while the `Lag Plot` demonstrates the relationship between 
consecutive days (the serial dependence). Together, these visualizations highlight the importance of lag features 
in time series analysis.

One advantage of linear regression is its explainability; it’s easy to interpret the contribution of each feature 
to predictions.

The **Lag_1** feature is a strong indicator of **autoregressive behavior** in this time series, specifically **AR(1)**.

---

## Example 2: Tunnel Traffic Analysis

This example explores the daily vehicle traffic through the Baregg Tunnel in Switzerland from November 2003 to 
November 2005. I apply linear regression techniques to analyze time-based and lag-based features in this time series dataset.

- **Code:** `e2_tunnel_traffic.py`
- **Training Data:** `tunnel.csv`
- **Plots:** `e2_tunnel_traffic_time_plot.png` and `e2_tunnel_traffic_lag_plot.png`

### **Key Learnings:**
- **Time-Based Features:** We create a time-step feature to model the relationship between the number of vehicles 
and the passage of time. The time model indicates how traffic trends vary with time.
- **Lag Features:** By creating a 1-step lag feature (`Lag_1`), we capture the dependency of the current traffic 
volume on the previous day's volume. This helps us understand the serial dependence in the traffic data.
- **Visual Analysis:** The time plot shows overall traffic trends, while the lag plot illustrates the relationship 
between current and previous traffic volumes, highlighting the importance of lag features in forecasting models.

---

## Example 3: Store Sales Forecasting

This example shows time series forecasting for product sales using linear regression models. The dataset 
comes from Corporación Favorita, a major grocery retailer in Ecuador, and comprises daily sales records for 
various product families across multiple stores. Our analysis aims to understand sales trends and patterns 
and build predictive models that help forecast future sales.

- **Code:** `e3_product_sales.py`
- **Training Data:** `train.csv.zip`
- **Plots:** `e3_product_sales_time_plot.png` and `e3_product_sales_lag_plot.png`

---



