# Sleep Efficiency Analysis

## Overview

This project analyses sleep data to identify factors that impact sleep efficiency. The dataset was obtained from [Kaggle](https://www.kaggle.com/datasets/equilibriumm/sleep-efficiency). I explored the relationship between various demographic, lifestyle, and sleep-related factors and sleep efficiency using machine learning models.

## Dataset

The dataset consists of 452 rows and 15 columns, as follows:

- **Demographic Information:**

    1. Subject ID: Unique identifier for each test subject.
    2. Age: Age of the test subject (in years).
    3. Gender: Gender of the test subject (Male, Female).

- **Sleep Patterns and Sleep Quality:**

    4. Bedtime: The time the test subject goes to bed each night (as a timestamp YYYY-MM-DD HH:MM:SS).
    5. Wakeup Time: The time the test subject wakes up each morning (as a timestamp).
    6. Sleep Duration: The total amount of time the test subject spends sleeping each night (in hours between 5 - 10).
    7. Sleep Efficiency: The proportion of time spent in bed that is actually spent sleeping (between 0.5 - 0.99).
    8. REM Sleep Percentage: The percentage of total sleep spent in the REM (Rapid Eye Movement) stage (between 15 - 30 %).
    9. Deep Sleep Percentage: The percentage of total sleep spent in the deep sleep stage (between 18 - 75).
    10. Light Sleep Percentage: The percentage of total sleep spent in the light sleep stage (between 7 - 63).
    11. Awakenings: The number of times the test subject wakes up during the night (between 0 and 4).
 
- **Lifestyle Factors:**

    12. Caffeine Consumption: Amount of caffeine consumed in the 24 hours prior to bedtime (in milligrams).
    13. Alcohol Consumption: Amount of alcohol consumed in the 24 hours prior to bedtime (number of drinks, between 0 and 5).
    14. Smoking Status: Whether the test subject is a smoker (binary: yes/no).
    15. Exercise Frequency: How frequently the test subject exercises (numerical: number of hours per week, between 0 and 5).

## Methodology

### Data Preprocessing

- **Handling Missing Values:** Missing values in columns like Awakenings, Caffeine Consumption, Alcohol Consumption, and Exercise Frequency were filled using the median.
- **Feature Engineering:** Time data was converted into minutes since midnight for consistency. Gender and smoking status were encoded as numerical values.

### Exploratory Data Analysis (EDA)

Various visualizations were created to understand the distribution of features and their correlations with sleep efficiency.

The correlation matrix shows:

1. A significant degree of correlation between sleep efficiency, deep sleep percentage, light sleep percentage, REM sleep percentage, and awakenings
2. A negative correlation between alcohol consumption and sleep efficiency and deep sleep percentage.
3. A positive correlation between alcohol consumption and light sleep percentage and awakenings.
4. A moderate positive correlation between exercise and sleep efficiency and deep sleep percentage.
5. A moderate negative correlation between exercise and awakenings and light sleep percentage.
6. A negative correlation between smoking and sleep efficiency and deep sleep percentage.
7. A positive correlation between smoking and light sleep percentage.

### Modeling

**Random Forest Regressor:** The Random Forest Regressor was used to model sleep efficiency using the features 'Age', 'Exercise frequency', 'Alcohol consumption', 'Smoking_status_encoded'. Feature importance was calculated to identify the most significant predictors.

**Evaluation:** The model was evaluated using metrics like Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R²).

### Results
- **Feature Importance:**
    - Age: 36.67%
    - Alcohol Consumption: 26.43%
    - Exercise Frequency: 19.31%
    - Smoking Status: 17.59%

- **Model Performance:**
    - Mean Squared Error (MSE): 0.0094
    - Mean Absolute Error (MAE): 0.0720
    - R-squared (R²): 0.4966

The results indicate that age, alcohol consumption, exercise frequency, and smoking status are the most significant factors affecting sleep efficiency. However, the R-squared value suggests that there is considerable variance not explained by these features.
