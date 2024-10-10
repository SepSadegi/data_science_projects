'''
# Sleep Efficiency Analysis and Modeling

This script performs data analysis and machine learning modelling on a dataset related to sleep efficiency.
The main objective is to explore the relationships between various demographic, lifestyle, and sleep-related factors
and sleep efficiency, and to build a predictive model.

## Dataset

The dataset consists of 452 rows and 15 columns, as follows:
  Demographic Information:
    1. Subject ID: Unique identifier for each test subject.
    2. Age: Age of the test subject (in years).
    3. Gender: Gender of the test subject (Male, Female).

  Sleep Patterns and Sleep Quality:
    4. Bedtime: The time the test subject goes to bed each night (in hours, or as a timestamp).
    5. Wakeup Time: The time the test subject wakes up each morning (in hours, or as a timestamp).
    6. Sleep Duration: The total amount of time the test subject spends sleeping each night (in hours). (5 - 10)
    7. Sleep Efficiency: The proportion of time spent in bed that is actually spent sleeping (0.5 - 0.99).
    8. REM Sleep Percentage: The percentage of total sleep spent in the REM (Rapid Eye Movement) stage. (15 - 30)
    9. Deep Sleep Percentage: The percentage of total sleep spent in the deep sleep stage. (18 - 75)
    10. Light Sleep Percentage: The percentage of total sleep spent in the light sleep stage. (7 - 63)
    11. Awakenings: The number of times the test subject wakes up during the night.

  Lifestyle Factors:
    12. Caffeine Consumption: Amount of caffeine consumed in the 24 hours prior to bedtime (in milligrams).
    13. Alcohol Consumption: Amount of alcohol consumed in the 24 hours prior to bedtime (number of drinks).
    14. Smoking Status: Whether the test subject is a smoker (binary: yes/no).
    15. Exercise Frequency: How frequently the test subject exercises (numerical: number of hours per week).

The script is organised into the following steps:

1. **Data Loading**: Load the dataset from the specified directory.
2. **Missing Value Handling**: Impute missing values in relevant columns using the median.
3. **Time Data Preprocessing**: Convert bedtime and wake-up time into a format suitable for analysis.
4. **Categorical Encoding**: Encode categorical variables (e.g., gender, smoking status) using Label Encoding.
5. **Exploratory Data Analysis**:
    - Plot histograms to visualize the distribution of demographic, sleep, and lifestyle features.
    - Generate a correlation matrix to identify relationships between the variables.
6. **Modeling**:
    - Train a Random Forest Regressor model using selected features to predict sleep efficiency.
    - Evaluate the model's performance using metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R²).
    - Analyze the importance of features in predicting sleep efficiency.
'''

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Define the path to the dataset directory
data_directory = '../sleep_efficiency/data'

# Flag for whether to plot histograms or not
plot_histogram = True

# Load the dataset
df = pd.read_csv(os.path.join(data_directory, 'Sleep_Efficiency.csv'))

# Set display options to show all columns
pd.set_option('display.max.columns', None)

# Display column headers to understand the dataset structure
print(df.columns)
print(df.info())

# Data Preprocessing: Handle Missing Values
# Checking for missing values in the dataset
missing_values = df.isnull().sum()
print(f"Columns with missing values:\n{missing_values[missing_values > 0]}")

# Filling missing values with the median of the respective columns
df['Awakenings'] = df['Awakenings'].fillna(df['Awakenings'].median())
df['Caffeine consumption'] = df['Caffeine consumption'].fillna(df['Caffeine consumption'].median())
df['Alcohol consumption'] = df['Alcohol consumption'].fillna(df['Alcohol consumption'].median())
df['Exercise frequency'] = df['Exercise frequency'].fillna(df['Exercise frequency'].median())

print('Awakenings median:', df['Awakenings'].median())
print('Caffeine consumption median:', df['Caffeine consumption'].median())
print('Alcohol consumption median:', df['Alcohol consumption'].median())
print('Exercise frequency median:', df['Exercise frequency'].median())

# Statistical info
print(df.describe().T)

# Visualizing demographic features: Age and Gender distributions
if plot_histogram:
    demographic_features = ['Age', 'Gender']
    colors = ['skyblue', 'pastel']
    titles = ['Age Distribution', 'Gender Distribution']

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
    for i, feature in enumerate(demographic_features):
        if feature == 'Gender':
            sns.countplot(x=feature, data=df, ax=axes[i], hue=feature, palette=colors[i], edgecolor='k')
        else:
            sns.histplot(df[feature], bins=10, ax=axes[i], color=colors[i], kde=True)
        axes[i].set_title(titles[i])
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('demographic_features_histograms.png', format='png', dpi=300)
    plt.show()

    # Histogram of Sleep Patterns and Sleep Quality
    sleep_features = [
        'Sleep duration', 'Sleep efficiency', 'REM sleep percentage',
        'Deep sleep percentage', 'Light sleep percentage', 'Awakenings'
    ]
    colors = ['lightgreen', 'lightcoral', 'gold', 'dodgerblue', 'violet', 'orange']
    titles = [
        'Sleep Duration (hours)', 'Sleep Efficiency (proportion)',
        'REM Sleep Percentage (%)', 'Deep Sleep Percentage (%)',
        'Light Sleep Percentage (%)', 'Awakenings'
    ]

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
    for i, feature in enumerate(sleep_features):
        row, col = divmod(i, 3)
        if feature == 'Awakenings':
            sns.countplot(x=feature, data=df, ax=axes[row, col], color=colors[i], edgecolor='k', alpha=0.7)
        else:
            sns.histplot(df[feature], bins=10, ax=axes[row, col], color=colors[i], kde=True)
        axes[row, col].set_title(titles[i])
        axes[row, col].set_xlabel(feature)
        axes[row, col].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('sleep_features_histograms.png', format='png', dpi=300)
    plt.show()

    # Histogram of Lifestyle Factors
    lifestyle_features = ['Caffeine consumption', 'Alcohol consumption', 'Smoking status', 'Exercise frequency']
    colors = ['brown', 'red', 'gray', 'green']
    titles = [
        'Caffeine Consumption Distribution (in milligrams)', 'Alcohol Consumption Distribution (number of drinks)',
        'Smoking Status Distribution', 'Exercise Frequency Distribution (number of hours per week)'
    ]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 8))
    for i, feature in enumerate(lifestyle_features):
        row, col = divmod(i, 2)

        if feature == 'Caffeine consumption':
            sns.histplot(df[feature], bins=10, ax=axes[row, col], color=colors[i], kde=False)
        elif feature == 'Smoking status':
            sns.countplot(x=feature, data=df, ax=axes[row, col], hue=feature, palette=colors[i], edgecolor='k')
        else:
            sns.countplot(x=feature, data=df, ax=axes[row, col], color=colors[i], edgecolor='k', alpha=0.7)
        axes[row, col].set_title(titles[i])
        axes[row, col].set_xlabel(feature)
        axes[row, col].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('lifestyle_factors_histograms.png', format='png', dpi=300)
    plt.show()

# Convert 'Bedtime' and 'Wakeup time' to datetime format for better manipulation
df['Bedtime'] = pd.to_datetime(df['Bedtime'])
df['Wakeup time'] = pd.to_datetime(df['Wakeup time'])

# Extract hours and minutes from Bedtime and Wakeup time
df['Bedtime_Hour'] = df['Bedtime'].dt.hour
df['Bedtime_Minute'] = df['Bedtime'].dt.minute
df['Wakeup_Hour'] = df['Wakeup time'].dt.hour
df['Wakeup_Minute'] = df['Wakeup time'].dt.minute

# Convert Bedtime and Wakeup Time into minutes since midnight
df['Bedtime_Minutes'] = df['Bedtime_Hour'] * 60 + df['Bedtime_Minute']
df['Wakeup_Minutes'] = df['Wakeup_Hour'] * 60 + df['Wakeup_Minute']

# Adjust for overnight times (e.g., Bedtime after midnight)
# Adjust Bedtime to a consistent range (0 to 300 minutes for times after 20:00, otherwise add 1440 minutes)
df['Adjusted_Bedtime_Minutes'] = df['Bedtime_Minutes'].apply(lambda x: x if x >= 20 * 60 else x + 1440)
df['Adjusted_Wakeup_Minutes'] = df['Wakeup_Minutes'].apply(lambda x: x if x >= 20 * 60 else x + 1440)

# Define start and end times based on your data
bedtime_start_time = 1250  # Starting point for bedtime in minutes
bedtime_end_time = 1600  # Ending point for bedtime in minutes

wakeup_start_time = 1600  # Starting point for wakeup time in minutes
wakeup_end_time = 2200  # Ending point for wakeup time in minutes

# Set tick locations (e.g., every 30 minutes)
bedtime_ticks = range(bedtime_start_time, bedtime_end_time + 1, 30)
wakeup_ticks = range(wakeup_start_time, wakeup_end_time + 1, 30)

# Convert tick locations back to HH:MM format for labels
bedtime_labels = [(f"{(t // 60) % 24:02d}:{t % 60:02d}") for t in bedtime_ticks]
wakeup_labels = [(f"{(t // 60) % 24:02d}:{t % 60:02d}") for t in wakeup_ticks]

# Plot adjusted Bedtime and Wakeup times
if plot_histogram:
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 12))

    sns.histplot(df['Adjusted_Bedtime_Minutes'], bins=15, kde=True, color='skyblue', ax=axes[0])
    axes[0].set_title('Distribution of Adjusted Bedtime Minutes')
    axes[0].set_xlabel('Time (Minutes)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_xticks(bedtime_ticks)
    axes[0].set_xticklabels(bedtime_labels)

    sns.histplot(df['Adjusted_Wakeup_Minutes'], bins=20, kde=True, color='gold', ax=axes[1])
    axes[1].set_title('Distribution of Adjusted Wakeup Minutes')
    axes[1].set_xlabel('Time (Minutes)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_xticks(wakeup_ticks)
    axes[1].set_xticklabels(wakeup_labels)

    plt.tight_layout()
    plt.savefig('adjusted_bedtime_wakeup_histograms.png', format='png', dpi=300)
    plt.show()

# Encode categorical variables
# Gender Encoding
df['Gender_Female'] = (df['Gender'] == 'Female')
df['Gender_Male'] = (df['Gender'] == 'Male')
label_encoder_gender = LabelEncoder()
df['Gender_encoded'] = label_encoder_gender.fit_transform(df['Gender'])

# Print mapping of labels to numbers:
smoking_mapping = dict(zip(label_encoder_gender.classes_, label_encoder_gender.transform(label_encoder_gender.classes_)))
print("Gender encoding:", smoking_mapping)

# Smoking Status Encoding
label_encoder_smoking = LabelEncoder()
df['Smoking_status_encoded'] = label_encoder_smoking.fit_transform(df['Smoking status'])

# Print mapping of labels to numbers:
smoking_mapping = dict(zip(label_encoder_smoking.classes_, label_encoder_smoking.transform(label_encoder_smoking.classes_)))
print("Smoking status encoding:", smoking_mapping)

# Correlation Matrix
# Define the columns to include in the correlation matrix
columns_of_interest = ['Age', 'Gender_Female', 'Gender_Male', 'Caffeine consumption', 'Alcohol consumption',
                       'Smoking_status_encoded', 'Exercise frequency', 'Adjusted_Bedtime_Minutes',
                       'Adjusted_Wakeup_Minutes', 'Sleep duration', 'Awakenings', 'REM sleep percentage',
                       'Light sleep percentage', 'Deep sleep percentage', 'Sleep efficiency']

# Subset the DataFrame
df_subset = df[columns_of_interest]

# Compute and plot the correlation matrix
correlation_matrix = df_subset.corr()
# Create a mask for the upper triangle
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Print the correlation matrix
print(correlation_matrix)

# Plot the correlation matrix using a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Selected Features')
plt.tight_layout()
plt.savefig('correlation_matrix.png', format='png', dpi=300)
plt.show()

# Machine Learning: Random Forest Regressor
# Define the feature matrix (X) and the target variable (y)
X = df[['Age', 'Exercise frequency', 'Alcohol consumption', 'Smoking_status_encoded']]
y = df['Sleep efficiency']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the RandomForestRegressor model with 10 decision trees
model = RandomForestRegressor(n_estimators=10, random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Calculate and display regression metrics to evaluate model performance
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R^2): {r2:.4f}")

# Get numerical feature importance
feature_list = list(X.columns)
feature_importance = pd.Series(model.feature_importances_, index=feature_list).sort_values(ascending=False)
print(feature_importance)

# Mean Squared Error (MSE): 0.0094
# Mean Absolute Error (MAE): 0.0720
# R-squared (R^2): 0.4966

# Age                       0.366666
# Alcohol consumption       0.264334
# Exercise frequency        0.193076
# Smoking_status_encoded    0.175924
