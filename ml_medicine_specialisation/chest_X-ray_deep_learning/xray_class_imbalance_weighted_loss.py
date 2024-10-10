'''
To avoid having class imbalance impact the loss function, we use weighted losses.
This practice helps in understanding how to calculate and apply class weights.
In this code, we calculate class frequencies and apply weights to the loss function.

Key learning points:
- Calculating loss for multiple classes.
- Avoiding log of zero by adding a small number (covered in the assignment).

On real data, it is possible to:
- Take the average loss for all examples (instead of the sum).
- Use TensorFlow equivalents of numpy operations.
'''

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

# Configure matplotlib settings
mpl.rc('xtick', labelsize=16)
mpl.rc('ytick', labelsize=16)
mpl.rcParams.update({'font.size': 15})
plt.ioff()

# Define paths
datapath = '../chest_X-ray_deep_learning'

# Flag for plotting sections
plot_distributions = False

# Load training data
train_df = pd.read_csv(os.path.join(datapath, 'train-small.csv'))

# Count the number of instances for each class (excluding non-class columns)
class_counts = train_df.sum().drop(['Image', 'PatientId'])

# Print the number of samples for each class
for col in class_counts.keys():
    print(f"The class {col} has {train_df[col].sum()} samples.")

# Plot class distribution if the flag is set
if plot_distributions:
    plt.figure(figsize=(14.4, 8))
    plt.barh(class_counts.index, class_counts.values, color='royalblue')
    plt.title('Class Distribution of Chest X-Ray Images in Training Dataset', fontsize=15)
    plt.xlabel('Number of Patients', fontsize=15)
    plt.ylabel('Diseases', fontsize=15)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('class_distribution_xray_training_data.jpg', format='jpg', dpi=300)
    plt.show()

# Weighted Loss Function Practice
# Generate 'ground truth' labels (an array of 4 binary label values, 3 positive and 1 negative)
y_true = np.array(
    [[1],
     [1],
     [1],
     [0]]
)
print(f"y_true: \n{y_true}")

# Understanding Weighted Loss Function
# Module 1 always outputs 0.9 for any example
# Module 2 always outputs 0.1 for any example
# Generate module predictions
y_pred_1 = 0.9 * np.ones(y_true.shape)
print(f"y_pred_1: \n{y_pred_1}\n")

y_pred_2 = 0.1 * np.ones(y_true.shape)
print(f"y_pred_2: \n{y_pred_2}")

# Regular Loss Function
# Calculate loss for Model 1 and Model 2 without weights
loss_reg_1 = -1 * np.sum(y_true * np.log(y_pred_1)) + \
             -1 * np.sum((1 - y_true) * np.log(1 - y_pred_1))
print(f"loss_reg_1: {loss_reg_1:.4f}")

loss_reg_2 = -1 * np.sum(y_true * np.log(y_pred_2)) + \
             -1 * np.sum((1 - y_true) * np.log(1 - y_pred_2))
print(f"loss_reg_2: {loss_reg_2:.4f}")

print(f"When the model 1 always predicts 0.9, the regular loss is {loss_reg_1:.4f}")
print(f"When the model 2 always predicts 0.1, the regular loss is {loss_reg_2:.4f}")

# Weighted Loss Function
# Calculate weights based on class frequencies
w_p = 1 / 4  # weight for positive class
w_n = 3 / 4  # weight for negative class

print(f"positive weight w_p: {w_p}")
print(f"negative weight w_n {w_n}")

# Weighted Loss: Model 1
# Calculate weighted loss for positive and negative predictions
loss_1_pos = -1 * np.sum(w_p * y_true * np.log(y_pred_1))
print(f"loss_1_pos: {loss_1_pos:.4f}")

loss_1_neg = -1 * np.sum(w_n * (1 - y_true) * np.log(1 - y_pred_1))
print(f"loss_1_neg: {loss_1_neg:.4f}")

# Total weighted loss for Model 1
loss_1 = loss_1_pos + loss_1_neg
print(f"loss_1: {loss_1:.4f}\n")

# Weighted Loss: Model 2
# Calculate weighted loss for positive and negative predictions
loss_2_pos = -1 * np.sum(w_p * y_true * np.log(y_pred_2))
print(f"loss_2_pos: {loss_2_pos:.4f}")

loss_2_neg = -1 * np.sum(w_n * (1 - y_true) * np.log(1 - y_pred_2))
print(f"loss_2_neg: {loss_2_neg:.4f}")

# Total weighted loss for Model 2
loss_2 = loss_2_pos + loss_2_neg
print(f"loss_2: {loss_2:.4f}")

# Comparing Model 1 and Model 2 Weighted Losses
print(f"When the model always predicts 0.9, the total weighted loss is {loss_1:.4f}")
print(f"When the model always predicts 0.1, the total weighted loss is {loss_2:.4f}")

print(f"loss_1_pos: {loss_1_pos:.4f} \t loss_1_neg: {loss_1_neg:.4f}\n")
print(f"loss_2_pos: {loss_2_pos:.4f} \t loss_2_neg: {loss_2_neg:.4f}")

# Multi-Class Weighted Loss Practice
# Generate 'ground truth' labels for multiple classes
y_true = np.array(
    [[1, 0],
     [1, 0],
     [1, 0],
     [1, 0],
     [0, 1]]
)
print(f"y_true: \n{y_true}")

# The difference between axis=0 or axis=1
print(f"Using axis = 0 {np.sum(y_true, axis=0)}")  # sum is taken for each two columns
print(f"Using axis = 1 {np.sum(y_true, axis=1)}")  # sum is taken for each row

# Calculate class weights for multi-class labels
w_p = np.sum(y_true == 0, axis=0) / y_true.shape[0]  # boolean array with elements = True if y_true == 0
print(f"Class weights (positive): {w_p}")

w_n = np.sum(y_true == 1, axis=0) / y_true.shape[0]  # boolean array with elements = True if y_true == 1
print(f"Class weights (negative): {w_n}")

# Set model predictions where all predictions are the same
y_pred = np.ones(y_true.shape)
y_pred[:, 0] = 0.3 * y_pred[:, 0]
y_pred[:, 1] = 0.7 * y_pred[:, 1]
print(f"y_pred: \n{y_pred}")

# Calculate the weighted loss for class 0
print(f"w_p[0]: {w_p[0]}")
print(f"w_n[0]: {w_n[0]}")
print(f"y_true[:,0]: \n{y_true[:,0]}")
print(f"y_pred[:,0]: \n{y_pred[:,0]}")

loss_0_pos = -1 * np.sum(w_p[0] * y_true[:, 0] * np.log(y_pred[:, 0]))
print(f"loss_0_pos: {loss_0_pos:.4f}")

loss_0_neg = -1 * np.sum(w_n[0] * (1 - y_true[:, 0]) * np.log(1 - y_pred[:, 0]))
print(f"loss_0_neg: {loss_0_neg:.4f}")

# Total loss for class 0
loss_0 = loss_0_pos + loss_0_neg
print(f"loss_0: {loss_0:.4f}")

# Calculate the weighted loss for class 1
print(f"w_p[1]: {w_p[1]}")
print(f"w_n[1]: {w_n[1]}")
print(f"y_true[:,1]: \n{y_true[:,1]}")
print(f"y_pred[:,1]: \n{y_pred[:,1]}")

loss_1_pos = -1 * np.sum(w_p[1] * y_true[:, 1] * np.log(y_pred[:, 1]))
print(f"loss_1_pos: {loss_1_pos:.4f}")

loss_1_neg = -1 * np.sum(w_n[1] * (1 - y_true[:, 1]) * np.log(1 - y_pred[:, 1]))
print(f"loss_1_neg: {loss_1_neg:.4f}")

# Total loss for class 1
loss_1 = loss_1_pos + loss_1_neg
print(f"loss_1: {loss_1:.4f}")