"""
The data is taken from the public ChestX-ray8 dataset
(https://arxiv.org/abs/1705.02315).
The dataset was downloaded from: https://www.kaggle.com/datasets/truptipillai/nih-chest-xray-dataset
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configure matplotlib
mpl.rc('xtick', labelsize=16)
mpl.rc('ytick', labelsize=16)
mpl.rcParams.update({'font.size': 15})
plt.ioff()

# Paths
datapath = '../chest_X-ray_deep_learning'
image_dir = os.path.join(datapath, 'images-small/')

# Flags for plotting sections
plot_images = False
plot_charts_histograms = True
plot_processed_image = False

# Load training data
train_df = pd.read_csv(os.path.join(datapath, 'train-small.csv'))
print(f'There are {train_df.shape[0]} rows and {train_df.shape[1]} columns in this data frame')
print(train_df.head())

# Data types and null values
print(train_df.info())

# Check for unique patient IDs
# In medical dataset it is important to check whether there are repeated data
# for certain patients or each image represent a different person.
# It is important to check that patients with multiple records do not show up in both training and test sets
# Otherwise we will have data leakage
total_patient_ids = train_df['PatientId'].count()
unique_patient_ids = train_df['PatientId'].nunique()
print(f"The total patient IDs are {total_patient_ids}, with {unique_patient_ids} unique IDs")
# There must be some overlap on IDs.

# Data Labels
columns = [col for col in train_df.columns if col not in ['Image', 'PatientId']]
print(f"There are {len(columns)} columns of labels for these conditions: {columns}")

# Count and display positive labels (classes have 0 and 1 values)
positive_counts = [train_df[col].sum() for col in columns]
total_counts = sum(positive_counts)
sorted_counts, sorted_columns = zip(*sorted(zip(positive_counts, columns), reverse=True))
positive_percentage = [count / total_counts * 100 for count in sorted_counts]

# Printing out the number of positive labels for each class
for col in columns:
    print(f"The class {col} has {train_df[col].sum()} samples.")

# Plot positive diagnosis distribution
if plot_charts_histograms:
    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap("tab20")
    plt.pie(positive_percentage, labels=sorted_columns, autopct='%1.1f%%', startangle=0, colors=cmap(np.arange(len(columns))))
    plt.title('Distribution of Positive Diagnoses', fontweight='bold')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('distribution_of_positive_diagnoses.jpg')
    plt.show()

# Sample random images
images = train_df['Image'].values
random_images = np.random.choice(images, size=9, replace=False)

# Plot random images
if plot_images:
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for i, ax in enumerate(axes.flatten()): # Flatten the 3x3 array to easily iterate over it
        img_path = os.path.join(image_dir, random_images[i])
        img = plt.imread(img_path)
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    # plt.savefig('random_chestX-ray_images.jpg', format='jpg', dpi=300)
    plt.show()

# Investigate a single image
sample_img_path = os.path.join(image_dir, train_df.Image.iloc[0])
raw_image = plt.imread(sample_img_path)

print(f"The image dimensions are {raw_image.shape[0]}x{raw_image.shape[1]} pixels, with one color channel.")
print(f"Maximum pixel value: {raw_image.max():.4f}")
print(f"Minimum pixel value: {raw_image.min():.4f}")
print(f"Mean pixel value: {raw_image.mean():.4f}")
print(f"Standard deviation: {raw_image.std():.4f}")

if plot_images:
    plt.imshow(raw_image, cmap='gray')
    plt.colorbar()
    plt.title('Raw Chest X Ray Image')
    plt.show()

# Plot pixel value distribution
if plot_charts_histograms:
    plt.figure(figsize=(10, 7))
    sns.histplot(raw_image.ravel(), color='blue', kde=False,
                 label=f'Original Image: mean {raw_image.mean():.4f}, std {raw_image.std():.4f}\n'
                       f'Min {raw_image.min():.4f}, Max {raw_image.max():.4f}')
    plt.title("Pixel Intensity Distribution in Raw Image")
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig('pixel_intensity_distribution_raw_image.jpg', format='jpg', dpi=300)
    plt.show()

# Image preprocessing with Keras
# Normalise Image:
# The generator will replace each pixel value in the image with a new value calculated by subtracting
# the mean and divided by the standard deviation
image_generator = ImageDataGenerator(
    samplewise_center=True, #Set each sample mean to 0.
    samplewise_std_normalization=True # Divide each input by its standard deviation
)

# Flow from directory with specified batch size and target image size
generator = image_generator.flow_from_dataframe(
    dataframe=train_df,
    directory=image_dir,
    x_col="Image",    # features
    y_col=['Mass'],   # label # Building a model for mass detection
    class_mode="raw", # 'Mass' column should be in train_df
    batch_size=1,     # image per batch
    shuffle=False,    # shuffle the rows or not
    target_size=(320, 320)
)

# Plot raw vs preprocessed image
# generated_image, label = generator.__getitem__(0)
generated_image, _ = generator[0] # Choose image to plot

if plot_processed_image:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    im1 = axes[0].imshow(raw_image, cmap='gray')
    axes[0].set_title('Raw Chest X-Ray Image')
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(generated_image[0], cmap='gray')
    axes[1].set_title('Preprocessed Chest X-Ray Image')
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig('chestX-ray_image_raw_vs_preprocessed.jpg', format='jpg', dpi=300)
    plt.show()

print(f"The preprocessed image dimensions are {generated_image.shape[1]}x{generated_image.shape[2]} pixels, with one color channel.")
print(f"Maximum pixel value: {generated_image.max():.4f}")
print(f"Minimum pixel value: {generated_image.min():.4f}")
print(f"Mean pixel value: {generated_image.mean():.4f}")
print(f"Standard deviation: {generated_image.std():.4f}")

# Plot histogram for original vs preprocessed images
if plot_charts_histograms:
    plt.figure(figsize=(10, 7))
    sns.histplot(raw_image.ravel(), color='blue', kde=False,
                 label=f'Original Image: mean {raw_image.mean():.4f}, std {raw_image.std():.4f}\n'
                       f'Min {raw_image.min():.4f}, Max {raw_image.max():.4f}')
    sns.histplot(generated_image[0].ravel(), color='red', kde=False, alpha=0.5,
                label=f'Preprocessed Image: mean {generated_image[0].mean():.4f}, std {generated_image[0].std():.4f}\n'
                      f'Min {generated_image[0].min():.4f}, Max {generated_image[0].max():.4f}')
    plt.title("Pixel Intensity Distribution in Raw vs Preprocessed Image")
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig('pixel_intensity_distribution_raw_vs_preprocessed_image.jpg', format='jpg', dpi=300)
    plt.show()