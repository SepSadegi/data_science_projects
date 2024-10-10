import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


datapath = '../data/'

# Each row belongs to one image with a size of 8x8x3 (width x hight x color channel) + one label for the class target
low_res_color = pd.read_csv(os.path.join(datapath, 'hmnist_8_8_RGB.csv')) # 5000 rows x 193 columns
# print(low_res_color.head())
print("Low resolution colored data: ", low_res_color.shape)

# 8*8*1 + label
low_res_grey = pd.read_csv(os.path.join(datapath, 'hmnist_8_8_L.csv')) # 5000 rows x 65 columns
# print(low_res_grey.head())
print("Low resolution grayscale data: ",low_res_grey.shape)

# 28x28x3 + label
medium_res_color = pd.read_csv(os.path.join(datapath, 'hmnist_28_28_RGB.csv')) # 5000 rows x 2353 columns
# print(medium_res_color.head())
print("Medium resolution colored data: ", medium_res_color.shape)

# 28x28x1 + label
medium_res_grey = pd.read_csv(os.path.join(datapath, 'hmnist_28_28_L.csv')) # 5000 rows x 785 columns
# print(medium_res_grey.head())
print("Medium resolution grayscale data: ", medium_res_grey.shape)

# 64x64x1 + label
high_res_grey = pd.read_csv(os.path.join(datapath, 'hmnist_64_64_L.csv')) # 5000 rows x 4097 columns
# print(high_res_grey.head())
print("High resolution grayscale data: ", high_res_grey.shape)

# How are the pixels ordered?
# The following plotting shows that the pixels order is: color, width, height.

# Plotting the first image
# Flags to control individual plotting sections
plot_images = True

# Drop the 'label' column to isolate pixel data
image_pixels = medium_res_color.drop("label", axis=1).values[0] # Get the vallues of the first image

# Reshape the flat array of pixel data into a 28x28x3 image
reshaped_image = image_pixels.reshape((28, 28, 3))

# Display the image
if plot_images:
    plt.imshow(reshaped_image)
    plt.title('First Image in Dataset')
    plt.show()

# Display the three color channels next to the main image
if plot_images:
    fig, axes = plt.subplots(1, 4, figsize=(21, 6))
    channel_label = ['Red', 'Green', 'Blue']
    cmaps = ['Reds_r', 'Greens_r', 'Blues_r']
    # Plot color channels
    for channel in range(3):
        axes[channel].imshow(reshaped_image[:,:,channel], cmap=cmaps[channel])
        axes[channel].set_title(f"Channel {channel_label[channel]}")
        axes[channel].set_xlabel("Width")

    axes[0].set_ylabel("Height")
    # Plot the original image
    axes[3].imshow(reshaped_image)
    axes[3].set_title("All channels together")
    axes[3].set_xlabel("Width")
    plt.tight_layout()
    # plt.savefig('firs_image_example.jpg', format='jpg', dpi=300)
    plt.show()


def extract_and_reshape(data, color=True, resolution=(28,28)):
    image_pixels = data.drop("label", axis=1).values[0]
    if color:
        return image_pixels.reshape((*resolution, 3))
    else:
        return image_pixels.reshape(resolution)

# Comparing first image of each data set
image1 = extract_and_reshape(low_res_grey, color=False, resolution=(8,8))
image2 = extract_and_reshape(low_res_color, color=True, resolution=(8,8))
image3 = extract_and_reshape(medium_res_grey, color=False, resolution=(28,28))
image4 = extract_and_reshape(medium_res_color, color=True, resolution=(28,28))
image5 = extract_and_reshape(high_res_grey, color=False, resolution=(64,64))

if plot_images:
    image_names = ['image1', 'image2', 'image3', 'image4', 'image5']
    titles = ['Grayscale 8x8', 'Color 8x8', 'Grayscale 28x28', 'Color 28x28', 'Grayscale 64x64']
    cmaps = ['gray', None, 'gray', None, 'gray']
    fig, axes = plt.subplots(1, 5, figsize=(21, 6))
    for i in range(5):
        image = globals()[image_names[i]]
        axes[i].imshow(image, cmap=cmaps[i])
        axes[i].set_title(titles[i])
        axes[i].set_xlabel("Width")
        axes[i].set_ylabel("Height")
    plt.tight_layout()
    plt.savefig('firs_image_different_resolution.jpg', format='jpg', dpi=300)
    plt.show()