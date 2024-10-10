'''
This Python code provides a basic framework for processing images using the OpenCV Library.
'''

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Flags to control individual plotting sections
plot_image_loading = False
plot_grayscale_quantization = False
plot_color_channels = False
plot_indexing = False

# Image Files and Paths
path = os.getcwd()
lenna_image = "images/lenna.png"
lenna_image_path = os.path.join(path, lenna_image)
print(f"Image path: {lenna_image_path}")

# Load Images in Python
image = cv2.imread(lenna_image_path)
print(f"Image type: {type(image)}")

# Print image information (using the attributes)
print(f'The image shape is: {image.shape}')
print(f"Minimum pixel value: {image.min()}")
print(f"Maximum pixel value: {image.max()}")

# Plotting the Image
if plot_image_loading:
    cv2.imshow('image', image)
    cv2.waitKey(0) #Wait for a key press
    cv2.destroyAllWindows() #Close the window

# To Display the image using matplotlib.pyplot we should change the order of BGR
if plot_image_loading:
    new_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(new_image)
    plt.show()

# Saving the image in jpg format
# cv2.imwrite("images/lenna2.jpg", image)

# Grayscale Images, Quantization, and Color Channels
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(f'The image shape is: {image_gray.shape}')
if plot_grayscale_quantization:
    cv2.imshow('image_gray', image_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# Saving the image in jpg format
# cv2.imwrite("images/lenna_gray_cv.jpg", image_gray)

# Load Grayscale image
barbara_gray_image = cv2.imread('images/barbara.png', cv2.IMREAD_GRAYSCALE)
if plot_grayscale_quantization:
    cv2.imshow('barbara_gray_image', barbara_gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Color Channels
baboon = cv2.imread('images/baboon.png')
if plot_color_channels:
    cv2.imshow('baboon', baboon)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Obtaining the different RGB color channels and assign them to the variables red, green, and blue
blue, green, red = baboon[:, :, 0], baboon[:, :, 1], baboon[:, :, 2]

im_brg = cv2.vconcat([blue, green, red])
if plot_color_channels:
    cv2.imshow('im_brg', im_brg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Indexing
# Returning the first 256 rows, corresponding to the top half of the image
rows = 256
image_new_r = image[0:rows, :, :]
if plot_indexing:
    cv2.imshow('image_new_r', image_new_r)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Returning the first 256 columns, corresponding to the first half of the image
columns= 256
image_new_c = image[:, 0:columns, :]
if plot_indexing:
    cv2.imshow('image_new_c', image_new_c)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# To reassign an array to another variable, we use copy method
A = image.copy()
if plot_indexing:
    cv2.imshow('A', A)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Without usin copy(), the variable will point to the same location in memory.
B = A
A[:,:,:] = 0
if plot_indexing:
    cv2.imshow('B', B)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Manipulate elements using indexing
baboon_red = baboon.copy()
baboon_red[:, :, 0] = 0
baboon_red[:, :, 1] = 0
if plot_color_channels:
    cv2.imshow('baboon_red', baboon_red)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Same but for blue
baboon_blue = baboon.copy()
baboon_blue[:, :, 1] = 0
baboon_blue[:, :, 2] = 0
if plot_color_channels:
    cv2.imshow('baboon_blue', baboon_blue)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Same but for green
baboon_green = baboon.copy()
baboon_green[:, :, 0] = 0
baboon_green[:, :, 2] = 0
if plot_color_channels:
    cv2.imshow('baboon_green', baboon_green)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Task: Open the baboon image, create an OpenCV object 'baboon_image', convert BGR to RGB, extract the blue channel, and plot the result.
baboon_image = cv2.imread('baboon.png')
baboon_image = cv2.cvtColor(baboon_image, cv2.COLOR_BGR2RGB)
baboon_image[:, :, 0] = 0
plt.figure(figsize=(10,10))
plt.imshow(baboon_image)
plt.show()