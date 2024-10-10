"""
This script practices pixel transformations using OpenCV, including:

- Creating histograms to display image intensity and optimize image characteristics.
- Performing intensity transformations to enhance image contrast and brightness.
- Using thresholding to segment objects from images.

Only OpenCV is used.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

# Flags to control plotting sections
plot_histograms = True

def plot_image_comparison(image1, image2, title1="Original", title2="Transformed"):
    """Plot two images side by side."""
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image1, cmap='gray')
    plt.title(title1)
    plt.subplot(1, 2, 2)
    plt.imshow(image2, cmap='gray')
    plt.title(title2)
    plt.show()

def plot_histogram_comparison(old_image, new_image, title_old="Original", title_new="Transformed"):
    """Plot histograms of two images for comparison."""
    intensity_values = np.arange(256)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.bar(intensity_values, cv2.calcHist([old_image], [0], None, [256], [0, 256]).flatten(), width=1)
    plt.title(title_old)
    plt.xlabel('Intensity')
    plt.subplot(1, 2, 2)
    plt.bar(intensity_values, cv2.calcHist([new_image], [0], None, [256], [0, 256]).flatten(), width=1)
    plt.title(title_new)
    plt.xlabel('Intensity')
    plt.show()

# Sample Image and Histogram Analysis
toy_image = np.array([[0, 2, 2], [1, 1, 1], [1, 1, 2]], dtype=np.uint8)
print("toy_image:\n", toy_image)

if plot_histograms:
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(toy_image, cmap="gray")
    plt.title("Toy Image")

    plt.subplot(1, 3, 2)
    plt.bar(range(6), [1, 5, 2, 0, 0, 0])
    plt.title("Histogram r")
    plt.xlabel("Pixel Values")
    plt.ylabel("Occurrences")

    plt.subplot(1, 3, 3)
    plt.bar(range(6), [0, 1, 0, 5, 0, 2])
    plt.title("Transform r to s: s=2r+1")
    plt.xlabel("Intensity")
    plt.ylabel("Counts")
    plt.show()

# Gray Scale Histogram Analysis
goldhill = cv2.imread("images/goldhill.bmp", cv2.IMREAD_GRAYSCALE)
hist = cv2.calcHist([goldhill], [0], None, [256], [0, 256])
intensity_values = np.arange(hist.shape[0])
PMF = hist / (goldhill.size)

if plot_histograms:
    plt.figure(figsize=(21, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(goldhill, cmap="gray")
    plt.title("Goldhill Image")

    plt.subplot(1, 3, 2)
    plt.bar(intensity_values, hist.flatten(), width=1)
    plt.title("Histogram (Bar)")

    plt.subplot(1, 3, 3)
    plt.plot(intensity_values, hist)
    plt.title("Histogram (Line)")
    plt.show()

# Color Image Histogram Analysis
baboon = cv2.imread("images/baboon.png")

if plot_histograms:
    colors = ('blue', 'green', 'red')
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(baboon, cv2.COLOR_BGR2RGB))
    plt.title("Baboon Image")

    plt.subplot(1, 2, 2)
    for i, color in enumerate(colors):
        hist = cv2.calcHist([baboon], [i], None, [256], [0, 256])
        plt.plot(intensity_values, hist, color=color, label=f'{color} channel')
    plt.title("Histogram Channels")
    plt.xlim([0, 256])
    plt.legend()
    plt.show()
