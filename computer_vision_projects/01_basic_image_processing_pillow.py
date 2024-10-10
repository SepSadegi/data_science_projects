'''
This Python code provides a basic framework for processing images using the Pillow Library (PIL).
'''

import os
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# Helper function to concatenate two images horizontally (side-by-side)
def get_concat_h(im1, im2):
    """
      This function combines two images (im1 and im2) into a single image by placing them side-by-side.

      Args:
          im1 (PIL.Image): The first image to be concatenated.
          im2 (PIL.Image): The second image to be concatenated.

      Returns:
          PIL.Image: The combined image with the two images placed horizontally.

      References:
          https://note.nkmk.me/en/python-pillow-concat-images/ (for further details on image concatenation)
      """

    # Create a new RGB image with a width equal to the sum of both image widths and the same height as the original images
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))

    # Paste the first image onto the new image at (0, 0) coordinates
    dst.paste(im1, (0, 0))

    # Paste the second image onto the new image starting from the width of the first image (im1.width) at (0, 0) coordinates
    dst.paste(im2, (im1.width, 0))
    return dst


# Flags to control individual plotting sections
plot_image_loading = False
plot_grayscale_quantization = False
plot_color_channels = False
plot_pil_to_numpy = False
plot_indexing = False

# Image Files and Paths
path = os.getcwd()
lenna_image = "images/lenna.png"
lenna_image_path = os.path.join(path, lenna_image)
print(f"Image path: {lenna_image_path}")

# Load Images in Python
image = Image.open(lenna_image)
print(f"Image type: {type(image)}")
if plot_image_loading:
    image.show()

# Displays the image using matplotlib.pyplot
if plot_image_loading:
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.show()

# Print image information (using the attributes)
print(f'The image size is: {image.size}')
print(f'The image mode is: {image.mode}')

# The Image.open method does not load image data into the computer memory.
# The load method of PIL object reads the file content, decodes it, and expands the image into memory.

im = image.load()
# Checkin the intensity of the image at the x-th column and y-th row
x = 0
y = 1
print(f'Pixel value at (x={x}, y={y}): {im[y, x]}')

# Saving the image in jpg format
# image.save("lenna.jpg")

# Grayscale Images, Quantization, and Color Channels
image_gray = ImageOps.grayscale(image)
if plot_grayscale_quantization:
    image_gray.show()
print(f'The mode of the grey image is: {image_gray.mode}')

'''
Quantization

The Quantization of an image is the number of unique intensity values any given pixel
of the image can take. For a grayscale image, this means the number of different shades
of gray. Most images have 256 different levels. You can decrease the levels using the
method quantize.
'''
# Here we repeatably cut the number of levels in half and observe what happens:
image_gray_128 = image_gray.quantize(256 // 2)
if plot_grayscale_quantization:
    image_gray_128.show()

# Comaprison:
if plot_grayscale_quantization:
    get_concat_h(image_gray, image_gray.quantize(256 // 2)).show(title="Lena")

    for n in range(3,8):
        plt.figure(figsize=(10, 10))
        plt.imshow(get_concat_h(image_gray, image_gray.quantize(256 // 2** n)))
        plt.title(f'256 Quantization Levels left vs {256 // 2** n} Quantization Levels right')
        plt.show()

# Color Channels

baboon_image = "images/baboon.png"
baboon_image_path = os.path.join(path, baboon_image)
baboon = Image.open(baboon_image)

# Obtaining the different RGB color channels and assign them to the variables red, green, and blue
red, green, blue = baboon.split()

# Plotting the color image next to the RGB color channels as a grayscale
if plot_color_channels:
    get_concat_h(baboon, red).show(title="Baboon R")
    get_concat_h(baboon, green).show(title="Baboon G")
    get_concat_h(baboon, blue).show(title="Baboon B")

# Convert PIL Images into NumPy Arrays
'''
np.asarray turns the original image into a numpy array. if we don't want to manipulate 
the image directly, but instead, create a copy of the image to manipulate, we use the 
np.array method creates a new copy of the image, such that the original one will remain unmodified.

The attribute shape of a numpy.array object returns a tuple corresponding to the dimensions of it,
the first element gives the number of rows or height of the image, the second is element is the
number of columns or width of the image. The final element is the number of colour channels.
'''
array = np.asarray(image)
print(f"Array type: {type(array)}")
print(f"Array shape: {array.shape}")
print(array)
print(f'The intensity values: {array[0,0]}') # The intensity values are 8-bit unsigned datatype
print(f"Minimum pixel value: {array.min()}")
print(f"Maximum pixel value: {array.max()}")

# Indexing
# Plotting the array as an image
if plot_pil_to_numpy:
    plt.figure(figsize=(10, 10))
    plt.imshow(array)
    plt.show()

# We can use numpy slicing to return first 256 rows, corresponding to the top half of the image
rows = 256
if plot_indexing:
    plt.figure(figsize=(10, 10))
    plt.imshow(array[0:rows, :, :])
    plt.show()

# Returning the first 256 columns, corresponding to the first half of the image
columns = 256
if plot_indexing:
    plt.figure(figsize=(10, 10))
    plt.imshow(array[:, 0:columns, :])
    plt.show()

# To reassign an array to another variable, we use copy method
A = array.copy()
if plot_indexing:
    plt.imshow(A)
    plt.show()

#If we do not apply the method copy(), the variable will point to the same
# location in memory. Consider the array B. If we set all values of array A
# to zero, as B points to A, the values of B will be zero too:

B = A
A[:, :, :] = 0
if plot_indexing:
    plt.imshow(B)
    plt.show()

# Same can be applied on different color channels
baboon_array = np.array(baboon)
if plot_color_channels:
    plt.figure(figsize=(10, 10))
    plt.imshow(baboon_array[:,:,0], cmap='gray')
    plt.show()

# Create a new array and set all but the red channels to zero
baboon_red = baboon_array.copy()
baboon_red[:, :, 1] = 0
baboon_red[:, :, 2] = 0
if plot_color_channels:
    plt.figure(figsize=(10, 10))
    plt.imshow(baboon_red)
    plt.show()

# Same but for blue
baboon_blue = baboon_array.copy()
baboon_blue[:, :, 0] = 0
baboon_blue[:, :, 1] = 0
if plot_color_channels:
    plt.figure(figsize=(10, 10))
    plt.imshow(baboon_blue)
    plt.show()

# Baboon omit blue
baboon_omit_blue = baboon_array.copy()
baboon_omit_blue[:, :, 2] = 0
if plot_color_channels:
    plt.figure(figsize=(10, 10))
    plt.imshow(baboon_omit_blue)
    plt.show()

# Process the lenna image to get blue channel out of it
lenna = Image.open('images/lenna.png')
lenna_array = np.array(lenna)
lenna_omit_blue = lenna_array.copy()
lenna_omit_blue[:, :, 2] = 0
plt.figure(figsize=(10, 10))
plt.imshow(lenna_omit_blue)
plt.show()
