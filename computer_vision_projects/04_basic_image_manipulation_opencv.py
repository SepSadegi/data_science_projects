'''
Manipulate images using OpenCV, focusing on:
    - Copying images (to avoid aliasing),
    - Efficient image flipping using `flip()`,
    - Rotating images with `rotate()`,
    - Cropping images,
    - Changing specific image pixels,
    - Adding text and shapes to images.
'''

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Flags to control individual plotting sections
plot_images_copying = False
plot_images_flipping = False
plot_images_rotating = False
plot_images_cropping = False
plot_pixel_altering = True

# Image Files and Paths
path = os.getcwd()
baboon_image_path = "images/baboon.png"
cat_image_path = "images/cat.png"
baboon = cv2.imread(baboon_image_path)
cat = cv2.imread(cat_image_path)

if plot_images_copying:
    plt.figure(figsize=(5, 5))
    plt.imshow(cv2.cvtColor(baboon, cv2.COLOR_BGR2RGB))
    plt.show()

# Copying Images
if plot_images_copying:
    A = baboon
    B = baboon.copy()
    baboon[:, :, :] = 0  # Modify the original image

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(cv2.cvtColor(baboon, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Baboon")
    axes[1].imshow(cv2.cvtColor(A, cv2.COLOR_BGR2RGB))
    axes[1].set_title("A (Same as Baboon)")
    axes[2].imshow(cv2.cvtColor(B, cv2.COLOR_BGR2RGB))
    axes[2].set_title("B (Copied)")
    plt.tight_layout()
    plt.show()

# Flipping Images using OpenCV's `flip()`
# The flipCode value indicating what kind of flip we would like to perform;
# flipCode = 0: flip vertically around the x-axis
# flipCode > 0: flip horizontally around y-axis positive value
# flipCode < 0: flip vertically and horizontally, flipping around both axes negative value
if plot_images_flipping:
    # Original image
    plt.figure(figsize=(5, 5))
    plt.imshow(cv2.cvtColor(cat, cv2.COLOR_BGR2RGB))
    plt.title("Original image")
    plt.show()
    # Flipped image
    for flipcode in [0, 1, -1]:  # 0: vertical, 1: horizontal, -1: both
        im_flip = cv2.flip(cat, flipcode)
        plt.imshow(cv2.cvtColor(im_flip, cv2.COLOR_BGR2RGB))
        plt.title(f"flipCode: {flipcode}")
        plt.show()

# Rotating Images using OpenCV's `rotate()`; The parameter is an integer indicating what kind of flip we would like to perform.
print(f"Rotate code for 90 Clockwise: {cv2.ROTATE_90_CLOCKWISE}")
print(f"Rotate code for 90 Counterclockwise: {cv2.ROTATE_90_COUNTERCLOCKWISE}")
print(f"Rotate code for 180: {cv2.ROTATE_180}")

if plot_images_rotating:
    for rotate_code, name in [(cv2.ROTATE_90_CLOCKWISE, "90 Clockwise"),
                              (cv2.ROTATE_90_COUNTERCLOCKWISE, "90 Counterclockwise"),
                              (cv2.ROTATE_180, "180")]: # Built-in attributes in OpenCV
        im_rotate = cv2.rotate(cat, rotate_code)
        plt.imshow(cv2.cvtColor(im_rotate, cv2.COLOR_BGR2RGB))
        plt.title(f"Rotated {name}")
        plt.show()

# Cropping an Image using array slicing
upper, lower = 150, 400
left, right = 150, 400
cat_crop_top = cat[upper:lower, :, :]
if plot_images_cropping:
    plt.figure(figsize=(5, 5))
    plt.imshow(cv2.cvtColor(cat_crop_top, cv2.COLOR_BGR2RGB))
    plt.title("Crop Top")
    plt.show()

cat_crop_horizontal = cat_crop_top[:, left:right, :]
if plot_images_cropping:
    plt.figure(figsize=(5, 5))
    plt.imshow(cv2.cvtColor(cat_crop_horizontal, cv2.COLOR_BGR2RGB))
    plt.title("Cropped Image")
    plt.show()

# Changing Specific Image Pixels with OpenCV
if plot_pixel_altering:
    cat_copy = cat.copy()
    # Draw a rectangle
    start_point, end_point = (left, upper), (right, lower)
    cv2.rectangle(cat_copy, pt1=start_point, pt2=end_point, color=(0, 255, 0), thickness=3)

    # Add text to the image
    cat_text = cv2.putText(img=cat.copy(), text='Stuff', org=(10, 500), color=(255, 255, 255),
                           fontFace=4, fontScale=5, thickness=2)

    # Plot altered images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(cat_copy, cv2.COLOR_BGR2RGB))
    plt.title("Rectangle Drawn")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(cat_text, cv2.COLOR_BGR2RGB))
    plt.title("Text Added")
    plt.show()


# Exercise with baboon image
im = cv2.imread(baboon_image_path)
im_flip = cv2.flip(im, flipCode=0) # flip im vertically around the x-axis
im_mirror = cv2.flip(im, flipCode=1) # mirror im by flipping it horizontally around the y-axis
plt.figure(figsize=(15, 5))
plt.subplot(1,3,1)
plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.subplot(1,3,2)
plt.imshow(cv2.cvtColor(im_flip, cv2.COLOR_BGR2RGB))
plt.title("Flipped Vertically")
plt.subplot(1,3,3)
plt.imshow(cv2.cvtColor(im_mirror, cv2.COLOR_BGR2RGB))
plt.title("Flipped Horizontally")
plt.show()
