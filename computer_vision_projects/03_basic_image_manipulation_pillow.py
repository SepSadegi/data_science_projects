'''
Manipulate images as both NumPy arrays and PIL image objects with a focus on:
    - Copying images to avoid aliasing,
    - Flipping images,
    - Cropping images,
    - Changing specific image pixels,
    - Overlaying images.
'''

import os
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# Flags to control individual plotting sections
plot_images_copying = False
plot_images_flipping = False
plot_parameter_values = False
plot_cropping = False
plot_pixel_altering = True

# Image Files and Paths
path = os.getcwd() # Get current working directory
baboon_image_path = "images/baboon.png"
cat_image_path = "images/cat.png"

print(f"Image path: {baboon_image_path}")

# Load Baboon Image
baboon_image = Image.open(baboon_image_path)
baboon = np.array(baboon_image)

# Display the baboon image
if plot_images_copying:
    plt.figure(figsize=(5, 5))
    plt.imshow(baboon)
    plt.show()

# Understanding Image Copying Behavior
# Without applying copy(), the two variables take the same location in memory
A = baboon  # Assigning without copy
B = baboon.copy()  # Proper copy with new memory allocation
print(id(baboon)==id(A)) # A points to the same memory location as baboon
print(id(baboon)==id(B)) # B is a separate copy

# By setting array of baboon to zero, all entries of A also becomes zero but B will not be affected
baboon[:,:,] = 0

if plot_images_copying:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(baboon)
    axes[0].set_title("Baboon")
    axes[1].imshow(A)
    axes[1].set_title("A")
    axes[2].imshow(B)
    axes[2].set_title("B")
    plt.tight_layout()
    plt.show()

# Image flipping operations
cat_image = Image.open(cat_image_path)
cat = np.array(cat_image)
cat_width, cat_height, cat_C = cat.shape
print("Cat image width, height, Channels: ", cat_width, cat_height, cat_C)

if plot_images_flipping:
    plt.figure(figsize=(5, 5))
    plt.imshow(cat)
    plt.show()

# Flipping the image manually using NumPy
cat_flip = np.zeros((cat_width, cat_height, cat_C), dtype=np.uint8)
for i, row in enumerate(cat):
    cat_flip[cat_width -1 -i, :, :] = row  # Flip image along vertical axis

if plot_images_flipping:
    plt.figure(figsize=(5, 5))
    plt.imshow(cat_flip)
    plt.title("Flip using NumPy arrays")
    plt.show()

# Flipping the image using PIL methods
# Flip using PIL's transpose method
cat_flip_trans = cat_image.transpose(Image.FLIP_TOP_BOTTOM) #  cat_image.transpose(1)
if plot_images_flipping:
    plt.figure(figsize=(5, 5))
    plt.imshow(cat_flip_trans)
    plt.title("Flip using transpose method")
    plt.show()

# Flip using PIL's built-in function
cat_flip_pil = ImageOps.flip(cat_image)
if plot_images_flipping:
    plt.figure(figsize=(5, 5))
    plt.imshow(cat_flip_pil)
    plt.title("Flip using PIL flip")
    plt.show()

cat_mirror = ImageOps.mirror(cat_image)
if plot_images_flipping:
    plt.figure(figsize=(5, 5))
    plt.imshow(cat_mirror)
    plt.title("Mirror using PIL flip")
    plt.show()

# Image module has built-in attributes that describe the type of flip.
flip_parameters = {
    "FLIP_LEFT_RIGHT": Image.FLIP_LEFT_RIGHT,
    "FLIP_TOP_BOTTOM": Image.FLIP_TOP_BOTTOM,
    "ROTATE_90": Image.ROTATE_90,
    "ROTATE_180": Image.ROTATE_180,
    "ROTATE_270": Image.ROTATE_270,
    "TRANSPOSE": Image.TRANSPOSE,
    "TRANSVERSE": Image.TRANSVERSE
}

print(flip_parameters["ROTATE_90"])
# We can plot each of the outputs using the different parameter values:
if plot_parameter_values:
    for key, values in flip_parameters.items():
        plt.figure(figsize=(10, 5))
        plt.subplot(1,2,1)
        plt.imshow(cat_image)
        plt.title("Original")
        plt.subplot(1,2,2)
        plt.imshow(cat_image.transpose(values))
        plt.title(key)
        plt.show()

# Cropping the image
upper, lower = 150, 400
# Crop using NumPy
cat_crop_top = cat[upper:lower, :, :]
if plot_cropping:
    plt.figure(figsize=(5, 5))
    plt.imshow(cat_crop_top)
    plt.title("Crop Top")
    plt.show()

left, right = 150, 400
# Crop using NumPy
cat_crop_horizontal = cat_crop_top[:, left:right, :]
if plot_cropping:
    plt.figure(figsize=(5, 5))
    plt.imshow(cat_crop_horizontal)
    plt.title("Crop using array")
    plt.show()

# Cropping using PIL
cat_crop_pil = cat_image.crop((left, upper, right, lower))
if plot_cropping:
    plt.figure(figsize=(5, 5))
    plt.imshow(cat_crop_pil)
    plt.title("Crop using PIL")
    plt.show()

cat_crop_trans = cat_crop_pil.transpose(Image.FLIP_LEFT_RIGHT)
if plot_cropping:
    plt.figure(figsize=(5, 5))
    plt.imshow(cat_crop_trans)
    plt.title("Crop Transposed")
    plt.show()

# Changing specific pixels in the image
cat_altered = np.copy(cat)
cat_altered[upper:lower, left:right, 1:2] = 0  # Zero out the green channel in the selected region
if plot_pixel_altering:
    plt.figure(figsize=(10, 5))
    plt.subplot(1,2,1)
    plt.imshow(cat)
    plt.title("Original")
    plt.subplot(1,2,2)
    plt.imshow(cat_altered)
    plt.title("Altered Image")
    plt.show()

# Drawing and placing text using ImageDraw
cat_image_draw = cat_image.copy()
draw = ImageDraw.Draw(im=cat_image_draw)
shape = (left, upper, right, lower)  # tuple
draw.rectangle(xy=shape, fill="red")  # Draw a red rectangle
if plot_pixel_altering:
    plt.figure(figsize=(5, 5))
    plt.imshow(cat_image_draw)
    plt.title("Image with Rectangle")
    plt.show()

# Placing text on the image
draw.text(xy=(200, 250), text="There is a cat behind the red square!", fill=(0, 0, 0))  # Add text to the image
if plot_pixel_altering:
    plt.figure(figsize=(5, 5))
    plt.imshow(cat_image_draw)
    plt.title("Image with Rectangle and Text")
    plt.show()

# Overlaying an image onto another image
lenna_image = Image.open("images/lenna.png")
lenna = np.array(lenna_image)
lenna[upper:lower, left:right, :] = cat[upper:lower, left:right, :] # Overlay cat crop onto Lenna
if plot_pixel_altering:
    plt.figure(figsize=(5, 5))
    plt.imshow(lenna)
    plt.title("Cat Overlay on Lenna")
    plt.show()

# Pasting cropped and flipped image onto Lenna using PIL
lenna_image.paste(cat_crop_trans, box=(left, upper))
if plot_pixel_altering:
    plt.figure(figsize=(5, 5))
    plt.imshow(lenna_image)
    plt.title("Cat Overlay using PIL Paste")
    plt.show()

# Exercise
new_image = cat_image
copy_image = cat_image.copy()
image_fn = ImageDraw.Draw(im=cat_image)
image_fn.rectangle(xy=shape, fill="yellowgreen")
image_fn.text(xy=(200,250), text="This is going to be interesting!", fill=(0,0,0))

if plot_pixel_altering:
    plt.figure(figsize=(10, 5))
    plt.subplot(1,2,1)
    plt.imshow(new_image)
    plt.title("New Image")
    plt.subplot(1,2,2)
    plt.imshow(copy_image)
    plt.title("Copied Image")
    plt.show()