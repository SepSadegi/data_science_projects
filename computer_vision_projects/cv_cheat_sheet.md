# Digital Image Processing in Python Cheat Sheet

- A **digital image** is a rectangular array of numbers.
- A **grayscale image** consists of shades of gray, represented by intensity values between `0` and `255`.
  - `0` represents black, and `255` represents white.
  - The **contrast** is the difference between these values.

### Pixel Intensity Values
- **256 intensity values** provide high image quality.
- **Lower intensity values** result in loss of detail:
  - 32 values: Similar to original.
  - 16 values: Noticeable loss in low-contrast areas.
  - 8 values: Image looks off.
  - 2 values: Image appears cartoonish.

### Image Dimensions
- A digital image has dimensions:
  - **Height** (number of rows).
  - **Width** (number of columns).
- Each **pixel** has an index based on its row and column position.

## Color Images (RGB)
- **RGB image**: Combination of Red, Green, and Blue channels.
- Each channel has its own intensity values.
  - Index `0`: Red Channel.
  - Index `1`: Blue Channel.
  - Index `2`: Green Channel.
- Grayscale images are 2D, while color images are 3D (with channels as the 3rd dimension).

## Image Masks
- **Binary image masks**: Identify objects with intensity values of `0` (black) or `1` (white).
  - Useful for segmentation and object detection.

## Video Frames
- A **video** is a sequence of images or frames.
- Each frame can be treated as an individual image.

## Common Image Formats
- **JPEG**: Joint Photographic Expert Group image, commonly used for compression.
- **PNG**: Portable Network Graphics, supports lossless compression.

## Python Libraries for Image Processing

### Pillow (PIL)
- **Pillow (PIL)** is a popular library for image processing in Python.

#### Basic Operations with PIL:
1. **Importing and Loading Images**:
   ```python
   from PIL import Image
   img = Image.open('image_path.png')
   
2. **Displaying an Image**:
   ```python
   img.show()

3. **Image Attributes**:
- `img.format`: Image format (e.g., JPEG, PNG).
- `img.size`: Dimensions of the image (width, height).
- `img.mode`: Color space (e.g., RGB, L for grayscale).

4. **Converting to Grayscale**:
   ```python
    gray_img = img.convert('L')
    gray_img.show()

5. **Saving an Image**:
   ```python
    gray_img.save('new_image.jpg')
    
6. **Quantizing an Image**:
- Reducing color depth:
   ```python
   quantized_img = img.quantize(16)  # 16 quantization levels

7. **Working with Color Channels**:
   ```python
   red, green, blue = img.split()
   red.show()  # Displays the red channel
   
8. **Converting PIL Image to NumPy Array**:
   ```python
   import numpy as np
   img_array = np.array(img)

### OpenCV (cv2)
- **OpenCV** is another popular library used for computer vision tasks. It has more advanced functionality than Pillow.

#### Basic Operations with OpenCV:
1. **Importing OpenCV and Loading Images**:
   ```python
   import cv2
   img = cv2.imread('image_path.png')

2. **Displaying an Image**:
   ```python
   cv2.imshow('Image', img)
   cv2.waitKey(0)
   cv2.destroyAllWindows()

3. **Image Attributes**:
- `img.shape`: Returns the shape (height, width, channels).

4. **Color Space in OpenCV**:
- OpenCV uses BGR instead of RGB.

5. **Converting from BGR to RGB**:
   ```python
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
6. **Converting to Grayscale**:
   ```python
   gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

7. **Saving an Image**:
   ```python
   cv2.imwrite('output_image.jpg', img)
   
8. **Extracting Color Channels**:
   ```python
   blue, green, red = img[:, :, 0], img[:, :, 1], img[:, :, 2]
   
## Manipulating Images

### 1. Copying Images
- **Copying** allows you to create a new image independent of the original image array.

1. **Memory Address**:
   - Using the `id()` function, you can verify the memory address of an image object.
   - **Direct Assignment**:
     - Assigning an image array (e.g., `A = baboon`) points both variables to the same memory address.
     - Changes to `A` will affect the original image.
   - **Using `copy()`**:
     - `B = baboon.copy()` creates an independent copy of the image array.
     - Changes to `B` will not affect the original image or `A`.

2. **Common Mistake**:
   - If an image array is modified without using `copy()`, the original image may also change due to shared memory addresses.
   - Use `copy()` when you want to avoid unintentional modifications to the original image.

### 2. Flipping Images
- **Flipping** changes the orientation of an image. It can be done by modifying the pixel indices or using library functions.

#### Using PIL:
1. **PIL's ImageOps Module**:
   - **Flip Image Vertically**:
     ```python
     from PIL import ImageOps
     flipped_img = ImageOps.flip(img)
     ```
   - **Mirror Image Horizontally**:
     ```python
     mirrored_img = ImageOps.mirror(img)
     ```
   - **Transpose Image**:
     - Flip or rotate using `transpose()`. Example:
       ```python
       flipped_img = img.transpose(Image.FLIP_TOP_BOTTOM)
       ```

#### Using OpenCV:
1. **OpenCV's `flip()` Function**:
   - Flip an image vertically, horizontally, or both by specifying the `flipCode`:
     - `flipCode = 0`: Flips vertically (around y-axis).
     - `flipCode = 1`: Flips horizontally (around x-axis).
     - `flipCode = -1`: Flips both vertically and horizontally.
     ```python
     flipped_img = cv2.flip(img, flipCode=0)
     ```

2. **OpenCV's `rotate()` Function**:
   - Rotate images using predefined attributes:
     - **Rotate 90 Degrees Clockwise**:
       ```python
       rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
       ```
     - Other rotations:
       - `cv2.ROTATE_90_COUNTERCLOCKWISE`
       - `cv2.ROTATE_180`
       
## Pixel Manipulations

### 3. Cropping Images
- **Cropping** involves cutting out a part of an image and discarding the rest.

1. **Slicing Arrays**:
   - Use array slicing to select specific rows and columns.
   - Example: Select rows 2 to 4 and columns 1 and 2:
     ```python
     cropped_image = image_array[2:4, 1:3]
     ```
   - Cropping can be performed on both grayscale and color images using slicing. For color images, you can slice across multiple channels using a colon (`:`).

2. **Cropping with PIL**:
   - Use the `crop()` method in PIL to crop images:
     ```python
     cropped_img = img.crop((left, upper, right, lower))
     ```

### 4. Changing Pixel Values
- You can change pixel values by directly modifying elements in the image array.

1. **Setting Pixel Values**:
   - Example: Set pixels in a grayscale image to 255 to create a white rectangle:
     ```python
     image_array[2:5, 2:5] = 255
     ```

2. **Color Image Manipulation**:
   - Specify the channel (e.g., red, green, blue) when changing pixel values for a color image:
     ```python
     image_array[2:5, 2:5, 0] = 255  # Change red channel
     ```

### 5. Drawing Shapes
- You can draw simple shapes like rectangles or lines by changing pixel values or using drawing functions.

1. **Using PIL**:
   - **Drawing Shapes**:
     - Use the `ImageDraw` module to draw on images:
       ```python
       from PIL import ImageDraw
       draw = ImageDraw.Draw(image)
       draw.rectangle([(x1, y1), (x2, y2)], fill=color)
       ```
   - **Adding Text**:
     - Use the `text()` method to overlay text:
       ```python
       draw.text((x, y), "Hello", fill="white")
       ```

2. **Using OpenCV**:
   - **Drawing Rectangles**:
     - Example to draw a rectangle:
       ```python
       cv2.rectangle(image, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 0), thickness=2)
       ```
   - **Overlaying Text**:
     - Example to add text:
       ```python
       cv2.putText(image, "Hello", org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255))
       ```

### 6. Pasting and Superimposing Images
- You can paste part of one image onto another by specifying the coordinates of the target area.

1. **Using PIL**:
   - Use the `paste()` method to superimpose one image over another:
     ```python
     background_img.paste(foreground_img, (x, y))
     ```

2. **Using OpenCV**:
   - Directly assign pixel values from one image to another:
     ```python
     target[y1:y2, x1:x2] = source[y1:y2, x1:x2]
     ```

## Pixel Transformations

### 7. Histograms
- **Definition**: Represents the frequency of pixel intensity values in an image.
- **Calculation**: Use `cv2.calcHist()` to compute the histogram: 
  ```python
  hist = cv2.calcHist(images=[image], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
  intensity_values = np.array([x for x in range(hist.shape[0])])
  plt.plot(intensity_values, hist)
  ```
- **Interpretation**:
  - The x-axis represents pixel intensity values (0 to 255 for grayscale).
  - The y-axis shows the count of pixels at each intensity level.
  
- **Probability Mass Function (PMF = hist / (goldhill.size))**: Normalizes the histogram to provide a probabilistic view of the intensity distribution, which is often useful in advanced image processing and analysis tasks.

- **intensity_values (np.array([x for x in range(256)]))**: Array of possible pixel intensity levels (0 to 255) for grayscale images or each color channel in an image.
It Provides the x-axis values for histogram plots, allowing to visualize the distribution of pixel intensities.
 
### 8. Intensity Transformations

- **Concept**:
Modify the pixel values of an image based on a transformation function. An image can be viewed as a function $f(x, y)$, and a transformation $T$ maps it to a new image:

$$
  g(x, y) = T(f(x, y))
$$

Intensity transforms depend only on one value; as a result, it is sometimes referred to as a grey-level mapping.

- **Linear Transformation Formula**:
  
$$
  g(x, y) = 2 f(x, y) + 1
$$

### 9. Image Negatives
- **Concept**:
Image negatives reverse the intensity levels of an image, transforming bright areas into dark ones and vice versa. 

- **Formula**:

$$
  g(x, y) = L - 1 - f(x, y)
$$

-**For 8-bit images** ($L=256$): The formula simplifies to:

$$
  g(x, y) = 255 - f(x, y)
$$

```python
negative_img = cv2.bitwise_not(image)
```

### 10. Brightness and Contrast Adjustments:

$$
  g(x, y) = \alpha f(x, y) + \beta
$$

- **`𝛼`**: Contrast control
- **`𝛽`**: Brightness control

Rather than implementing via array operations, we use the function `cv2.convertScaleAbs()`:
```python
bright_contrast_img = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
```
It scales, calculates absolute values, and converts the result to 8-bit so the values fall between [0,255].

### 11. Histogram Equalization
- **Purpose**: Increases contrast by flattening the histogram.
- **Function**: `cv2.equalizeHist()`.

### 12. Thresholding and Simple Segmentation

- **Purpose**: Extract objects or features from an image by applying a threshold.
- **Thresholding Function**: Sets pixel values to 255 if above the threshold, otherwise to 0.

```python
_, thresh_img = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
```

### Automatic Thresholding (Otsu's Method)
- **Purpose**: Automatically determines the optimal threshold value.
- **Function**: `cv2.threshold()` with `cv2.THRESH_OTSU` flag:

```python
_, otsu_img = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```


