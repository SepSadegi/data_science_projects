# Computer Vision Projects

This is a repository for computer vision and image processing.

 **PIL (Pillow):** For basic image processing.
 **OpenCV:** Advanced computer vision tasks.

## Projects

### 1. Basic Image Processing with Pillow (PIL)

Basic image processing using the Pillow library to explore various image manipulation techniques.

- **Code:** `1_basic_image_processing_pillow.py`
- **Features:** 
    - Grayscale Conversion and Quantization
    - Color Channel Extraction
    - Conversion to NumPy Arrays
    - Array Indexing and Manipulation
- **Data:** 
    - `lenna.png`
    - `baboon.png`

### 2. Basic Image Processing with OpenCV (cv2)

Basic image processing using the OpenCV library to explore various image manipulation techniques. 

- **Code:** `2_basic_image_processing_opencv.py`
- **Features:** 
    - Grayscale Conversion and Quantization
    - Color Channel Extraction
    - Array Indexing and Manipulation
- **Data:** 
    - `lenna.png`
    - `baboon.png`
    - `barbara.png`

### 3. Basic Image Manipulation with Pillow (PIL) and NumPy

In this script, various image manipulation techniques are explored using both Pillow (PIL) and NumPy arrays. The code demonstrates flipping, cropping, pixel alteration, and overlaying images, while utilizing both PIL's built-in methods and NumPy operations for comparison.

- **Code:** `3_basic_image_manipulation_pillow.py`
- **Features:**
  - Copying and Memory Management
  - Image Flipping using PIL's `ImageOps.flip` and `ImageOps.mirror` as well as NumPy's array manipulation.
  - Cropping Performed using NumPy slicing and `PIL.Image.crop()` for more controlled manipulation.
  - Pixel Manipulation using NumPy and PIL's `ImageDraw` for direct pixel control.
  - Adding text and shapes to images using the `ImageDraw.Draw()` method.
  - Combining and overlaying one image onto another using both NumPy arrays and PIL's `paste()` method.

- **Data:**
  - `baboon.png`
  - `cat.png`
  - `lenna.png`
  
### 4. Basic Image Manipulation with OpenCV

In this code, I practice the same features as the previous script but utilize OpenCV's built-in functions (`flip()`, `rotate()`, `putText()`, and `rectangle()`) instead of manual manipulations through array operations or PIL functions.

- **Code:** `4_basic_image_manipulation_opencv.py`
- **Features:**
  - Copying and Memory Management
  - Efficient image flipping using `flip()` and `rotate()`
  - Cropping images
  - Changing specific image pixels
  - Adding text and shapes to images using `putText()` and `rectangle()`

- **Data:**
  - `baboon.png`
  - `cat.png`

