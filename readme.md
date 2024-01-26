# Image Manipulation with Linear Algebra 
> This repo contains the mid-term project for my *Fundamentals of Matrices and Linear Algebra* course in SBU.

## Structure of the project
The main part of the code is written in the `LinImMan_module.py` file, which contains the source codes for manipulating images. The `main.ipynb` jupyter notebook shows what this module can do and contains some explanations on how the module codes are implemented.

## `LinImMan` class in `LinImMan_module.py` file
`LinImMan` (short form of *Linear algebra Image Manipulator*) is a Python class that provides methods for image processing, including applying filters and computing histograms using linear algebra algorithms and techniques.

## Prerequisites
To use the LinImMan module, you need to have the following dependencies installed:
- OpenCV (`cv2`)
- NumPy (`numpy`)
- Matplotlib (`matplotlib`)

## Usage

## 1. Import the necessary modules:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```
## 2. Create an instance of the `Image` class and initialize it with an image:
```python
image = Image('path/to/image.jpg')
```
Thus any instance made through this module is assumed to be an image.

## 3. The methods of the `Image` class
> There are many methods in this class but here, I talk about a few of them. For further understanding, one can read the source code of the module.

- `__init__(self, path_to_image: str)` 
Initializes the Image object with the image located at the given path.

- `show_img(self)`
Displays the image.

- `save_img(self, name: str, path: str = '')`
Saves the image with the given name and path.

- `make_gray(self)`
Converts the image to grayscale using the formula:
```latex
gray value = 0.114 * blue + 0.587 * green + 0.299 * red
```
A widely used approximation for converting a color image to grayscale. It is based on the relative luminance of the RGB color model and is often referred to as the "ITU-R Recommendation BT.601" or "Rec. 601" formula.
It's worth noting that there are other formulas and color transformation techniques available for converting color images to grayscale, each with its own characteristics and purposes. The choice of the Rec. 601 formula in this particular implementation is due to its simplicity and historical usage in various imaging standards and applications.

- `invert_color(self)`
Inverts the colors of the image by subtracting each color channel value from 255.

- `adjust_brightness(self, factor: float)`
It first converts the self.image array to a floating-point data type using astype(np.float32). This is done to ensure that the brightness adjustment can be performed accurately with decimal values.
The brightness adjustment is applied by multiplying the `image_float` array by the factor parameter. This scales the pixel values of the image by the specified factor.
Next, the `adjusted_image` array is clipped using `np.clip` to ensure that all pixel values are within the valid range of 0 to 255. Any values below 0 are set to 0, and any values above 255 are set to 255.

- `rotate_image(self, degree: float, direction: str = 'counterclockwise')`
The degree parameter is converted to radians using `np.radians`. This step is necessary because the trigonometric functions in NumPy expect angles in radians.
The height and width of the image are extracted using `self.image.shape[:2]`. This provides the dimensions of the image, which are needed for the rotation transformation.
The center coordinates of the image are calculated. This determines the point around which the image will be rotated.
The cosine (`cos_theta`) and sine (`sin_theta`) of the angle in radians are calculated using NumPy's trigonometric functions np.cos and np.sin.


A rotation matrix is created using the calculated cosine, sine, and center coordinates. The rotation matrix is a 2x3 array where the first row represents the X-axis transformation and the second row represents the Y-axis transformation.
If the direction parameter is set to `counterclockwise`, the rotation matrix is inverted using cv2.invertAffineTransform. This is necessary because OpenCV's `warpAffine` function expects an affine transformation matrix for a counterclockwise rotation.
The `cv2.warpAffine` function is used to apply the rotation transformation to the image. It takes the image, rotation matrix, and the desired output size (width and height) as parameters and returns the rotated image.
Finally, the image attribute of the Image object is updated with the rotated image.
After calling the `rotate_image` method, the image attribute of the Image object will contain the rotated image according to the specified degree and direction.

- `edge_detection(self)` 
Performs edge detection on the image *using matrix functions*.

- `built_in_edge_detection(self, threshold1: float, threshold2: float)`
Performs edge detection on the image using the Canny edge detection algorithm.

- `apply_blur_filter(self)` 
Applies a blur filter to the image using a 5x5 kernel.
