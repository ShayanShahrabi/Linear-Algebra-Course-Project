# PyLinMan

PyLinMan is a Python class that provides methods for image processing, including applying filters and computing histograms using linear algebra algorithms and techniques.

## Prerequisites

To use the PyLinMan module, you need to have the following dependencies installed:
- OpenCV (`cv2`)
- NumPy (`numpy`)
- Matplotlib (`matplotlib`)

## Usage

1. Import the necessary modules:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```
2. Create an instance of the `Image` class and initialize it with an image:
```python
image = Image('path/to/image.jpg')
```
3. The methods of the `Image` class

- `__init__(self, path_to_image: str)` 
Initializes the Image object with the image located at the given path.
- `show_img(self)`
Displays the image.
- `save_img(self, name: str, path: str = '')`
Saves the image with the given name and path.
- `make_gray(self)`
Converts the image to grayscale.
- `invert_color(self)`
Inverts the colors of the image.
- `adjust_brightness(self, factor: float)`
Adjusts the brightness of the image by a given factor.
- `rotate_image(self, degree: float, direction: str = 'counterclockwise')`
Rotates the image by the given degree in the specified direction.
- resize_image(self, scale_factor: float)`
Resizes the image by the given scale factor.
- `mirror_image(self, axis: str = 'horizontal')`
Mirrors the image along the specified axis.
- `edge_detection(self)` 
Performs edge detection on the image using matrix functions.
- `built_in_edge_detection(self, threshold1: float, threshold2: float)`
Performs edge detection on the image using the Canny edge detection algorithm.
- `apply_blur_filter(self)` 
Applies a blur filter to the image using a 5x5 kernel.
- `apply_sharpen_filter(self)`
Applies a sharpen filter to the image.
- `apply_emboss_filter(self)`
Applies an emboss filter to the image.
- `compute_histogram(self)`
Computes and displays the histogram of the image.
