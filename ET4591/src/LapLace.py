import cv2
import numpy as np
from Canny import *


def LaplaceFilter(image, kernel_type=1):
    if kernel_type == 2:
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    elif kernel_type == 3:
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    else:
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    output = convolute(image, kernel)
    return output
