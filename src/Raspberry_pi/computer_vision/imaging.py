import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('testmaze10.jpg', 0)

# possibly resize image to help with walls (maze2lowres ran under this)
# img = cv.resize(img, (400, 350))

edges = cv.Canny(img, 100, 200)

cv.imwrite('testmaze10Edited.jpg', edges)
