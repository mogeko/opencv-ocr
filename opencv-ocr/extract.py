import cv2 as cv
import numpy as np


def localize(img, threshold1=100, threshold2=200, shape=(5, 5), iterations=1):
    # edge detection algorithms
    # See: https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html
    edges = cv.Canny(img, threshold1, threshold2)

   # morphological operations
   # See: https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    kernel = np.ones(shape, np.uint8)
    dilation = cv.dilate(edges, kernel, iterations=iterations)
    erosion = cv.erode(dilation, kernel, iterations=iterations)

    return (dilation, erosion)


if __name__ == '__main__':
    from preprocess import preprocess

    img = cv.imread('examples/simple_1.png')
    img = preprocess(img)
    cv.imshow('Original', img)
    (dilation, erosion) = localize(img)
    cv.imshow('Dilation', dilation)
    cv.imshow('Erosion', erosion)
    cv.waitKey(0)
    cv.destroyAllWindows()
