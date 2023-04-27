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


# 3. 字符分割 - 这一步通常需要更复杂的方法，这里只是一个简单的开始
def segment_characters(contours, image):
    characters = []
    for contour in contours:
        # 获取每个轮廓的边界框
        x, y, w, h = cv.boundingRect(contour)

        # 提取字符
        character = image[y:y+h, x:x+w]
        characters.append(character)

    return characters


if __name__ == '__main__':
    from preprocess import preprocess

    img = cv.imread('examples/simple_1.png')
    img = preprocess(img)
    cv.imshow('Original', img)
    (dilation, erosion) = localize(img)
    cv.imshow('Dilation', dilation)
    cv.imshow('Erosion', erosion)
    seg = segment_characters([erosion], img)
    for i in range(len(seg)):
        cv.imshow('Segmented Character {}'.format(i), seg[i])
    cv.waitKey(0)
    cv.destroyAllWindows()
