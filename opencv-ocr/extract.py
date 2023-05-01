import cv2 as cv
import numpy as np


def localize_text(img_bin, threshold1=100, threshold2=200, ksize=(5, 5)):
    # 使用Canny边缘检测
    # See: https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html
    edges = cv.Canny(img_bin, threshold1, threshold2)

    # morphological operations
    # See: https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    # 定义结构元素
    kernel = cv.getStructuringElement(cv.MORPH_RECT, ksize)

    # 图像膨胀
    dilated = cv.dilate(edges, kernel)

    # 查找轮廓
    contours, _ = cv.findContours(
        dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    return contours


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
    from preprocess import preprocess_image

    img = cv.imread('examples/simple_1.png')
    img = preprocess_image(img)
    cv.imshow('Original', img)
    contours = localize_text(img)
    seg = segment_characters(contours, img)
    for i in range(len(seg)):
        cv.imshow('Segmented Character {}'.format(i), seg[i])
    cv.waitKey(0)
    cv.destroyAllWindows()
