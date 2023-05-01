import cv2 as cv


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
