import cv2 as cv


def preprocess(img):
    # 去除噪声，使用高斯模糊
    img = cv.GaussianBlur(img, (3, 3), 0)
    # 增强对比度
    img = cv.convertScaleAbs(img, alpha=1.5, beta=0)
    # 将图像转为灰度图像
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 二值化处理
    thresh = cv.threshold(
        gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    return thresh


if __name__ == '__main__':
    img = cv.imread('examples/simple_1.png')
    cv.imshow('Original', img)
    cv.imshow('Processed', preprocess(img))
    cv.waitKey(0)
    cv.destroyAllWindows()
