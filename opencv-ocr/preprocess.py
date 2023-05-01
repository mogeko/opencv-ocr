import cv2 as cv


def preprocess_image(img, ksize=3):
    # 灰度化
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 降噪
    img = cv.medianBlur(img, ksize=ksize)

    # 二值化
    _, img_bin = cv.threshold(
        img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # 对二值图像取反
    img_bin = 255 - img_bin

    return img_bin


if __name__ == '__main__':
    img = cv.imread('examples/simple_1.png')
    cv.imshow('Original', img)
    cv.imshow('Processed', preprocess_image(img))
    cv.waitKey(0)
    cv.destroyAllWindows()
