import cv2 as cv
import numpy as np
from itertools import product


def get_h_progection(img):
    r, c = img.shape
    h_progection = np.zeros(img.shape, np.uint8)
    hrowsum = [0]*r
    for i, j in product(range(r), range(c)):
        if img[i, j] == 255:
            hrowsum[i] += 1
    for i in range(r):
        for j in range(hrowsum[i]):
            h_progection[i, j] = 255
    return hrowsum


def get_v_progection(img):
    r, c = img.shape
    v_progection = np.zeros(img.shape, np.uint8)
    vcolsum = [0]*c
    for i, j in product(range(r), range(c)):
        if img[i, j] == 255:
            vcolsum[j] += 1
    for j in range(c):
        for i in range(r-vcolsum[j], r):
            v_progection[i, j] = 255
    return vcolsum


def extract_feature(img):
    # 二值化
    _, img_bin = cv.threshold(
        img, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    # 计算水平投影
    h_proj = np.sum(img, axis=1)

    # 根据水平投影分割行
    rows, start, y = [], 0, []
    for i in range(len(h_proj)):
        if h_proj[i] == 0:
            if i > start:
                rows.append(img_bin[start:i, :])
                y.append([i, i-start])
            start = i+1

    # 计算垂直投影
    v_proj = [np.sum(row, axis=0) for row in rows]

    # 根据垂直投影分割字符
    for i in range(len(rows)):
        cols, start, x = [], 0, []
        for j in range(len(v_proj[i])):
            if v_proj[i][j] == 0:
                if j > start:
                    cols.append(rows[i][:, start:j])
                    x.append([y[i][0], j, y[i][1], j-start])
                start = j+1
        rows[i] = cols
        y[i] = x

    # 为字符添加边框
    for row in range(len(rows)):
        for col in range(len(rows[row])):
            rows[row][col] = cv.copyMakeBorder(
                rows[row][col], 20, 20, 20, 20, cv.BORDER_CONSTANT, value=0)

    return rows, y


def location(xy, img):
    for i in range(len(xy)):
        for j in range(len(xy[i])):
            img = cv.rectangle(img, (xy[i][j][1]-xy[i][j][3], xy[i][j]
                               [0]-xy[i][j][2]), (xy[i][j][1], xy[i][j][0]), (0, 255, 0), 2)

    cv.imshow('Location', img)
    return img


if __name__ == '__main__':
    from preprocess import preprocess_image
    from localize import localize_text

    img = cv.imread('examples/simple_1.jpg')
    img = preprocess_image(img)
    cv.imshow('Original', img)
    contours = localize_text(img)
    imgs, p = extract_feature(img)

    for i in range(len(imgs)):
        for j in range(len(imgs[i])):
            cv.imshow('Character {}-{}'.format(i, j), imgs[i][j])

    location(p, cv.imread('examples/simple_1.jpg'))

    cv.imwrite('examples/extracted.png', imgs[0][0])
    cv.waitKey(0)
    cv.destroyAllWindows()
