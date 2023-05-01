import cv2 as cv
import numpy as np
from localize import localize_text


def segment_characters(contours, image, min_char_height=20, min_char_width=10):
    character_bboxes = []

    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)

        if h > min_char_height or w > min_char_width:
            character_bboxes.append((x, y, w, h))

    return character_bboxes


def extract_features(character_bboxes, img_bin):
    features = []

    # 使用 ORB 特征来改进字符分割。
    # 我们将首先使用 ORB 特征检测器提取轮廓内的关键点，然后根据这些关键点找到可能的字符边界。
    orb = cv.ORB_create(nfeatures=1000)

    for x, y, w, h in character_bboxes:
        # 提取字符区域
        char_roi = img_bin[y:y + h, x:x + w]

        # 二值化 (防御性)
        _, roi_binary = cv.threshold(
            char_roi, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

        # 使用 ORB 提取关键点
        keypoints, _ = orb.detectAndCompute(roi_binary, None)

        # 如果没有检测到关键点，则将整个区域作为一个字符
        if not keypoints:
            features.append(char_roi)
            continue

        # 计算 x 坐标的最小值和最大值
        x_coords = [kp.pt[0] for kp in keypoints]
        min_x, max_x = int(min(x_coords)), int(max(x_coords))

        # 根据关键点的 x 坐标分割字符
        if min_x > 0:
            features.append(char_roi[:, :min_x])
        if max_x < w:
            features.append(char_roi[:, max_x:])

    return features


if __name__ == '__main__':
    from preprocess import preprocess_image

    img = cv.imread('examples/simple_2.png')
    img = preprocess_image(img)
    cv.imshow('Original', img)
    contours = localize_text(img)
    character_bboxes = segment_characters(contours, img)
    features = extract_features(character_bboxes, img)
    for i in range(len(features)):
        cv.imshow('Character {}'.format(i), features[i])
    cv.waitKey(0)
    cv.destroyAllWindows()
