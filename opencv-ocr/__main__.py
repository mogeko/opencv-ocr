from extract_proj import extract_feature
from preprocess import preprocess_image
from recognize import recognize_text
import cv2 as cv
import sys


def main(path):
    img = cv.imread(path)
    img_bin = preprocess_image(img)
    rows, p = extract_feature(img_bin)

    str = []
    for i in range(len(rows)):
        for j in range(len(rows[i])):
            text = recognize_text(rows[i][j]).strip()
            cv.rectangle(img, (p[i][j][1]-p[i][j][3], p[i][j]
                               [0]-p[i][j][2]), (p[i][j][1], p[i][j][0]), (0, 255, 0), 2)
            cv.putText(img, text, (p[i][j][1]-p[i][j][3]-4, p[i][j][0]-p[i]
                       [j][2]-1), cv.FONT_HERSHEY_COMPLEX, 0.4, (50, 50, 255), 1)
            str.append(text)

    print(''.join(str), end=None)
    cv.imshow('Result', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv[1])
