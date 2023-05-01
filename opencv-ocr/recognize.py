import pytesseract
import cv2 as cv


def recognize_text(char):
    return pytesseract.image_to_string(char, lang='eng', config='--psm 10')


if __name__ == "__main__":
    t = cv.imread('examples/t.png')
    print(recognize_text(t))
