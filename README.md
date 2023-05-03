# opencv-ocr

![result](https://user-images.githubusercontent.com/26341224/235641560-74889377-f5e1-41ae-841f-5b44072e0f71.jpg)

Implement OCR based on OpenCV ([opencv-python](https://pypi.org/project/opencv-python)).

## General idea

To implement OCR (Optical character recognition) with OpenCV, we will follow these general steps:

1. **Preprocess the image**: OCR requires a clear, bright, and noise-free image, so the first step is to preprocess the image, such as removing noise, smoothing, enhancing contrast, binarizing, and so on.
2. **Text localization**: OCR needs to recognize the text, which must be localized first. We can use [edge detection algorithms](https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html) and [morphological operations](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html) provided by OpenCV, such as erosion and dilation, to detect and segment text regions.
3. **Character segmentation**: Our OCR task is to recognize individual characters rather than whole words, so that we need to use character segmentation algorithms to further segment the text regions into individual characters.
4. **Feature extraction**: Once our text regions or characters are segmented, we need to extract their features for recognition. we can use [feature extraction algorithms](https://docs.opencv.org/3.4/db/d27/tutorial_py_table_of_contents_feature2d.html) provided by OpenCV, such as [SIFT](https://docs.opencv.org/3.4/da/df5/tutorial_py_sift_intro.html), [SURF](https://docs.opencv.org/3.4/df/dd2/tutorial_py_surf_intro.html), or [ORB](https://docs.opencv.org/3.4/d1/d89/tutorial_py_orb.html), etc.
5. **Train the model**: Once we have prepared the feature data, we can start training the model. We can use various machine learning algorithms, such as [Support Vector Machine (SVM)](https://docs.opencv.org/3.4/d1/d73/tutorial_introduction_to_svm.html), [Neural Networks](https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_dnn.html), [Random Forest](https://docs.opencv.org/3.4/d0/d65/classcv_1_1ml_1_1RTrees.html), etc.
6. **Recognize text**: When our model is trained, we can use it to recognize text. We can use the model to predict the text in the image and return the results to the user.

Flowchart:

```mermaid
flowchart LR
    g[Deep Learning] -.-> e
    a[Original] --> b[Pre-processed] --> c["Chars (Image)"] --> e["Chars (String)"] --> f[Result]
    b --> d[Location] ---> f
    a -----> f
```

## Implementation principle

Next, I will explain the specific steps for us to implement this OCR program with examples.

Before we start, this is the example picture we used:

<p align="center">
  <img src="https://user-images.githubusercontent.com/26341224/235652932-d20549d2-7cbf-4028-bcf0-32e5b93f2447.jpg"
       alt="original" width="80%">
</p>

### Preprocess the image

Image preprocessing is a key step to achieve an efficient and accurate OCR program. This step is mainly to enhance image **quality** and **optimize the next steps**.

We use operations such as **denoising**, **smoothing**, **enhanced** **contrast** and binaryization to better identify text. Clear, bright and noise-free images help to improve the accuracy of the OCR system. In addition, image preprocessing can also adapt the OCR program to different types of images. In real life, the image may be affected by lighting, camera quality, angle and other factors, resulting in poor image quality. Preprocessing helps to solve these problems, so that the OCR program can work normally under different conditions. **The pre-processed image has better quality and clarity, which helps to improve the effect of next steps, such as text positioning, character segmentation, feature extraction, etc.**

```python
def preprocess_image(img, ksize=3):
    # Grayscale
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Denoising
    img = cv.medianBlur(img, ksize=ksize)

    # Binaryization
    _, img_bin = cv.threshold(
        img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Inverse the binary image
    img_bin = 255 - img_bin

    return img_bin
```

<p align="center">
  <img src="https://user-images.githubusercontent.com/26341224/235645117-2febb4e3-e4e7-4b32-b10e-5befe9f37866.jpg"
       alt="processed" width="80%">
</p>

### Character segmentation

Then we need to divide the text area in the image into single characters. Compared with complete sentences or words, recognizing a single letter is less prone to ambiguity (because there are only 26 possibilities in total), thus improving the overall recognition accuracy. And after dividing the sentence into a single letter, the problem can be simplified to a classification problem and a classifier can be established for each letter. In this way, the difficulty of the training model is reduced, and the calculation cost is also reduced.

Finally, after dividing into single letters, the OCR system can support multiple languages more easily, because most languages are composed of basic characters (our OCR software does not support multiple languages for the time being).

We use the projection method to determine the position of each character and then cut it according to its position. At the same time, we record the position of each character for subsequent use.

We first calculate the horizontal projection and cut the picture horizontally through it:

```python
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
    cv.show('h_progection', h_progection)
    return hrowsum
```

<p align="center">
  <img src="https://user-images.githubusercontent.com/26341224/235650099-2137a6df-f04c-49fd-ab93-41ee94fd42db.png"
       alt="h_progection" width="80%">
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/26341224/235650541-7182b552-dbe6-42ff-ad71-223bddb74be6.jpg"
       alt="h_cut" width="80%">
</p>

Then we calculate the vertical projection and cut the picture vertically through it (don't forget to border each character for deep learning model recognition):

```python
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
    cv.imshow('v_progection', v_progection)
    return vcolsum
```

<p align="center">
  <img src="https://user-images.githubusercontent.com/26341224/235652290-e29d4985-ac9e-4ae3-bb6b-634e5ba9a8cf.png"
       alt="v_progection" width="80%">
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/26341224/236053978-134c3324-b1bf-4a5c-a324-a585576943f3.jpg"
       alt="v_cut">
</p>

### Recognize text

In order to save time, we skipped the steps of training the deep learning model and directly used the open source model of Google's Tesseract team.

Directly call Tesseract's API to identify character images:

```python
def recognize_text(char):
    return pytesseract.image_to_string(char, lang='eng', config='--psm 10').strip()
```

### Synthesis results

Finally, draw the position information of the character (green box) and the recognition result (red character) to the original picture:

```python
str = []
font_size, font_weight = 1, 2
for i in range(len(rows)):
    for j in range(len(rows[i])):
        text = recognize_text(rows[i][j])
        cv.rectangle(img, (p[i][j][1]-p[i][j][3], p[i][j]
                           [0]-p[i][j][2]), (p[i][j][1], p[i][j][0]), (0, 255, 0), 2)
        cv.putText(img, text, (p[i][j][1]-p[i][j][3]-4, p[i][j][0]-p[i]
                   [j][2]-1), cv.FONT_HERSHEY_COMPLEX, font_size, (50, 50, 255), font_weight)
        str.append(text)

print(''.join(str), end=None)
cv.imshow('Result', img)
cv.waitKey(0)
```

<p align="center">
  <img src="https://user-images.githubusercontent.com/26341224/235641560-74889377-f5e1-41ae-841f-5b44072e0f71.jpg"
       alt="result" width="80%">
</p>

## License

The code in this project is released under the [MIT License](./LICENSE).
