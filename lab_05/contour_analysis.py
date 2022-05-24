import cv2
import streamlit as st
import numpy as np

from utils.utils import loader

# URL = "https://images.pexels.com/photos/11572738/pexels-photo-11572738.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=750&w=1260"
# PATTERN_URL = "https://i.ibb.co/c1Jp7Rd/Screenshot-2022-05-23-at-21-32-30.png"
URL = "https://i.ibb.co/RjLN0rL/mario.png"
PATTERN_URL = "https://i.ibb.co/GCWLkGV/mario-coin.jpg"


def description():

    task = """
        - ВК - вектор-контур
        - НСП - нормированное скалярное произведение
        - ВКР - взаимокорреляционная функция
        - АКЛ - автокорреляционная функция ВК
        
        Алгоритм:
        1. Предварительная обработка изображения (сглаживание, фильтрация помех, увеличение контраста) 
        2. Выделение границ объектов (желательно использовать Кэнни)
        3. Предварительная фильтрация границ по размерам и другим признакам (по периметру, площади и т.п.)
        4. Кодирование замкнутых границ без самопересечений в виде ВК
        5. Приведение ВК к единой длине, сглаживание (эквализация ВК)
        6. Сравнение всех найденных ВК с эталонным ВК с помощью "модуль тау макс"
        
        Реализовать:
        - в пейнте набросать геометрические фигуры, например треугольник
        - закодировать такой треугольник и найти на случайном изображении
        - программа должна найти фигуры
    """

    st.markdown("### Лабораторная работа №5")
    st.markdown("**Тема:** Контурный анализ")

    if st.checkbox("Показать задание"):
        st.write(task)
    st.markdown("---")


def get_contours(gray, color):
    c1, c2 = st.columns(2)
    if c1.checkbox("Сглаживание", value=True):
        gray = cv2.GaussianBlur(gray, (3, 3), cv2.BORDER_DEFAULT)

    if c2.checkbox("Определить границы", value=True):
        gray = cv2.Canny(gray, 50, 150)

    # apply binary thresholding
    ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    c1.image(thresh)

    # detect the contours on the binary image using cv2.CHAIN_APPROX_SIMPLE
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    # st.write(contours)

    image_copy = np.uint8(color).copy()
    cv2.drawContours(
        image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    # see the results
    c2.image(image_copy)


def match_by_template(c_img, img, pattern):
    w, h = pattern.shape[::-1]

    res = cv2.matchTemplate(img, pattern, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)

    for pt in zip(*loc[::-1]):
        cv2.rectangle(c_img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    st.image(c_img)


def equlizehist(image_path, denoise=False, verbose=False):
    bgr = cv2.imread(image_path)
    bgr[:, :, 0] = cv2.equalizeHist(bgr[:, :, 0])
    bgr[:, :, 1] = cv2.equalizeHist(bgr[:, :, 1])
    bgr[:, :, 2] = cv2.equalizeHist(bgr[:, :, 2])

    if denoise:
        # bgr = cv2.fastNlMeansDenoisingColoredMulti([bgr, bgr, bgr, bgr, bgr], 2, 5, None, 4, 5, 35)
        bgr = cv2.fastNlMeansDenoisingColored(bgr, None, 10, 10, 7, 21)

    if verbose:
        cv2.imshow("test", bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return bgr


def histogram_equalization(filepath):
    intensitiesList = []
    frequencyList = []
    cumulativeFrequencyList = []

    img = cv2.imread(filepath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    numberOfPixels = img.shape[0] * img.shape[1]

    for row in range(0, img.shape[0]):
        for column in range(0, img.shape[1]):
            if img[row, column] not in intensitiesList:
                intensitiesList.extend([img[row, column]])

    # print 'Intensities in the image: ', str(intensitiesList)

    intensitiesList.sort()
    # print 'Sorted intensities in the image: ', str(intensitiesList)

    for intensity in intensitiesList:
        count = 0
        for row in range(0, img.shape[0]):
            for column in range(0, img.shape[1]):
                if intensity == img[row, column]:
                    count += 1

        frequencyList.extend([count])
        if not cumulativeFrequencyList:
            cumulativeFrequencyList.extend([count])
        else:
            cumulativeValue = cumulativeFrequencyList[len(cumulativeFrequencyList) - 1] + count
            cumulativeFrequencyList.extend([cumulativeValue])

    # print 'Frequencies of each intensity (Sorted intensities): ', str(frequencyList)

    # print 'Cumulative frequencies: ', str(cumulativeFrequencyList)

    for row in range(0, img.shape[0]):
        for column in range(0, img.shape[1]):
            i = intensitiesList.index(img[row, column])
            img[row, column] = int(((cumulativeFrequencyList[i] - cumulativeFrequencyList[0]) / (
                        numberOfPixels - cumulativeFrequencyList[0])) * (255))

    cv2.imwrite('..\\results\\histogram_equalization.png', img)


def main():
    # TODO drop:
    #         https://robocraft.ru/computervision/640
    #         https://docs.opencv.org/3.3.0/d4/dc6/tutorial_py_template_matching.html
    #         https://www.geeksforgeeks.org/how-to-detect-shapes-in-images-in-python-using-opencv/

    description()

    color_img, gray_image = loader(URL, txt="изображение")
    # get_contours(gray_image, color_img)
    st.markdown("---")

    c_pattern, gray_pattern = loader(PATTERN_URL, txt="паттерн")
    get_contours(gray_pattern, c_pattern)

    match_by_template(color_img, gray_image, c_pattern)
    # match_by_template(gray_pattern, gray_pattern)


if __name__ == "__main__":
    main()
