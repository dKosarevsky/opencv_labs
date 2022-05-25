import cv2
import streamlit as st
import numpy as np

from utils.utils import loader

PATTERN_URL = "https://img.favpng.com/18/12/25/triangle-blue-shape-png-favpng-GiyVGMhuhVfrT688Zw5x4DGXH.jpg"
URL = "https://mpng.subpng.com/20190607/clo/kisspng-triangle-symmetry-pattern-tessellation-filetriangular-orthobianticupola-net-png-wikipe-5cfa077caa4479.7233607715598897886974.jpg"
COLOR = (0, 255, 0)


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


def match_by_template(c_img, img, pattern, threshold=0.8):
    w, h = pattern.shape[::-1]

    res = cv2.matchTemplate(img, pattern, cv2.TM_CCOEFF_NORMED)

    loc = np.where(res >= threshold)

    for pt in zip(*loc[::-1]):
        cv2.rectangle(c_img, pt, (pt[0] + w, pt[1] + h), COLOR, 2)

    st.image(c_img)


def main():

    description()

    c_pattern, gray_pattern = loader(PATTERN_URL, txt="паттерн")
    # get_contours(gray_pattern, c_pattern)

    color_img, gray_image = loader(URL, txt="изображение")
    h, w = gray_image.shape[:2]
    # get_contours(gray_image, color_img)
    st.markdown("---")

    # Template matching using edges instead of the original image can greatly improve the accuracy of template matching.
    found = []
    (tH, tW) = gray_pattern.shape[:2]
    template_edged = cv2.Canny(gray_pattern, 50, 200)

    st.image(template_edged)

    # Traverse the image size
    i = 0
    for scale in np.linspace(1, 2, 20):
        resized = cv2.resize(gray_image, dsize=(0, 0), fx=scale, fy=scale)

        r = gray_image.shape[1] / float(resized.shape[1])

        if resized.shape[0] < tH or resized.shape[1] < tW:
            break
        edged = cv2.Canny(resized, 50, 200)
        result = cv2.matchTemplate(edged, template_edged, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

        if not found:
            found.append((maxVal, maxLoc, r))

        if maxVal > found[i][0]:
            found.append((maxVal, maxLoc, r))
            i += 1

    for f in found:
        (_, maxLoc, r) = f
        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
        color_img = np.asarray(color_img)
        cv2.rectangle(color_img, (startX, startY), (endX, endY), COLOR, 2)

    st.image(color_img)

    # match_by_template(color_img, gray_image, gray_pattern)
    # match_by_template(gray_pattern, gray_pattern)


if __name__ == "__main__":
    main()
