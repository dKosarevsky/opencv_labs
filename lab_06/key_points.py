import streamlit as st
import numpy as np
import cv2

from utils.utils import loader

URL = "https://i.ibb.co/QQx07rW/pexels-anna-rye-12043861.jpg"
COLOR = (0, 255, 0)


def description():

    task = """
        Требования:
        1. Отличимость
        2. Инвариантность к афинным преобразованиям
        3. Стабильность
        4. Уникальность
        5. Интерпретируемость
        
        Детекторы углов. 
        
        Углы - простейшие из особых точек.
        
        Реализовать:
        
        1. Детектор углов Моравеца (самостоятельно)
        2. Детектор углов Харриса-Стефана (из OpenCV)
        3. Детектор FAST (Features From Accelerated Segment Test) (из OpenCV)
        
        Сравнить результаты сразу по всем трём, вывести на одном экране все три
    """

    st.markdown("### Лабораторная работа №6")
    st.markdown("**Тема:** Характерные (особые, ключевые) точки изображения")

    if st.checkbox("Показать задание"):
        st.write(task)
    st.markdown("---")


def moravec_detector(gray, color, window_size=5, threshold=5000):
    r = window_size // 2
    rows = gray.shape[0]
    cols = gray.shape[1]
    corners = []

    for y in range(r, rows - r):
        for x in range(r, cols - r):
            w_v1 = 0
            w_v2 = 0
            w_v3 = 0
            w_v4 = 0
            for k in range(-r, r):
                w_v1 += (gray[y, x + k] - gray[y, x + k + 1]) * (gray[y, x + k] - gray[y, x + k + 1])
            for k in range(-r, r):
                w_v2 += (gray[y + k, x] - gray[y + k + 1, x]) * (gray[y + k, x] - gray[y + k + 1, x])
            for k in range(-r, r):
                w_v3 += (gray[y + k, x + k] - gray[y + k + 1, x + k + 1]) * (gray[y + k, x + k] - gray[y + k + 1, x + k + 1])
            for k in range(-r, r):
                w_v4 += (gray[y + k, x - k] - gray[y + k + 1, x - k - 1]) * (gray[y + k, x - k] - gray[y + k + 1, x - k - 1])
            arr = np.array([w_v1, w_v2, w_v2, w_v2])
            val = min(arr)
            if val > threshold:
                corners.append((x, y))

    for corner in corners:
        cv2.circle(color, corner, 3, COLOR)

    st.write("Детектор Моравеца.")
    st.write(f"Кол-во углов: {len(corners)}")

    return color


def harris_detector(gray, color, block_size=2, aperture_size=5, k=0.07):
    gray = np.float32(gray)

    dest = cv2.cornerHarris(gray, block_size, aperture_size, k)
    thresh = 0.01 * dest.max()
    num_corners = np.sum(dest > thresh)
    dest = cv2.dilate(dest, None)

    for i in range(dest.shape[0]):
        for j in range(dest.shape[1]):
            if int(dest[i, j]) > thresh:
                cv2.circle(color, (j, i), 3, COLOR)

    st.write("Харриса-Стефана")
    st.write(f"Кол-во углов: {num_corners}")

    return color


def fast_feature_detector(gray, color):
    fast = cv2.FastFeatureDetector_create()

    corners = fast.detect(gray, None)
    img = cv2.drawKeypoints(color, corners, None, color=COLOR)

    st.write("Детектор FAST.")
    st.write(f"Кол-во углов: {len(corners)}")

    return img


def main():
    description()

    color_img, gray_image = loader(URL, txt="изображение")
    color_img = np.asarray(color_img)

    moravec_img = moravec_detector(gray_image.copy(), color_img.copy(), 5, 500)
    st.image(moravec_img)

    harris_img = harris_detector(gray_image.copy(), color_img.copy())
    st.image(harris_img)

    res = fast_feature_detector(gray_image.copy(), color_img.copy())
    st.image(res)


if __name__ == "__main__":
    main()
