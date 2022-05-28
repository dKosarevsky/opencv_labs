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
        
        Сравнить результаты по всем трём
    """

    st.markdown("### Лабораторная работа №6")
    st.markdown("**Тема:** Характерные (особые, ключевые) точки изображения")

    if st.checkbox("Показать задание"):
        st.write(task)
    st.markdown("---")


def moravec_detector(img_gray, img, block_size, threshold):
    r = block_size // 2
    row = img_gray.shape[0]
    col = img_gray.shape[1]

    min_val = np.zeros(img_gray.shape)
    count = 0

    for y in range(r, row - r):
        for x in range(r, col - r):
            v1 = 0
            v2 = 0
            v3 = 0
            v4 = 0
            for k in range(-r, r):
                v1 += np.square(img_gray[y, x + k] - img_gray[y, x + k + 1])
                v2 += np.square(img_gray[y + k, x] - img_gray[y + k + 1, x])
                v3 += np.square(img_gray[y + k, x + k] - img_gray[y + k + 1, x + k + 1])
                v4 += np.square(img_gray[y + k, x - k] - img_gray[y + k + 1, x - k - 1])

            min_val[y, x] = min(v1, v2, v3, v4)

            if min_val[y, x] < threshold:
                min_val[y, x] = 0
            else:
                count += 1

    for i in range(block_size, row - block_size):
        for j in range(block_size, col - block_size):
            mat = min_val[i - block_size:i + block_size, j - block_size:j + block_size]
            if np.max(mat) != 0:
                pos = np.unravel_index(np.argmax(mat), mat.shape)
                corn_y = i + pos[0] - block_size
                corn_x = j + pos[1] - block_size
                cv2.circle(img, (corn_x, corn_y), 3, COLOR)

    st.write("Детектор Моравеца.")
    st.write(f"Кол-во углов: {count}")

    return img


def harris_detector(img_gray, img, block_size, aperture_size, k):
    img_gray = np.float32(img_gray)

    dest = cv2.cornerHarris(img_gray, block_size, aperture_size, k)
    thresh = 0.01 * dest.max()
    num_corners = np.sum(dest > thresh)
    dest = cv2.dilate(dest, None)

    for i in range(dest.shape[0]):
        for j in range(dest.shape[1]):
            if int(dest[i, j]) > thresh:
                cv2.circle(img, (j, i), 3, COLOR)

    st.write("Детектор Харриса-Стефана.")
    st.write(f"Кол-во углов: {num_corners}")

    return img


def fast_detector(img_gray, img, thresh, non_max_suppression):
    fast = cv2.FastFeatureDetector_create(thresh, non_max_suppression)

    corners = fast.detect(img_gray, None)
    img = cv2.drawKeypoints(img, corners, None, color=COLOR)

    st.write("Детектор FAST.")
    st.write(f"Кол-во углов: {len(corners)}")

    return img


def main():
    description()

    detector = st.radio(
        "Выберите Детектор:", (
            "1. Моравеца",
            "2. Харриса-Стефана",
            "3. FAST",
        ),
        index=2
    )[:1]
    st.markdown("---")

    color_img, gray_image = loader(URL, txt="изображение")
    color_img = np.asarray(color_img)
    st.markdown("---")

    if detector == "1":
        c3, c4 = st.columns(2)
        block_size = c3.number_input("Размер блока:", min_value=1, max_value=99, value=5, step=1)
        thresh = c4.number_input("Порог:", min_value=1, max_value=9999, value=300, step=1)
        moravec_img = moravec_detector(gray_image.copy(), color_img.copy(), block_size, thresh)
        st.image(moravec_img)
        st.markdown("---")

    if detector == "2":
        c5, c6, c7 = st.columns(3)
        harris_block = c5.number_input("Размер блока:", min_value=1, max_value=99, value=2, step=1)
        harris_a = c6.number_input("Размер апертуры:", min_value=1, max_value=9999, value=5, step=2)
        harris_k = c7.number_input("k:", min_value=.01, max_value=1., value=.07, step=.01)
        harris_img = harris_detector(gray_image.copy(), color_img.copy(), harris_block, harris_a, harris_k)
        st.image(harris_img)
        st.markdown("---")

    if detector == "3":
        c8, c9 = st.columns(2)
        fast_thresh = c8.number_input("Порог:", min_value=1, max_value=9999, value=10, step=1)
        non_max_suppression = c9.checkbox("Подавление не-максимумов", value=True)
        res = fast_detector(gray_image.copy(), color_img.copy(), fast_thresh, non_max_suppression)
        st.image(res)
        st.markdown("---")


if __name__ == "__main__":
    main()
