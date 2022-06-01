import streamlit as st
import numpy as np
import cv2

from utils.utils import loader

URL_1 = "https://images.pexels.com/photos/8825134/pexels-photo-8825134.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
URL_2 = "https://images.pexels.com/photos/6561253/pexels-photo-6561253.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"


def description():

    task = """
        Задание на ЛР:
        
        1. С помощью алгоритма метода SIFT из OpenCV найти на двух и изображениях ключевые точки. 
        Использовать одно и того же изображение, но снятое с немного разных ракурсов, 
        Например один и тот же объект (сцена) снят под разными углами.
        
        2. Найти соответствие точек друг другу. Использовать готовый метод OpenCV.
        
        3. Отсортировать результаты сопоставления по качеству.
        
        4. Отобразить результат на двух картинках с помощью линий. Показать наилучшие 10 соответствий.
        
        Туториал по [SIFT](https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_sift_intro/py_sift_intro.html)
        
        Туториал по [матчингу](https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html)
    """

    st.markdown("### Лабораторная работа №7")
    st.markdown("**Тема:** SIFT (Scale Invariant Feature Transform)")

    if st.checkbox("Показать задание"):
        st.markdown(task)
    st.markdown("---")


def show_key_points(img, key_points, img_name):
    st.write(f"Ключевые точки на изображении {img_name}")
    kp_1 = cv2.drawKeypoints(img, key_points, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    st.image(kp_1)
    st.write("---")


def show_matches(img_1, kp_1, img_2, kp_2, matches, matches_cnt):
    flag = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    img_result = cv2.drawMatches(img_1, kp_1, img_2, kp_2, matches[:matches_cnt], None, flags=flag)
    st.image(img_result)


def match_key_points(rgb_1, gray_1, rgb_2, gray_2):
    rgb_1 = np.asarray(rgb_1.copy())
    rgb_2 = np.asarray(rgb_2.copy())

    sift = cv2.SIFT_create()

    kp_1, desc_1 = sift.detectAndCompute(gray_1, None)
    kp_2, desc_2 = sift.detectAndCompute(gray_2, None)

    show_key_points(rgb_1, kp_1, "1")
    show_key_points(rgb_2, kp_2, "2")

    st.write("Сопоставление ключевых точек на изображениях")
    good_matches_cnt = st.number_input("Количество соответствий:", min_value=1, max_value=999, value=100, step=1)

    match = cv2.BFMatcher()
    matches = match.knnMatch(desc_1, desc_2, k=2)

    good_matches = np.array([m1 for m1, m2 in matches if m1.distance < 0.75 * m2.distance])

    show_matches(rgb_1, kp_1, rgb_2, kp_2, good_matches, good_matches_cnt)
    show_matches(gray_1, kp_1, gray_2, kp_2, good_matches, good_matches_cnt)


def main():
    description()

    rgb_1, gray_1 = loader(URL_1, txt="изображение 1")
    st.write("---")
    rgb_2, gray_2 = loader(URL_2, txt="изображение 2")
    st.write("---")

    match_key_points(rgb_1, gray_1, rgb_2, gray_2)


if __name__ == "__main__":
    main()
