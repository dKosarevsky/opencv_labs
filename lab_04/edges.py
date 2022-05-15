import streamlit as st
import numpy as np
import cv2

from utils.utils import uploader, get_image, FILE_TYPES, validate_url, binary

URL = "https://i.ibb.co/Yj7dj1w/pexels-photo-9035242.jpg"


def description():
    task = """
        Реализовать поиск границ изображения изученными методами.
        Поиск границ реализовать самостоятельно операторами:
            - Roberts
            - Prewitt
            - Sobel
            - Scharr
            - Laplace
        
        Оператор Canny использовать готовый из OpenCV.
    """

    st.markdown("### Лабораторная работа №4")
    st.markdown("**Тема:** Определение Границ на Изображении")

    if st.checkbox("Показать задание"):
        st.write(task)
    st.markdown("---")


def canny(img, thresh_lower, thresh_upper):
    return cv2.Canny(img, thresh_lower, thresh_upper)


def gradient(x, y):
    return np.sqrt(pow(x, 2) + pow(y, 2))


def threshold_cutoff(grad_x, grad_y, thresh):
    grad_mod = gradient(grad_x, grad_y)
    if grad_mod > thresh:
        return 255
    return 0


def is_max_exceeded(x, y, max_x, max_y, img):
    is_exceeded = x == max_x or y == max_y
    if x == max_x or y == max_y:
        img[x, y] = 0
    return is_exceeded


def apply_gradient(grad_x, grad_y, width, height, img, thresh):
    for x in range(0, width):
        for y in range(0, height):
            pixel = threshold_cutoff(grad_x[x][y], grad_y[x][y], thresh)
            img[x, y] = pixel


def roberts(img, threshold=20):
    width, height = img.shape
    max_x = width - 1
    max_y = height - 1

    edge_img = np.zeros((width, height))
    img = np.asarray(img, dtype="float64")

    for x in range(0, width):
        for y in range(0, height):
            if is_max_exceeded(x, y, max_x, max_y, img):
                continue
            grad_x = img[x, y] - img[x + 1, y + 1]
            grad_y = img[x + 1, y] - img[x, y + 1]
            edge_img[x, y] = threshold_cutoff(grad_x, grad_y, threshold)
    return np.asarray(np.clip(edge_img, 0, 255), dtype="uint8")


def convolution(img, matrix_x, matrix_y):
    img = np.asarray(img, dtype="float64")
    width, height = img.shape
    edge_img = np.zeros((width, height))

    for x in range(width - 2):
        for y in range(height - 2):
            grad_x = np.sum(np.multiply(matrix_x, img[x:x + 3, y:y + 3]))
            grad_y = np.sum(np.multiply(matrix_y, img[x:x + 3, y:y + 3]))
            edge_img[x + 1, y + 1] = np.sqrt(grad_x ** 2 + grad_y ** 2)
    return np.asarray(np.clip(edge_img, 0, 255), dtype="uint8")


def prewitt(img):
    prewitt_x = np.array([[1, 0, -1],
                          [1, 0, -1],
                          [1, 0, -1]])
    prewitt_y = np.array([[1, 1, 1],
                          [0, 0, 0],
                          [-1, -1, -1]])

    return binary(convolution(img, prewitt_x, prewitt_y))


def sobel(img):
    sobel_x = np.array([[1.0, 0.0, -1.0],
                        [2.0, 0.0, -2.0],
                        [1.0, 0.0, -1.0]])
    sobel_y = np.array([[1.0, 2.0, 1.0],
                        [0.0, 0.0, 0.0],
                        [-1.0, -2.0, -1.0]])

    return binary(convolution(img, sobel_x, sobel_y))


def scharr(img):
    scharr_x = np.array([[-3, 0, 3],
                         [-10, 0, 10],
                         [-3, 0, 3]])
    scharr_y = np.array([[-3, -10, -3],
                         [0, 0, 0],
                         [3, 10, 3]])

    return binary(convolution(img, scharr_x, scharr_y))


def zeros_crossing(img, thresh):
    d_size = (img.shape[1], img.shape[0])

    mask = np.array([[1, 0, -1],
                     [0, 1, 0]], dtype=np.float32)
    shift_left = cv2.warpAffine(img, mask, d_size)
    mask = np.array([[1, 0, 1],
                     [0, 1, 0]], dtype=np.float32)
    shift_right = cv2.warpAffine(img, mask, d_size)

    mask = np.array([[1, 0, 0],
                     [0, 1, -1]], dtype=np.float32)
    shift_up = cv2.warpAffine(img, mask, d_size)
    mask = np.array([[1, 0, 0],
                     [0, 1, 1]], dtype=np.float32)
    shift_down = cv2.warpAffine(img, mask, d_size)

    mask = np.array([[1, 0, 1],
                     [0, 1, 1]], dtype=np.float32)
    shift_right_down = cv2.warpAffine(img, mask, d_size)
    mask = np.array([[1, 0, -1],
                     [0, 1, -1]], dtype=np.float32)
    shift_left_up = cv2.warpAffine(img, mask, d_size)

    mask = np.array([[1, 0, 1],
                     [0, 1, -1]], dtype=np.float32)
    shift_right_up = cv2.warpAffine(img, mask, d_size)
    mask = np.array([[1, 0, -1],
                     [0, 1, 1]], dtype=np.float32)
    shift_left_down = cv2.warpAffine(img, mask, d_size)

    shift_left_right_sign = shift_left * shift_right
    shift_up_down_sign = shift_up * shift_down
    shift_rd_lu_sign = shift_right_down * shift_left_up
    shift_ru_ld_sign = shift_right_up * shift_left_down

    shift_left_right_norm = np.abs(shift_left - shift_right)
    shift_up_down_norm = np.abs(shift_up - shift_down)
    shift_rd_lu_norm = np.abs(shift_right_down - shift_left_up)
    shift_ru_ld_norm = np.abs(shift_right_up - shift_left_down)

    zero_crossing = (
        ((shift_left_right_sign < 0) & (shift_left_right_norm > thresh)).astype('uint8') +
        ((shift_up_down_sign < 0) & (shift_up_down_norm > thresh)).astype('uint8') +
        ((shift_rd_lu_sign < 0) & (shift_rd_lu_norm > thresh)).astype('uint8') +
        ((shift_ru_ld_sign < 0) & (shift_ru_ld_norm > thresh)).astype('uint8')
    )

    result = np.zeros(shape=img.shape, dtype=np.uint8)
    result[zero_crossing >= 2] = 255

    return result


def laplacian(img, kernel_size, sigma=0, thresh=None, alpha=0.01):
    blur_img = cv2.GaussianBlur(img.astype('float32'), (kernel_size, kernel_size), sigmaX=sigma)
    laplacian_img = cv2.Laplacian(blur_img, cv2.CV_32F)
    if thresh is None:
        thresh = abs(laplacian_img).max() * alpha
    edge_image = zeros_crossing(laplacian_img, thresh)
    return edge_image


def main():
    description()

    method = st.radio(
        "Выберите Оператор:", (
            "1. Canny",
            "2. Roberts",
            "3. Prewitt",
            "4. Sobel",
            "5. Scharr",
            "6. Laplace",
        ),
        index=0
    )[:1]

    user_img = uploader(st.file_uploader("Загрузить изображение:", type=FILE_TYPES))

    user_url = validate_url(
        st.text_input(f"Ссылка на изображение {FILE_TYPES}: ", URL)
    )
    _, gray_image = get_image(user_img, user_url)

    c1, c2 = st.columns(2)
    if c1.checkbox("Сглаживание"):
        gray_image = cv2.GaussianBlur(gray_image, (3, 3), cv2.BORDER_DEFAULT)

    if method == "1":
        thresh_lower = c1.number_input("Нижний порог:", min_value=1, max_value=99, value=50, step=1)
        thresh_upper = c2.number_input("Верхний порог:", min_value=1, max_value=199, value=150, step=1)

        res = canny(gray_image, thresh_lower, thresh_upper)
        st.image(res, width=660)

    if method == "2":
        thresh = c2.number_input("Порог:", min_value=1, max_value=199, value=20, step=1)
        res = roberts(gray_image, thresh)
        st.image(res, width=660)

    if method == "3":
        res = prewitt(gray_image)
        st.image(res, width=660)

    if method == "4":
        res = sobel(gray_image)
        st.image(res, width=660)

    if method == "5":
        res = scharr(gray_image)
        st.image(res, width=660)

    if method == "6":
        alpha = c1.number_input("α:", min_value=0.01, max_value=0.1, value=0.04)
        kernel_size = c2.number_input("Размер ядра:", min_value=1, max_value=199, value=25, step=2)
        edge = laplacian(gray_image, kernel_size=kernel_size, alpha=alpha)
        st.image(edge, width=660)


if __name__ == "__main__":
    main()
