import streamlit as st
import numpy as np
import cv2

from utils.utils import uploader, get_image, FILE_TYPES, validate_url

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

    return convolution(img, prewitt_x, prewitt_y)


def sobel(img):
    sobel_x = np.array([[1.0, 0.0, -1.0],
                        [2.0, 0.0, -2.0],
                        [1.0, 0.0, -1.0]])
    sobel_y = np.array([[1.0, 2.0, 1.0],
                        [0.0, 0.0, 0.0],
                        [-1.0, -2.0, -1.0]])

    return convolution(img, sobel_x, sobel_y)


def scharr(img):
    scharr_x = np.array([[-3, 0, 3],
                         [-10, 0, 10],
                         [-3, 0, 3]])
    scharr_y = np.array([[-3, -10, -3],
                         [0, 0, 0],
                         [3, 10, 3]])

    return convolution(img, scharr_x, scharr_y)


class Laplacian:
    def __init__(self, kernel_type=4, center=None, c=None):
        kernel4 = np.array([[0, 1, 0],
                            [1, -4, 1],
                            [0, 1, 0]])
        kernel8 = np.array([[1, 1, 1],
                            [1, -8, 1],
                            [1, 1, 1]])
        self.kernel = kernel4 if kernel_type == 4 else kernel8
        c.write(self.kernel)
        if center is None:
            center = ((self.kernel.shape[0] - 1) // 2, (self.kernel.shape[1] - 1) // 2)
        self.center = center

    def __repr__(self):
        return f"kernel: {self.kernel} | center: {self.center}"

    def kernel_width(self):
        return self.kernel.shape[0]

    def kernel_height(self):
        return self.kernel.shape[1]

    def center_to_right(self):
        return self.kernel_width() - 1 - self.center[0]

    def center_to_bottom(self):
        return self.kernel_height() - 1 - self.center[1]

    def center_to_bottom_right(self):
        return self.center_to_right(), self.center_to_bottom()

    def apply_operator(self, gray):
        width, height = gray.shape

        result = np.zeros_like(gray, dtype='int')
        for i in range(self.center[0], width - self.center_to_right()):
            for j in range(self.center[1], height - self.center_to_bottom()):
                result[i, j] = np.sum(
                    self.kernel * gray[
                                  i - self.center[0]:i + self.center_to_right() + 1,
                                  j - self.center[1]:j + self.center_to_bottom() + 1
                                  ]
                )
        return result


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
        st.image(res, clamp=True, width=660)

    if method == "4":
        res = sobel(gray_image)
        st.image(res, width=660)

    if method == "5":
        res = scharr(gray_image)
        st.image(res, width=660)

    if method == "6":
        kernel_type = c2.selectbox("Выберите фильтр", options=(4, 8))
        res = Laplacian(kernel_type=kernel_type, c=c1)
        res = np.abs(res.apply_operator(gray_image))
        st.image(res, clamp=True, width=660)


if __name__ == "__main__":
    main()
