from utils.utils import validate_url, uploader, get_image, FILE_TYPES
from PIL import Image

import streamlit as st
import numpy as np
import cv2

URL = "https://upload.wikimedia.org/wikipedia/commons/9/9b/Carl_Friedrich_Gauss.jpg"


def description():
    task = """
    1. Загрузить изображение
    2. Создать пустую матрицу для шага 3, используем тип матрицы CV_8UC1 - для оттенков серого
    3. Преобразовать изображение из RGB в оттенки серого с помощью встроенной в opencv функции
    4. Вывести на экран полученный результат
    
    Задание на ЛР:
    Реализовать сглаживание в оттенках серого двумя способами:
     - с помощью встроенной функции ГауссианБлюр
     - вручную с фильтром Гауса, пишем свою функцию свёртки (пригодится для других лаб)
    """

    st.markdown("### Лабораторная работа №1")
    st.markdown("**Тема:** Реализация сглаживания (размытия) по Гауссу.")
    if st.checkbox("Показать задание"):
        st.write(task)
    st.markdown("---")


def show_cols():
    c1, c2 = st.columns(2)
    with c1:
        dim = st.number_input("Размер ядра:", min_value=1, max_value=99, value=3, step=2)
    with c2:
        sig = st.number_input("Сигма (σ):", min_value=1, max_value=99, value=4, step=1)

    return c1, c2, dim, sig


def normalize(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)


def gaussian_kernel(size, sigma=1):
    kernel_1d = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1d[i] = normalize(kernel_1d[i], 0, sigma)
    kernel_2d = np.outer(kernel_1d.T, kernel_1d.T)

    kernel_2d *= 1.0 / kernel_2d.max()

    return kernel_2d


def convolution(image, kernel, column, average=False):
    with column:
        if len(image.shape) == 3:
            st.write(f"Найдено 3 канала: {image.shape}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            st.write(f"Конвертация в серый канал. Размер: {image.shape}")
        else:
            st.write(f"Размер изображения: {image.shape}")

        st.write(f"Размер ядра: {kernel.shape}")

        image_row, image_col = image.shape
        kernel_row, kernel_col = kernel.shape

        output = np.zeros(image.shape)

        pad_height = int((kernel_row - 1) / 2)
        pad_width = int((kernel_col - 1) / 2)

        padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))

        padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

        for row in range(image_row):
            for col in range(image_col):
                output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
                if average:
                    output[row, col] /= kernel.shape[0] * kernel.shape[1]

        st.write(f"Размер изображения на выходе: {output.shape}")

        output_img = Image.fromarray(np.uint8(output))

    return output_img


def gaussian_blur(image, kernel_size, sigma, col):
    kernel = gaussian_kernel(kernel_size, sigma=sigma)
    return convolution(image, kernel, col, average=True)


def main():
    description()

    user_img = uploader(st.file_uploader("Загрузить изображение:", type=FILE_TYPES))
    user_url = validate_url(st.text_input(f"Вставьте ссылку на изображение {FILE_TYPES}: ", URL))

    color_image, gray_image = get_image(user_img, user_url)

    func = st.radio(
        "Выберите Фильтр:", (
            "1. OpenCV GaussianBlur.",
            "2. Собственная реализация.",
        ),
        index=0
    )

    if func[:1] == "1":
        c1, c2, dimension, sigma = show_cols()

        res = cv2.GaussianBlur(gray_image, (dimension, dimension), sigma)

        with c2:
            st.image(res, width=300)

    elif func[:1] == "2":
        c1, c2, dimension, sigma = show_cols()
        res = gaussian_blur(gray_image, dimension, sigma, c1)

        with c2:
            st.image(res, width=300)


if __name__ == "__main__":
    main()
