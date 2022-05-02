import streamlit as st
import numpy as np
import cv2

from utils.utils import uploader, validate_url, FILE_TYPES, get_image
from random import choice

CAPTCHA = [
    # "https://i.ibb.co/cLnchh3/image3-4.png",
    # "https://i.ibb.co/KyCzGpN/1-1.jpg",
    "https://i.ibb.co/pQ6KPT2/0-wiwh8t-UKBZBZSHiy.jpg"
]
SKELETON = [
    "https://i.ibb.co/ZccJRvp/92607ff7228cdc404d6a2a24aea1c7b9c8fde65e-original.jpg",
    "https://i.ibb.co/n6pv8qF/1000-F-195035719-Yd7-LNacdb-H7-Bn-Cfr-XIJRqev6-GZ5-K1-ZBQ.jpg"
    "https://i.ibb.co/8YqpjVD/istockphoto-167634727-612x612.jpg"
]


def binary(img):
    return cv2.threshold(src=img, thresh=0, maxval=255, type=cv2.THRESH_OTSU)[1]


def dilate(img, k=5, k_size=(3, 3)):
    kernel = cv2.getStructuringElement(cv2.MORPH_DILATE, ksize=k_size)
    return cv2.dilate(~img, kernel, iterations=k)


def erode(img, k=5, k_size=(3, 3)):
    kernel = cv2.getStructuringElement(cv2.MORPH_ERODE, ksize=k_size)
    return cv2.erode(~img, kernel, iterations=k)


def closing(img, k):
    return dilate(erode(~img, k), k)


def opening(img, k):
    return erode(dilate(~img, k), k)


def condition_dilate(img, dilation_level=3):
    dilation_level = 3 if dilation_level < 3 else dilation_level

    structuring_kernel = np.full(shape=(dilation_level, dilation_level), fill_value=255)

    orig_shape = img.shape
    pad_width = dilation_level - 2

    image_pad = np.pad(array=img, pad_width=pad_width, mode='constant')
    pimg_shape = image_pad.shape
    h_reduce, w_reduce = (pimg_shape[0] - orig_shape[0]), (pimg_shape[1] - orig_shape[1])

    flat_matrix = np.array([
        image_pad[i:(i + dilation_level), j:(j + dilation_level)]
        for i in range(pimg_shape[0] - h_reduce) for j in range(pimg_shape[1] - w_reduce)
    ])

    image_dilate = np.array([255 if (i == structuring_kernel).any() else 0 for i in flat_matrix])
    return image_dilate.reshape(orig_shape)


def skeletoning(img, k_size=(3, 3)):
    skeleton = np.zeros(img.shape, np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, ksize=k_size)
    while True:
        erode_img = cv2.erode(img, kernel)
        dilate_img = cv2.dilate(erode_img, kernel)
        subtract = cv2.subtract(img, dilate_img)
        skeleton = cv2.bitwise_or(skeleton, subtract)
        img = erode_img.copy()

        if cv2.countNonZero(img) == 0:
            break
    return skeleton


def description():
    task = """
    Математическая морфология
    
        1. Дилатация (наращивание)
        2. Эрозия (сужение)
        3. Замыкание (закрытие)
        4. Размыкание (открытие)
        5. Условная дилатация
        6. Преобразование "попадание-пропуск" (Hit-or-Miss transformation)
        7. Морфологический скелет бинарного изображения
    
        Найти изображение с зашумлёнными простыми формами, например капча.
        Биноризовать изображение, например методом Оцу или Брэдли.
        
        Реализовать все изученные морфологические операции, кроме Hit-or-Miss.
        
        В OpenCV есть Erode, Dilate, getStructuringElement.
        Эрозию и Дилатацию можно взять готовые. Но их нужно показать.
        
        Морфологический скелет лучше делать не на капче, а например на какой-либо надписи толстым шрифтом (огромными буквами) или например белая лошадь.
    """

    st.markdown("### Лабораторная работа №3")
    st.markdown("**Тема:** Математическая морфология")

    if st.checkbox("Показать задание"):
        st.write(task)


def main():
    description()

    method = st.radio(
        "Выберите Метод:", (
            "1. Дилатация.",
            "2. Эрозия.",
            "3. Замыкание.",
            "4. Размыкание.",
            "5. Условная дилатация.",
            "6. Морфологический скелет.",
        ),
        index=0
    )[:1]

    user_img = uploader(st.file_uploader("Загрузить изображение:", type=FILE_TYPES))

    user_url = validate_url(
        st.text_input(f"Ссылка на изображение {FILE_TYPES}: ", choice(SKELETON) if method == "6" else choice(CAPTCHA))
    )
    _, gray_image = get_image(user_img, user_url)

    k = st.number_input("Количество итераций:", min_value=1, max_value=99, value=2, step=1)
    c1, c2 = st.columns(2)
    with c1:
        bin_img = binary(gray_image)
        st.write("Бинаризация методом Оцу:")
        st.image(bin_img, width=300)

    with c2:
        if method == "1":
            dilate_img = dilate(bin_img, k)
            st.write("Дилатация:")
            st.image(dilate_img, width=300)

        if method == "2":
            erode_img = erode(bin_img, k)
            st.write("Эрозия:")
            st.image(erode_img, width=300)

        if method == "3":
            closing_img = closing(bin_img, k)
            st.write("Замыкание:")
            st.image(closing_img, width=300)

        if method == "4":
            opening_img = opening(bin_img, k)
            st.write("Размыкание:")
            st.image(opening_img, width=300)

        if method == "5":
            dilate_c_img = condition_dilate(bin_img)
            st.write("Условная дилатация:")
            st.image(dilate_c_img, width=300)

        if method == "6":
            skeletoning_img = skeletoning(bin_img)
            st.write("Морфологический скелет:")
            st.image(skeletoning_img, width=300)
            st.button("Другое изображение")


if __name__ == "__main__":
    main()
