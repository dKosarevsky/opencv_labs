import streamlit as st
import numpy as np
import cv2

from utils.utils import uploader, validate_url, FILE_TYPES, get_image
from random import choice

CAPTCHA = [
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSgLg-c-AluBHgDZmvJ7jf1fmj99We5Nc2wdw&usqp=CAU",
    "https://neumeka.ru/images/uchebnik/internet/help/captcha/1.jpg"
]
SKELETON = [
    "https://img.joomcdn.net/92607ff7228cdc404d6a2a24aea1c7b9c8fde65e_original.jpeg",
    "https://habrastorage.org/r/w1560/getpro/habr/upload_files/7b9/0c6/00f/7b90c600f701ba16073176b9fbdea1d4.png"
]


def binary(img):
    _, bin_img = cv2.threshold(src=img, thresh=15, maxval=255, type=cv2.THRESH_OTSU)
    return bin_img


def dilate(img, k=5, k_size=(3, 3)):
    kernel = cv2.getStructuringElement(cv2.MORPH_DILATE, ksize=k_size)
    for x in range(k):
        img = cv2.dilate(img, kernel)
    return img


def erode(img, k=5, k_size=(3, 3)):
    kernel = cv2.getStructuringElement(cv2.MORPH_ERODE, ksize=k_size)
    for x in range(k):
        img = cv2.erode(img, kernel)
    return img


def closing(img):
    return dilate(erode(img))


def opening(img):
    return erode(dilate(img))


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
    
        Найти изображение с зашумлёнными простыми формами, например каптча.
        Биноризовать изображение, например методом Оцу или Бредли.
        
        Реализовать все изученные морфологические операции, кроме Hit-or-Miss.
        
        В OpenCV есть Erode, Dilate, getStructuringElement.
        Эрозию и Дилатацию можно взять готовые. Но их нужно показать.
        
        Морфологический скелет лучше делать не на каптче, а например на какой-либо надписи толстым шрифтом (огромными буквами) или например белая лошадь.
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

    c1, c2 = st.columns(2)

    if method == "1":
        bin_img = binary(gray_image)
        dilate_img = dilate(gray_image)

        with c2:
            st.image(dilate_img, width=300)

    if method == "2":
        bin_img = binary(gray_image)
        erode_img = erode(bin_img)

        with c2:
            st.image(erode_img, width=300)

    if method == "3":
        bin_img = binary(gray_image)
        closing_img = closing(bin_img)

        with c2:
            st.image(closing_img, width=300)

    if method == "4":
        bin_img = binary(gray_image)
        opening_img = opening(bin_img)

        with c2:
            st.image(opening_img, width=300)

    if method == "5":
        bin_img = binary(gray_image)
        dilate_c_img = condition_dilate(bin_img)

        with c2:
            st.image(dilate_c_img, width=300)

    if method == "6":
        bin_img = binary(gray_image)
        skeletoning_img = skeletoning(bin_img)

        with c2:
            st.image(skeletoning_img, width=300)


if __name__ == "__main__":
    main()
