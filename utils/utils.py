from PIL import Image, UnidentifiedImageError
from urllib.parse import urlparse
from io import BytesIO

import streamlit as st
import numpy as np
import requests
import cv2

FILE_TYPES = ["png", "bmp", "jpg", "jpeg"]


def uploader(file):
    show_file = st.empty()
    if not file:
        show_file.info("valid file extension: " + ", ".join(FILE_TYPES))
        return False
    return file


def validate_url(url):
    result = urlparse(url)
    if all([result.scheme, result.netloc]):
        return url
    else:
        st.error("Не похоже на ссылку с изображением, повторите ввод.")
        st.stop()


def get_image(user_img, user_url):
    img = None
    if user_img is not False:
        img = Image.open(user_img)
    else:
        response = requests.get(user_url)
        try:
            img = Image.open(BytesIO(response.content))
        except UnidentifiedImageError:
            st.error("Что-то пошло не так... Попробуйте другую ссылку или загрузите изображение со своего устройства.")
            st.stop()

    arr = np.uint8(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if len(arr.shape) > 2 else arr

    c1, c2 = st.columns(2)
    with c1:
        st.write("Исходное изображение:")
        st.image(img, width=300)

    with c2:
        st.write("Оттенки серого:")
        st.image(gray, width=300)

    return img, gray


def binary(img):
    return cv2.threshold(src=img, thresh=0, maxval=255, type=cv2.THRESH_OTSU)[1]


def loader(url, txt):
    user_img = uploader(st.file_uploader(f"Загрузить {txt}:", type=FILE_TYPES))

    user_url = validate_url(
        st.text_input(f"Ссылка на изображение {FILE_TYPES}: ", url)
    )
    return get_image(user_img, user_url)
