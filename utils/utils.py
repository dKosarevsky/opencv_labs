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


def not_valid_url_err():
    return st.error("Не похоже на ссылку с изображением, повторите ввод.")


def validate_url(url):
    try:
        result = urlparse(url)
        if all([result.scheme, result.netloc]):
            return url
        else:
            not_valid_url_err()
            return False
    except AttributeError:
        not_valid_url_err()
        return False


def get_image(user_img, user_url):
    img = None
    if user_img is not False:
        img = Image.open(user_img)
    else:
        response = requests.get(user_url)
        try:
            img = Image.open(BytesIO(response.content))
        except UnidentifiedImageError:
            st.write("Что-то пошло не так... Попробуйте другую ссылку или загрузите изображение со своего устройства.")
            st.stop()

    arr = np.uint8(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

    c1, c2 = st.columns(2)
    with c1:
        st.image(img, width=300)

    with c2:
        st.image(gray, width=300)

    return img, gray
