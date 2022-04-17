import plotly.express as px
import streamlit as st
import numpy as np
import cv2

from utils.utils import FILE_TYPES, uploader, validate_url, get_image

URL = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTWe0RVcWbYG_kkoc-Bsz8t8gVW_eX_8bSV3h23C209nTGtrl3vNC3yA7zYwio_d0FGsPA&usqp=CAU"


def description():
    task = """
        реализовать Бинаризацию
        
        1. Метод Оцу OpenCV
        2. Метод Оцу собственная реализация + построить гистограмму яркости и отметить порог
        3. Метод Брэдли - Интегральное представление изображения (не пытаться хранить как 8UC1, хотя бы 32UC1, иначе получим проблемы с яркостью)
        
        В п.1 и п.2 сравнить пороги 
        
    """

    st.markdown("### Лабораторная работа №2")
    st.markdown("**Тема:** Бинаризация")

    if st.checkbox("Показать задание"):
        st.write(task)
    st.markdown("---")


def show_cols(c1, c2):
    with c1:
        threshold = st.number_input("Порог:", min_value=1, max_value=999, value=15, step=1)
    with c2:
        max_value = st.number_input("Макс:", min_value=1, max_value=999, value=255, step=1)

    return threshold, max_value


def otsu_binary(img):
    pixel_number = img.shape[0] * img.shape[1]
    mean_weigth = 1.0 / pixel_number
    hist, bins = np.histogram(img, np.array(range(0, 256)))

    fig = px.histogram(hist, nbins=len(bins))
    st.plotly_chart(fig, use_container_width=True)

    final_thresh = -1
    final_value = -1
    for t in bins[1:-1]:
        wb = np.sum(hist[:t]) * mean_weigth
        wf = np.sum(hist[t:]) * mean_weigth

        mub = np.mean(hist[:t])
        muf = np.mean(hist[t:])

        value = wb * wf * (mub - muf) ** 2

        if value > final_value:
            final_thresh = t
            final_value = value
    final_img = img.copy()
    st.write(f"Порог: {final_thresh}")
    final_img[img > final_thresh] = 255
    final_img[img < final_thresh] = 0
    return final_img


def bradley_binary(img, t):
    s = np.round(img.shape[1] / 8)

    int_img = np.cumsum(np.cumsum(img, axis=1), axis=0)

    (rows, cols) = img.shape[:2]
    (X, Y) = np.meshgrid(np.arange(cols), np.arange(rows))

    X = X.ravel()
    Y = Y.ravel()

    s = s + np.mod(s, 2)

    x1 = X - s / 2
    x2 = X + s / 2
    y1 = Y - s / 2
    y2 = Y + s / 2

    x1[x1 < 0] = 0
    x2[x2 >= cols] = cols - 1
    y1[y1 < 0] = 0
    y2[y2 >= rows] = rows - 1

    x1 = x1.astype(np.int)
    x2 = x2.astype(np.int)
    y1 = y1.astype(np.int)
    y2 = y2.astype(np.int)

    count = (x2 - x1) * (y2 - y1)

    f1_x = x2
    f1_y = y2
    f2_x = x2
    f2_y = y1 - 1
    f2_y[f2_y < 0] = 0
    f3_x = x1 - 1
    f3_x[f3_x < 0] = 0
    f3_y = y2
    f4_x = f3_x
    f4_y = f2_y

    sums = int_img[f1_y, f1_x] - int_img[f2_y, f2_x] - int_img[f3_y, f3_x] + int_img[f4_y, f4_x]

    out = np.ones(rows * cols, dtype=np.bool)
    out[img.ravel() * count <= sums * (100.0 - t) / 100.0] = False

    out = 255 * np.reshape(out, (rows, cols)).astype(np.uint8)

    return out


def main():
    description()

    user_img = uploader(st.file_uploader("Загрузить изображение:", type=FILE_TYPES))
    user_url = validate_url(st.text_input(f"Вставьте ссылку на изображение {FILE_TYPES}: ", URL))

    _, gray_image = get_image(user_img, user_url)

    method = st.radio(
        "Выберите Метод:", (
            "1. OpenCV THRESH_OTSU.",
            "2. Реализация метода Оцу.",
            "3. Реализация метода Брэдли.",
        ),
        index=0
    )

    c1, c2 = st.columns(2)

    if method[:1] == "1":
        thresh, max_val = show_cols(c1, c2)
        final_thresh, final_img = cv2.threshold(src=gray_image, thresh=thresh, maxval=max_val, type=cv2.THRESH_OTSU)
        st.write(f"Порог: {final_thresh}")

        with c2:
            st.image(final_img, width=300)

    if method[:1] == "2":
        res = otsu_binary(gray_image)

        with c2:
            st.image(res, width=300)

    if method[:1] == "3":
        thresh, _ = show_cols(c1, c2)
        res = bradley_binary(gray_image, thresh)
        with c2:
            st.image(res, width=300)


if __name__ == "__main__":
    main()
