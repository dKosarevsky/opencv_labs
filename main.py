import streamlit as st

from lab_01 import gauss_blur
from lab_02 import binary
from lab_03 import math_morph
from lab_04 import edges
from lab_05 import contour_analysis
from lab_06 import key_points
from lab_07 import sift

# st.set_page_config(initial_sidebar_state="collapsed")
st.sidebar.image('logo.png', width=300)


def header():
    author = """
        made by [Kosarevsky Dmitry](https://github.com/dKosarevsky) 
        for [opencv](https://github.com/dKosarevsky/iu7/blob/master/8sem/opencv.md) labs
        in [BMSTU](https://bmstu.ru)
    """
    st.header("МГТУ им. Баумана. Кафедра ИУ7")
    st.markdown("**Курс:** Обработка цифровых сигналов")
    st.markdown("**Преподаватель:** Кивва К.А.")
    st.markdown("**Студент:** Косаревский Д.П.")
    st.sidebar.markdown(author)


def main():
    header()
    lab = st.sidebar.radio(
        "Выберите Лабораторную работу:", (
            "1. Фильтр Гаусса.",
            "2. Бинаризация.",
            "3. Мат. морфология.",
            "4. Границы изображения.",
            "5. Контурный анализ.",
            "6. Характерные точки.",
            "7. SIFT.",
        ), index=6
    )[:1]

    if lab == "1":
        gauss_blur.main()

    elif lab == "2":
        binary.main()

    elif lab == "3":
        math_morph.main()

    elif lab == "4":
        edges.main()

    elif lab == "5":
        contour_analysis.main()

    elif lab == "6":
        key_points.main()

    elif lab == "7":
        sift.main()


if __name__ == "__main__":
    main()
