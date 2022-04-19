import streamlit as st

from lab_01 import gauss_blur
from lab_02 import binary
from lab_03 import math_morph
# from lab_04 import _______
# from lab_05 import _______
# from lab_06 import _______
# from lab_07 import _______

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
            "3. Математическая морфология.",
            # "4. ______.",
            # "5. ______.",
            # "6. ______.",
            # "7. ______.",
        ),
        index=2
    )

    if lab[:1] == "1":
        gauss_blur.main()

    elif lab[:1] == "2":
        binary.main()

    elif lab[:1] == "3":
        math_morph.main()

    # elif lab[:1] == "4":
    #     _______.main()
    #
    # elif lab[:1] == "5":
    #     _______.main()
    #
    # elif lab[:1] == "6":
    #     _______.main()
    #
    # elif lab[:1] == "7":
    #     _______.main()


if __name__ == "__main__":
    main()

