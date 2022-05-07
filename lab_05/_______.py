import streamlit as st


def description():

    task = """
        реализовать ...
    """

    st.markdown("### Лабораторная работа №5")
    st.markdown("**Тема:** ______________")

    if st.checkbox("Показать задание"):
        st.write(task)
    st.markdown("---")


def main():
    description()


if __name__ == "__main__":
    main()
