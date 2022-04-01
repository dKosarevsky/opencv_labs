import streamlit as st


def description():

    task = """
        реализовать ...
    """

    st.markdown("### Лабораторная работа №1")
    st.markdown("**Тема:** ______________")
    st.markdown("""
        **_____________** 
    — это  ______________
    """)
    st.markdown("---")


def main():
    description()


if __name__ == "__main__":
    main()
