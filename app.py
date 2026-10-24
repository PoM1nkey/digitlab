"""Streamlit frontend: all inference goes through FastAPI."""
import os
from pathlib import Path

import requests
import streamlit as st

st.set_page_config(page_title="DigitLab", page_icon="✍️", layout="wide")
st.title("DigitLab")
st.subheader("Распознавание рукописных цифр")
st.write("Загрузите изображение одной цифры от 0 до 9. Готовая нейросеть MNIST определит её и покажет оценки всех классов.")
api_url = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
with st.sidebar:
    st.header("О проекте")
    st.write("Готовая CNN onnxmodelzoo/mnist-12 с Hugging Face. Обучение в проекте не выполняется.")
    st.caption("Streamlit → FastAPI → ONNX Runtime")
    st.write("Лучше всего: крупная цифра по центру на однотонном фоне.")
    st.caption("Изображение отправляется API для обработки и не сохраняется приложением.")
    st.link_button("Модель на Hugging Face", "https://huggingface.co/onnxmodelzoo/mnist-12")

left, right = st.columns([1, 1], gap="large")
with left:
    mode = st.radio("Источник изображения", ["Готовый пример", "Загрузить файл"], horizontal=True)
    data = None
    if mode == "Готовый пример":
        samples = sorted(Path("samples").glob("*.png"))
        if samples:
            selected = st.selectbox("Пример MNIST", samples, format_func=lambda p: p.stem)
            data = selected.read_bytes()
        else:
            st.info("Примеры пока не подготовлены. Загрузите свой файл.")
    else:
        uploaded = st.file_uploader("PNG, JPEG или WEBP, до 5 МБ", type=["png", "jpg", "jpeg", "webp"])
        if uploaded:
            data = uploaded.getvalue()
    if data:
        st.image(data, width=240, caption="Входное изображение")
    run = st.button("Распознать цифру", type="primary", disabled=data is None, use_container_width=True)
with right:
    if run and data:
        try:
            with st.spinner("Модель обрабатывает изображение…"):
                response = requests.post(f"{api_url}/predict", files={"file": ("digit.png", data)}, timeout=30)
            if response.ok:
                result = response.json()
                st.metric("Результат", str(result["digit"]), border=True)
                st.metric("Оценка модели", f"{result['confidence']:.1%}")
                st.bar_chart({"Цифра": list(map(str, range(10))), "Вероятность": result["probabilities"]}, x="Цифра", y="Вероятность")
                st.caption(result["warning"])
                with st.expander("Ответ API"):
                    st.json(result)
            else:
                st.error(response.json().get("detail", "Ошибка обработки изображения."))
        except (requests.RequestException, ValueError):
            st.error("API недоступен. Запустите сервер FastAPI или проверьте API_URL.")
    else:
        st.info("Выберите пример или загрузите изображение и нажмите «Распознать цифру».")
st.divider()
st.caption("Учебный проект по программной инженерии · Цифры MNIST · Не подходит для распознавания текста и многозначных чисел")
