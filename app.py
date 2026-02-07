from pathlib import Path

import streamlit as st

from src.bot import load_model, predict_intent
from src.internet import search_web

MODEL_PATH = Path("model.joblib")
CONFIDENCE_THRESHOLD = 0.55


@st.cache_resource
def load_bot():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Не найдена модель. Сначала запустите train.py.")
    return load_model(MODEL_PATH)


def render_chat():
    st.set_page_config(page_title="Нейросеть чат-бот", page_icon="🤖")
    st.title("Нейросеть чат-бот")

    model = load_bot()

    if "history" not in st.session_state:
        st.session_state.history = []

    for role, content in st.session_state.history:
        with st.chat_message(role):
            st.markdown(content)

    prompt = st.chat_input("Введите сообщение")
    if not prompt:
        return

    st.session_state.history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    tag, confidence = predict_intent(prompt, model)
    if confidence < CONFIDENCE_THRESHOLD:
        reply = search_web(prompt) or "Я не уверен в ответе. Попробуйте уточнить вопрос."
    else:
        options = model.responses.get(tag, ["Извините, я пока не знаю ответа на это."])
        reply = options[0]

    st.session_state.history.append(("assistant", reply))
    with st.chat_message("assistant"):
        st.markdown(reply)


if __name__ == "__main__":
    render_chat()
