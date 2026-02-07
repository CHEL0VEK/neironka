import random
from pathlib import Path

from src.bot import load_model, predict_intent
from src.internet import search_web

MODEL_PATH = Path("model.joblib")
CONFIDENCE_THRESHOLD = 0.55


def choose_response(tag: str, responses: dict) -> str:
    options = responses.get(tag)
    if not options:
        return "Извините, я пока не знаю ответа на это."
    return random.choice(options)


def maybe_internet_fallback(message: str) -> str | None:
    keywords = ("найди", "поиск", "что такое", "кто такой", "кто такая", "кто такие", "расскажи")
    lowered = message.lower()
    if lowered.startswith(keywords):
        return search_web(message)
    return None


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Не найдена модель. Сначала запустите train.py.")

    model = load_model(MODEL_PATH)

    print("Чат-бот готов. Напишите 'выход' для завершения.")
    while True:
        message = input("Вы: ").strip()
        if not message:
            continue
        if message.lower() in {"выход", "exit", "quit"}:
            print("Бот: До свидания!")
            break

        internet_reply = maybe_internet_fallback(message)
        if internet_reply:
            print(f"Бот: {internet_reply}")
            continue

        tag, confidence = predict_intent(message, model)
        if confidence < CONFIDENCE_THRESHOLD:
            internet_reply = search_web(message)
            if internet_reply:
                print(f"Бот: {internet_reply}")
            else:
                print("Бот: Извините, я пока не уверен в ответе. Попробуйте переформулировать.")
            continue

        response = choose_response(tag, model.responses)
        print(f"Бот: {response}")


if __name__ == "__main__":
    main()
