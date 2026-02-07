# Нейросеть чат-бот с нуля

Учебный чат-бот на Python: классифицирует интенты с помощью TF-IDF + Logistic Regression и умеет подсказывать ответы из интернета при низкой уверенности.

## Быстрый старт

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python start.py --train
python start.py --mode cli
```

## Веб-интерфейс

```bash
python start.py --train --mode web
```

Откроется Streamlit-интерфейс с чатом в браузере.

## Как устроено

- `intents.json` — набор интентов, вопросов и ответов.
- `train.py` — обучение модели и сохранение в `model.joblib`.
- `chat.py` — интерактивный чат в терминале.
- `app.py` — веб-интерфейс на Streamlit.
- `start.py` — единый вход для запуска (CLI/Web + опциональное обучение).

## Интернет-поиск

Если модель не уверена или запрос начинается с «найди»/«поиск», бот делает запрос в DuckDuckGo и возвращает краткий ответ.
