from pathlib import Path

from src.bot import load_intents, prepare_training_data, save_model, train_model

DATA_PATH = Path("intents.json")
MODEL_PATH = Path("model.joblib")


def main() -> None:
    intents = load_intents(DATA_PATH)
    texts, labels, responses = prepare_training_data(intents)
    model = train_model(texts, labels)
    save_model(model, responses, MODEL_PATH)
    print(f"Модель сохранена в {MODEL_PATH}")


if __name__ == "__main__":
    main()
