import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.nlp_utils import normalize_text, tokenize


@dataclass
class BotModel:
    vectorizer: TfidfVectorizer
    classifier: LogisticRegression
    tags: List[str]
    responses: Dict[str, List[str]]


def load_intents(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def prepare_training_data(intents: dict) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    texts = []
    labels = []
    responses = {}

    for intent in intents["intents"]:
        tag = intent["tag"]
        responses[tag] = intent["responses"]
        for pattern in intent["patterns"]:
            texts.append(normalize_text(pattern))
            labels.append(tag)

    return texts, labels, responses


def train_model(texts: List[str], labels: List[str]) -> BotModel:
    vectorizer = TfidfVectorizer(tokenizer=tokenize, ngram_range=(1, 2), min_df=1)
    features = vectorizer.fit_transform(texts)

    classifier = LogisticRegression(max_iter=2000, multi_class="auto")
    classifier.fit(features, labels)

    tags = sorted(set(labels))
    return BotModel(vectorizer=vectorizer, classifier=classifier, tags=tags, responses={})


def save_model(model: BotModel, responses: Dict[str, List[str]], path: Path) -> None:
    payload = {
        "vectorizer": model.vectorizer,
        "classifier": model.classifier,
        "tags": model.tags,
        "responses": responses,
    }
    joblib.dump(payload, path)


def load_model(path: Path) -> BotModel:
    data = joblib.load(path)
    return BotModel(
        vectorizer=data["vectorizer"],
        classifier=data["classifier"],
        tags=data["tags"],
        responses=data["responses"],
    )


def predict_intent(message: str, model: BotModel) -> Tuple[str, float]:
    normalized = normalize_text(message)
    features = model.vectorizer.transform([normalized])
    probs = model.classifier.predict_proba(features)[0]
    best_index = int(probs.argmax())
    return model.classifier.classes_[best_index], float(probs[best_index])
