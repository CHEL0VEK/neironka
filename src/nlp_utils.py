import re
from typing import List

RUSSIAN_STOPWORDS = {
    "и",
    "в",
    "на",
    "что",
    "это",
    "как",
    "я",
    "мы",
    "вы",
    "ты",
    "он",
    "она",
    "они",
    "а",
    "но",
    "или",
    "же",
    "то",
    "за",
    "по",
    "о",
    "об",
    "для",
    "с",
    "со",
    "у",
    "к",
    "от",
}


def normalize_text(text: str) -> str:
    cleaned = re.sub(r"[^а-яА-Яa-zA-Z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def tokenize(text: str) -> List[str]:
    normalized = normalize_text(text)
    tokens = [token for token in normalized.split() if token]
    return [token for token in tokens if token not in RUSSIAN_STOPWORDS]
