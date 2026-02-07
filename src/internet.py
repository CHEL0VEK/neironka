from typing import Optional
from urllib.parse import urlencode

import requests

DUCKDUCKGO_ENDPOINT = "https://api.duckduckgo.com/"


def search_web(query: str, timeout_s: int = 6) -> Optional[str]:
    params = {
        "q": query,
        "format": "json",
        "no_redirect": 1,
        "no_html": 1,
        "skip_disambig": 1,
    }
    url = f"{DUCKDUCKGO_ENDPOINT}?{urlencode(params)}"
    response = requests.get(url, timeout=timeout_s)
    response.raise_for_status()
    payload = response.json()

    abstract = payload.get("AbstractText")
    if abstract:
        source = payload.get("AbstractSource") or "DuckDuckGo"
        return f"{abstract} (Источник: {source})"

    related_topics = payload.get("RelatedTopics", [])
    for topic in related_topics:
        if isinstance(topic, dict) and topic.get("Text"):
            return topic["Text"]
        if isinstance(topic, dict) and topic.get("Topics"):
            for subtopic in topic["Topics"]:
                if subtopic.get("Text"):
                    return subtopic["Text"]
    return None
