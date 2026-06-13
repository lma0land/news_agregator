import re

SPORT_WORDS = [
    "football", "soccer", "basketball", "tennis",
    "sport", "match", "player", "league"
]

TECH_WORDS = [
    "technology", "tech", "ai", "artificial",
    "software", "computer", "robot", "programming"
]

POLITICS_WORDS = [
    "government", "president", "minister",
    "election", "politics", "parliament"
]


def classify_article(title, description):
    text = f"{title} {description}".lower()

    sport_score = sum(word in text for word in SPORT_WORDS)
    tech_score = sum(word in text for word in TECH_WORDS)
    politics_score = sum(word in text for word in POLITICS_WORDS)

    scores = {
        "Спорт": sport_score,
        "Технологии": tech_score,
        "Политика": politics_score
    }

    category = max(scores, key=scores.get)

    if scores[category] == 0:
        return "Другое"

    return category


def extract_tags(title, description):
    text = f"{title} {description}"

    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())

    stop_words = {
        "that", "this", "with", "from",
        "have", "will", "they", "their"
    }

    tags = []

    for word in words:
        if word not in stop_words and word not in tags:
            tags.append(word)

    return tags[:5]