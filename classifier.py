"""
classifier.py — стабильный классификатор новостей

Функции:
- Гибридная классификация (rules + ML)
- Единый стандарт категорий
- Извлечение тегов
- Обучение + сохранение модели
"""

import os
import re
import pickle
import logging
from typing import Optional, List, Dict

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ──────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# ЕДИНЫЕ КАТЕГОРИИ (ВАЖНО)
# ──────────────────────────────────────────────
CATEGORIES = {
    "SPORT": "sport",
    "POLITICS": "politics",
    "TECH": "technology",
    "GAMING": "gaming",
    "UNKNOWN": "unknown",
}

MODEL_PATH = os.path.join("models", "classifier.pkl")

# ──────────────────────────────────────────────
# TRAINING DATA
# ──────────────────────────────────────────────
TRAINING_DATA = [
    ("Football match goal championship league", "sport"),
    ("NBA basketball finals playoffs", "sport"),
    ("Olympic games athlete medal record", "sport"),

    ("President signs new law government", "politics"),
    ("Election results parliament vote", "politics"),
    ("UN summit international agreement", "politics"),

    ("Artificial intelligence neural network AI", "technology"),
    ("Apple releases new iPhone software update", "technology"),
    ("Cybersecurity hack data breach", "technology"),

    ("Video game Fortnite esports tournament win", "gaming"),
    ("PlayStation Xbox game release trailer", "gaming"),
    ("Twitch streamer gaming championship", "gaming"),
]

# ──────────────────────────────────────────────
# TAGS
# ──────────────────────────────────────────────
TAG_KEYWORDS = {
    "ai": ["ai", "artificial intelligence", "machine learning", "neural", "gpt"],
    "crypto": ["bitcoin", "ethereum", "crypto", "blockchain"],
    "war": ["war", "conflict", "military", "missile"],
    "election": ["election", "vote", "parliament", "president"],
    "gaming": ["game", "esports", "twitch", "playstation", "xbox"],
    "space": ["nasa", "rocket", "space", "spacex"],
}

# ──────────────────────────────────────────────
# PREPROCESS
# ──────────────────────────────────────────────
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zа-яё0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ──────────────────────────────────────────────
# TAG EXTRACTION
# ──────────────────────────────────────────────
def extract_tags(title: str, description: str = "") -> List[str]:
    text = (title + " " + description).lower()
    tags = []

    for tag, keywords in TAG_KEYWORDS.items():
        if any(k in text for k in keywords):
            tags.append(tag)

    return tags[:5]

# ──────────────────────────────────────────────
# RULE-BASED CLASSIFIER (FAST LAYER)
# ──────────────────────────────────────────────
def rule_classify(text: str) -> Optional[str]:
    text = text.lower()

    gaming_kw = ["game", "esports", "twitch", "playstation", "xbox", "steam", "fortnite", "minecraft"]
    tech_kw = ["ai", "software", "hardware", "neural", "robot", "cyber", "python", "api", "data"]
    politics_kw = ["president", "government", "election", "parliament", "law", "minister"]
    sport_kw = ["football", "basketball", "tennis", "match", "championship", "olympic"]

    if any(k in text for k in gaming_kw):
        return CATEGORIES["GAMING"]
    if any(k in text for k in tech_kw):
        return CATEGORIES["TECH"]
    if any(k in text for k in politics_kw):
        return CATEGORIES["POLITICS"]
    if any(k in text for k in sport_kw):
        return CATEGORIES["SPORT"]

    return None

# ──────────────────────────────────────────────
# TRAIN MODEL
# ──────────────────────────────────────────────
def train_model() -> Pipeline:
    texts, labels = zip(*TRAINING_DATA)
    texts = [preprocess(t) for t in texts]

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ("clf", LogisticRegression(max_iter=400))
    ])

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    logger.info("\n" + classification_report(y_test, preds))

    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    logger.info(f"Model saved to {MODEL_PATH}")
    return model

# ──────────────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────────────
def load_model() -> Optional[Pipeline]:
    if not os.path.exists(MODEL_PATH):
        return None

    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

# ──────────────────────────────────────────────
# MAIN CLASSIFIER (HYBRID)
# ──────────────────────────────────────────────
def classify_article(
    title: str,
    description: str = "",
    model: Optional[Pipeline] = None
) -> str:

    text = preprocess(title + " " + description)

    # 1. RULES FIRST (STABILITY)
    rule_result = rule_classify(text)
    if rule_result:
        return rule_result

    # 2. ML FALLBACK
    if model is None:
        model = load_model()

    if model:
        try:
            return model.predict([text])[0]
        except Exception:
            pass

    return CATEGORIES["UNKNOWN"]

# ──────────────────────────────────────────────
# BATCH
# ──────────────────────────────────────────────
def classify_batch(articles: List[Dict], model: Optional[Pipeline] = None) -> List[Dict]:
    if model is None:
        model = load_model()

    for article in articles:
        title = article.get("title", "")
        desc = article.get("description", "")

        article["category"] = classify_article(title, desc, model)
        article["tags"] = extract_tags(title, desc)

    return articles

# ──────────────────────────────────────────────
# TEST
# ──────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Training model...")
    m = train_model()

    test = [
        {"title": "Fortnite World Cup winner gets prize", "description": ""},
        {"title": "President signs new reform law", "description": ""},
        {"title": "New AI model released by OpenAI", "description": ""},
    ]

    res = classify_batch(test, m)

    for r in res:
        print(r["category"], "|", r["title"])