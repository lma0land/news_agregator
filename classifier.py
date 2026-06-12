"""
classifier.py — ML-классификатор для новостного агрегатора.

Функции:
  - Обучение модели на обучающем датасете (TF-IDF + LogisticRegression)
  - Классификация новостей по категориям: sport, politics, technology, gaming
  - Автоматическое извлечение тегов из текста
  - Сохранение / загрузка обученной модели на диск
"""

import os
import re
import pickle
import logging
from typing import Optional

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Обучающий датасет (заголовок + описание, метка)
# ──────────────────────────────────────────────
TRAINING_DATA = [
    # ── SPORT ───────────────────────────────────
    ("Football match results premier league goals scored", "sport"),
    ("NBA basketball playoffs championship finals game", "sport"),
    ("Tennis grand slam Wimbledon US Open tournament", "sport"),
    ("Olympic games athletes gold medal world record", "sport"),
    ("Soccer team wins championship league title trophy", "sport"),
    ("Boxing match heavyweight champion knockout fight", "sport"),
    ("Formula 1 race driver wins championship podium", "sport"),
    ("Cricket test match innings batting bowling wickets", "sport"),
    ("Rugby union world cup final team wins score", "sport"),
    ("Swimming world record broken athlete competition", "sport"),
    ("Baseball home run pitcher strikeout season stats", "sport"),
    ("Golf tournament birdie eagle player wins major", "sport"),
    ("Hockey NHL playoffs overtime goal season", "sport"),
    ("Athletics sprint marathon runner medal race", "sport"),
    ("Volleyball beach team wins set tournament", "sport"),

    # ── POLITICS ────────────────────────────────
    ("President signs new law policy government reform", "politics"),
    ("Election results vote count candidate party wins", "politics"),
    ("Parliament debate bill legislation passed senate", "politics"),
    ("Prime minister announces budget economic policy", "politics"),
    ("International summit diplomacy treaty agreement", "politics"),
    ("Government policy healthcare education funding", "politics"),
    ("Congress vote tax reform bill signed president", "politics"),
    ("United Nations resolution sanctions foreign affairs", "politics"),
    ("Political party campaign election promises vote", "politics"),
    ("Minister resigns cabinet reshuffle political crisis", "politics"),
    ("Protest demonstration government policy reform", "politics"),
    ("Trade agreement tariff negotiations bilateral deal", "politics"),
    ("Supreme court ruling constitutional law decision", "politics"),
    ("Military defense budget national security policy", "politics"),
    ("Mayor city council vote urban development plan", "politics"),

    # ── TECHNOLOGY ──────────────────────────────
    ("Artificial intelligence AI machine learning model", "technology"),
    ("Apple iPhone release new features specs price", "technology"),
    ("Software update bug fix security patch release", "technology"),
    ("Startup raises funding venture capital investment", "technology"),
    ("Cybersecurity data breach hack vulnerability", "technology"),
    ("Electric vehicle battery range charging tech", "technology"),
    ("Quantum computing breakthrough research lab", "technology"),
    ("5G network rollout smartphone connectivity speed", "technology"),
    ("Blockchain cryptocurrency bitcoin ethereum NFT", "technology"),
    ("Cloud computing AWS Azure Google infrastructure", "technology"),
    ("Robot automation factory manufacturing process", "technology"),
    ("SpaceX rocket launch satellite orbit NASA", "technology"),
    ("Neural network deep learning training dataset", "technology"),
    ("Open source developer GitHub code repository", "technology"),
    ("VR AR headset metaverse virtual reality platform", "technology"),

    # ── GAMING ──────────────────────────────────
    ("Video game release new title PlayStation Xbox PC", "gaming"),
    ("Esports tournament prize pool team wins championship", "gaming"),
    ("Nintendo Switch game update patch multiplayer", "gaming"),
    ("Steam sale indie game developer early access", "gaming"),
    ("Call of Duty Fortnite Minecraft update season", "gaming"),
    ("Gaming console graphics card GPU benchmark FPS", "gaming"),
    ("RPG MMO MOBA battle royale game launch", "gaming"),
    ("Twitch streamer gaming YouTube gameplay stream", "gaming"),
    ("Game developer studio announces sequel DLC content", "gaming"),
    ("PlayStation Xbox exclusive game trailer reveal", "gaming"),
    ("Mobile game iOS Android free-to-play gacha", "gaming"),
    ("E3 gaming expo showcase announcement trailer game", "gaming"),
    ("Cyberpunk GTA RDR Zelda Elden Ring game review", "gaming"),
    ("Overwatch League team tournament brackets winner", "gaming"),
    ("Gaming PC build hardware setup specs benchmark", "gaming"),
]

MODEL_PATH = os.path.join("models", "classifier.pkl")

# ──────────────────────────────────────────────
# Словари ключевых слов для тегов
# ──────────────────────────────────────────────
TAG_KEYWORDS: dict[str, list[str]] = {
    "AI": ["artificial intelligence", "machine learning", "neural network",
           "deep learning", "chatgpt", "openai", "llm", "gpt"],
    "crypto": ["bitcoin", "ethereum", "blockchain", "cryptocurrency",
               "nft", "defi", "crypto", "token", "coinbase"],
    "election": ["election", "vote", "ballot", "candidate", "campaign",
                 "polling", "democrat", "republican", "parliament"],
    "climate": ["climate", "global warming", "carbon", "emissions",
                "renewable", "solar", "wind energy", "green"],
    "war": ["war", "conflict", "military", "troops", "missile",
            "airstrike", "ceasefire", "invasion", "defense"],
    "health": ["health", "hospital", "vaccine", "pandemic", "covid",
               "disease", "medicine", "treatment", "doctor"],
    "space": ["space", "nasa", "rocket", "satellite", "mars", "moon",
              "astronaut", "orbit", "spacex", "launch"],
    "esports": ["esports", "tournament", "streamer", "twitch", "youtube gaming",
                "pro player", "team", "championship", "prize pool"],
    "console": ["playstation", "xbox", "nintendo", "switch", "ps5",
                "series x", "console", "controller"],
    "mobile": ["mobile", "android", "ios", "smartphone", "app",
               "google play", "app store", "tablet"],
    "startup": ["startup", "funding", "venture capital", "ipo",
                "unicorn", "series a", "investment"],
    "breaking": ["breaking", "urgent", "just in", "developing",
                 "exclusive", "alert"],
}


def preprocess_text(text: str) -> str:
    """Приводит текст к нижнему регистру и удаляет лишние символы."""
    text = text.lower()
    text = re.sub(r"[^a-zа-яёa-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_tags(title: str, description: str = "") -> list[str]:
    """
    Извлекает теги из заголовка и описания новости.

    Алгоритм:
      1. Объединяет заголовок и описание.
      2. Ищет совпадения с ключевыми словами из TAG_KEYWORDS.
      3. Возвращает список уникальных тегов (до 5 штук).
    """
    combined = (title + " " + description).lower()
    found_tags: list[str] = []
    for tag, keywords in TAG_KEYWORDS.items():
        if any(kw in combined for kw in keywords):
            found_tags.append(tag)
    return found_tags[:5]  # не более 5 тегов на новость


def train_model() -> Pipeline:
    """
    Обучает ML-пайплайн (TF-IDF → LogisticRegression).

    Шаги:
      1. Предобрабатывает тексты обучающей выборки.
      2. Разбивает на train/test (80/20).
      3. Обучает Pipeline.
      4. Печатает classification_report.
      5. Сохраняет модель в models/classifier.pkl.

    Returns:
        Обученный sklearn Pipeline.
    """
    texts, labels = zip(*TRAINING_DATA)
    texts = [preprocess_text(t) for t in texts]

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=500,
            C=5.0,
            solver="lbfgs",
        )),
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    logger.info("\n" + classification_report(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    logger.info("✅ Модель сохранена: %s", MODEL_PATH)

    return pipeline


def load_model() -> Optional[Pipeline]:
    """
    Загружает модель из файла.

    Returns:
        Pipeline или None, если файл не найден.
    """
    if not os.path.exists(MODEL_PATH):
        logger.warning("Модель не найдена — запустите train_model()")
        return None
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info("✅ Модель загружена: %s", MODEL_PATH)
    return model


def classify_article(title: str, description: str = "",
                     model: Optional[Pipeline] = None) -> str:
    """
    Определяет категорию новости.

    Args:
        title:       Заголовок статьи.
        description: Краткое описание (опционально).
        model:       Обученный Pipeline (если None — загружается с диска).

    Returns:
        Строка-категория: 'sport' | 'politics' | 'technology' | 'gaming'.
    """
    if model is None:
        model = load_model()
    if model is None:
        logger.error("Модель не загружена. Возвращается категория 'unknown'.")
        return "unknown"

    text = preprocess_text(title + " " + description)
    category = model.predict([text])[0]
    return category


def classify_batch(articles: list[dict],
                   model: Optional[Pipeline] = None) -> list[dict]:
    """
    Классифицирует список статей и добавляет теги.

    Args:
        articles: Список словарей с ключами 'title', 'description'.
        model:    Обученный Pipeline (если None — загружается с диска).

    Returns:
        Тот же список с добавленными ключами 'category' и 'tags'.
    """
    if model is None:
        model = load_model()

    for article in articles:
        title = article.get("title", "")
        description = article.get("description", "")
        article["category"] = classify_article(title, description, model)
        article["tags"] = extract_tags(title, description)

    return articles


# ──────────────────────────────────────────────
# Точка входа — запуск обучения и демо-теста
# ──────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=== Обучение модели ===")
    trained_model = train_model()

    demo_articles = [
        {
            "title": "New Call of Duty game released on PlayStation 5",
            "description": "Activision announces major update for the popular shooter."
        },
        {
            "title": "President signs new climate policy bill",
            "description": "The legislation targets carbon emissions by 2030."
        },
        {
            "title": "Apple unveils AI-powered iPhone with neural engine",
            "description": "New device features on-device machine learning capabilities."
        },
        {
            "title": "Champions League final: Real Madrid wins trophy",
            "description": "Dramatic match ends with penalty shootout victory."
        },
        {
            "title": "Esports team wins $1M prize at Fortnite World Cup",
            "description": "Teen streamer claims championship at Madison Square Garden."
        },
    ]

    logger.info("\n=== Классификация демо-статей ===")
    results = classify_batch(demo_articles, trained_model)
    for r in results:
        print(f"[{r['category'].upper():^12}] {r['title']}")
        print(f"               Теги: {r['tags']}\n")
