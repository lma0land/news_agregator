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
    ("Esports prize pool player wins video game championship", "gaming"),
    ("Nintendo Switch game update patch multiplayer", "gaming"),
    ("Steam sale indie game developer early access", "gaming"),
    ("Call of Duty Fortnite Minecraft update season", "gaming"),
    ("Gaming console graphics card GPU benchmark FPS", "gaming"),
    ("RPG MMO MOBA battle royale game launch", "gaming"),
    ("Twitch streamer gaming YouTube gameplay stream live", "gaming"),
    ("Game developer studio announces sequel DLC content", "gaming"),
    ("PlayStation Xbox exclusive game trailer reveal", "gaming"),
    ("Mobile game iOS Android free-to-play gacha", "gaming"),
    ("E3 gaming expo showcase announcement trailer game", "gaming"),
    ("Cyberpunk GTA RDR Zelda Elden Ring game review", "gaming"),
    ("Overwatch League esports tournament brackets gameplay", "gaming"),
    ("Gaming PC build hardware setup specs benchmark", "gaming"),
    ("Fortnite World Cup teen esports player wins prize million", "gaming"),
    ("Esports streamer claims gaming championship prize pool record", "gaming"),
    ("Pro gamer esports squad wins tournament Madison Square Garden", "gaming"),
    ("League of Legends worlds esports prize pool streamer gameplay", "gaming"),
    ("Dota 2 The International esports event prize pool biggest game", "gaming"),
    ("CS GO esports major tournament player gameplay clips twitch", "gaming"),
    ("Valorant Champions Tour esports team gameplay highlight reel", "gaming"),
    ("Minecraft speedrun world record gameplay twitch stream viewer", "gaming"),
    ("Elden Ring speedrun player beats game fastest time record", "gaming"),
    ("Genshin Impact new character update gameplay patch notes", "gaming"),
]

MODEL_PATH = os.path.join("models", "classifier.pkl")

# ──────────────────────────────────────────────
# Словари ключевых слов для тегов
# ──────────────────────────────────────────────
TAG_KEYWORDS: dict[str, list[str]] = {
    "AI": ["artificial intelligence", "machine learning", "neural network",
           "deep learning", "chatgpt", "openai", "llm", "gpt", "ии", "нейросеть"],
    "crypto": ["bitcoin", "ethereum", "blockchain", "cryptocurrency",
               "nft", "defi", "crypto", "token", "coinbase", "крипта", "биткоин"],
    "election": ["election", "vote", "ballot", "candidate", "campaign",
                 "polling", "democrat", "republican", "parliament", "выборы", "голосование"],
    "climate": ["climate", "global warming", "carbon", "emissions",
                "renewable", "solar", "wind energy", "green", "климат", "экология"],
    "war": ["war", "conflict", "military", "troops", "missile",
            "airstrike", "ceasefire", "invasion", "defense", "война", "конфликт"],
    "health": ["health", "hospital", "vaccine", "pandemic", "covid",
               "disease", "medicine", "treatment", "doctor", "здоровье", "медицина"],
    "space": ["space", "nasa", "rocket", "satellite", "mars", "moon",
              "astronaut", "orbit", "spacex", "launch", "космос", "ракета"],
    "esports": ["esports", "tournament", "streamer", "twitch", "youtube gaming",
                "pro player", "team", "championship", "prize pool", "киберспорт", "турнир"],
    "console": ["playstation", "xbox", "nintendo", "switch", "ps5",
                "series x", "console", "controller", "консоль", "приставка"],
    "mobile": ["mobile", "android", "ios", "smartphone", "app",
               "google play", "app store", "tablet", "смартфон", "телефон"],
    "startup": ["startup", "funding", "venture capital", "ipo",
                "unicorn", "series a", "investment", "стартап", "инвестиции"],
    "breaking": ["breaking", "urgent", "just in", "developing",
                 "exclusive", "alert", "срочно", "молния"],
}


def preprocess_text(text: str) -> str:
    """Приводит текст к нижнему регистру и удаляет лишние символы."""
    text = text.lower()
    text = re.sub(r"[^a-zа-яё0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_tags(title: str, description: str = "") -> list[str]:
    """Извлекает теги из заголовка и описания новости."""
    combined = (title + " " + description).lower()
    found_tags: list[str] = []
    for tag, keywords in TAG_KEYWORDS.items():
        if any(kw in combined for kw in keywords):
            found_tags.append(tag)
    return found_tags[:5]


def train_model() -> Pipeline:
    """Обучает ML-пайплайн (TF-IDF → LogisticRegression)."""
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
    """Загружает модель из файла."""
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
    Определяет категорию новости гибридным методом: жесткие лингвистические
    маркеры (RU/EN) + ML-предикшн в качестве фонового классификатора.
    """
    text = preprocess_text(title + " " + description)

    # 1. СИЛЬНЫЕ МАРКЕРЫ ДЛЯ ИГР (Исключаем gamedev-софт)
    GAMING_KEYWORDS = [
        "esports", "gameplay", "twitch", "streamer", "video game", "fortnite", "minecraft", "nintendo", "xbox", "playstation",
        "игры", "видеоигры", "геймер", "киберспорт", "геймплей", "стрим", "релиз игры", "steam", "epic games", "gamedev", 
        "dota", "valorant", "overwatch", "league of legends", "counter strike", "speedrun", "battle pass", "free to play"
    ]
    if any(kw in text for kw in GAMING_KEYWORDS):
        return "gaming"

    # 2. ЖЕСТКИЕ ТЕХНОЛОГИЧЕСКИЕ МАРКЕРЫ (Перехватываем код, тесты, базы данных, ИТ)
    TECH_KEYWORDS = [
        "artificial intelligence", "machine learning", "neural", "quantum", "cybersecurity", "software", "hardware",
        "технологии", "искусственный интеллект", "нейросеть", "микросхема", "процессор", "разработка", "программирование",
        "ит", "it", "смартфон", "гаджет", "робот", "автоматизация", "облачные", "безопасность", "сервер", "ии", "ai",
        "тестирование", "test", "база данных", "бэкенд", "фронтенд", "python", "javascript", "linux", "git", "github"
    ]
    if any(kw in text for kw in TECH_KEYWORDS):
        return "technology"

    # 3. МАРКЕРЫ ПОЛИТИКИ (Защита от отсутствия данных на Хабре)
    POLITICS_KEYWORDS = [
        "politics", "president", "election", "parliament", "law", "government", "senate", "governance", "sanctions",
        "политика", "президент", "госдума", "закон", "законопроект", "правительство", "депутат", 
        "министр", "минцифры", "роскомнадзор", "указ", "выборы", "санкции", "суд", "власть", "ведомство", "госуслуги"
    ]
    if any(kw in text for kw in POLITICS_KEYWORDS):
        return "politics"

    # 4. МАРКЕРЫ СПОРТА
    SPORT_KEYWORDS = [
        "football", "basketball", "tennis", "olympic", "soccer", "boxing", "match", "championship",
        "спорт", "футбол", "хоккей", "матч", "чемпионат", "олимпиада", "голы", "счет матча", "бокс", "турнир"
    ]
    if any(kw in text for kw in SPORT_KEYWORDS):
        return "sport"

    # 5. Если лингвистические маркеры молчат — отдаем ML-модели
    if model is None:
        model = load_model()
    
    if model is None:
        return "unknown"
        
    try:
        category = model.predict([text])[0]
        return category
    except Exception:
        return "unknown"


def classify_batch(articles: list[dict],
                   model: Optional[Pipeline] = None) -> list[dict]:
    """Классифицирует список статей и добавляет теги."""
    if model is None:
        model = load_model()

    for article in articles:
        title = article.get("title", "")
        description = article.get("description", "")
        article["category"] = classify_article(title, description, model)
        article["tags"] = extract_tags(title, description)

    return articles


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