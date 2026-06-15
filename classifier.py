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
import warnings
from typing import Optional

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix
)

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Расширенный и сбалансированный обучающий датасет
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
    ("Real Madrid Barcelona El Clasico stadium football fans", "sport"),
    ("Manchester City striker scores hat trick match", "sport"),
    ("LeBron James breaks scoring record basketball game NBA", "sport"),
    ("Lionel Messi win Ballon dOr trophy football player award", "sport"),
    ("World Athletics Championship runner gold medal final podium", "sport"),
    ("UFC championship fight main event victory decision round", "sport"),
    ("Tour de France cyclist wins stage mountain race", "sport"),
    ("Marathon runner sets new personal best time race marathon", "sport"),
    ("Wimbledon tennis final set victory grass court match", "sport"),
    ("Skateboarder wins gold medal extreme games championship competition", "sport"),
    ("Skiing world cup downhill run race alpine athlete score", "sport"),
    ("Badminton single tournament final match smash win trophy", "sport"),
    ("Swimming championship relay squad wins gold medal competition", "sport"),
    ("Liverpool stadium match ticket soccer game fan club score", "sport"),
    ("Basketball team coach signs extension contract season playoff", "sport"),

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
    ("White House press secretary briefing administration policy statement", "politics"),
    ("Bilateral relations alliance defense treaty diplomatic mission envoy", "politics"),
    ("Senate passes immigration reform package vote bipartisan agreement", "politics"),
    ("Opposition leader delivers speech campaign rally democracy election", "politics"),
    ("Foreign minister visits embassy talks diplomatic cooperation peace", "politics"),
    ("Government sanctions embargo global trade conflict department state", "politics"),
    ("Governor signs state executive order environment protection law", "politics"),
    ("European Union leaders council meeting crisis summit inflation", "politics"),
    ("Political coalition coalition government talks compromise leader majority", "politics"),
    ("Constitutional amendment citizens referendum voting rights bill parliament", "politics"),
    ("Chancellor announces tax cuts economic stimulus recovery plan budget", "politics"),
    ("Electoral commission registers candidates presidential ballot vote", "politics"),
    ("Civil rights legislation debate senators filibuster policy congress", "politics"),
    ("Diplomat expelled embassy tension international conflict threat border", "politics"),
    ("Local authorities municipal government voting system reform district", "politics"),

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
    ("Intel AMD processor architecture CPU desktop microchip nanometer", "technology"),
    ("Data center cloud server migration enterprise software system", "technology"),
    ("Generative AI language model tools chatbot API developer framework", "technology"),
    ("Linux kernel update open source patch security fix server OS", "technology"),
    ("Android smartphone operating system user interface software build", "technology"),
    ("Database server SQL optimization indexing big data engineer backend", "technology"),
    ("Tech giant monopoly hearing antitrust law commission browser app", "technology"),
    ("Smartwatch fitness tracking sensors biometrics tech hardware device", "technology"),
    ("Router WiFi internet wireless network protocol bandwidth optical fiber", "technology"),
    ("E-commerce website tech stack react programming framework platform engineering", "technology"),
    ("Tech incubation fund startup equity series seed accelerator program", "technology"),
    ("Machine learning pipeline training validation accuracy weights model deployment", "technology"),
    ("Cryptographic keys encryption algorithm security protocol hacking protection", "technology"),
    ("Autonomous driving software lidar sensors computer vision radar tech", "technology"),
    ("Supercomputer simulation laboratory research grid processing cluster computing", "technology"),

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
    ("Xbox Game Pass ultimate cloud gaming subscription service catalog", "gaming"),
    ("Retro gaming arcade handheld console remake classic emulator games", "gaming"),
    ("World of Warcraft expansion dungeon raid legendary loot item guide", "gaming"),
    ("Co-op multiplayer survival game open world crafting dedicated server lobby", "gaming"),
    ("Game engine graphics ray tracing physics simulation dev toolkit assets", "gaming"),
]

MODEL_PATH = os.path.join("models", "classifier.pkl")

# ──────────────────────────────────────────────
# ИСПРАВЛЕННЫЕ словари ключевых слов для тегов
# Добавлены русские ключевые слова для всех тегов,
# убраны слишком широкие слова ("team", "space" без контекста и т.д.)
# ──────────────────────────────────────────────
TAG_KEYWORDS: dict[str, list[str]] = {

    # ИИ и машинное обучение
    "AI": [
        # EN
        "artificial intelligence", "machine learning", "neural network",
        "deep learning", "chatgpt", "openai", "llm", "gpt", "large language model",
        "generative ai", "diffusion model", "ai model", "yandex gpt",
        # RU
        "искусственный интеллект", "машинное обучение", "нейросеть",
        "нейронная сеть", "языковая модель", "генеративный", "чатгпт",
        "гигачат", "яндекс гпт", "ии модель", "нейро",
    ],

    # Криптовалюты
    "crypto": [
        # EN
        "bitcoin", "ethereum", "blockchain", "cryptocurrency", "crypto",
        "nft", "defi", "token", "coinbase", "binance", "web3", "altcoin",
        # RU
        "биткоин", "крипта", "криптовалюта", "блокчейн", "токен",
        "майнинг", "майнер", "эфириум", "цифровая валюта", "децентрализованный",
    ],

    # Выборы и политическая жизнь
    "election": [
        # EN
        "election", "ballot", "candidate", "campaign", "polling station",
        "democrat", "republican", "referendum", "vote count", "electoral",
        # RU
        "выборы", "голосование", "кандидат", "избирательный", "референдум",
        "явка", "партия", "предвыборный", "депутат", "избиратель",
        "единая россия", "лдпр", "кпрф",
    ],

    # Климат и экология
    "climate": [
        # EN
        "climate change", "global warming", "carbon emissions", "greenhouse",
        "renewable energy", "solar panel", "wind energy", "net zero",
        "paris agreement", "cop28", "cop29", "fossil fuel", "deforestation",
        # RU
        "климат", "глобальное потепление", "выброс углерода", "парниковый",
        "возобновляемая энергия", "солнечная панель", "ветроэнергетика",
        "экология", "углеродный след", "зелёная энергия", "озоновый слой",
    ],

    # Война и конфликты
    "war": [
        # EN
        "war", "airstrike", "ceasefire", "invasion", "troops deployed",
        "military operation", "missile strike", "offensive", "frontline",
        "armed conflict", "nato", "ukraine", "gaza", "bombardment",
        # RU
        "война", "военная операция", "обстрел", "удар", "перемирие",
        "фронт", "наступление", "оборона", "беспилотник", "дрон",
        "вооружённый конфликт", "нато", "украина", "сво",
    ],

    # Здоровье и медицина
    "health": [
        # EN
        "hospital", "vaccine", "pandemic", "covid", "disease outbreak",
        "clinical trial", "drug approval", "epidemic", "treatment",
        "who health", "mental health", "cancer", "virus", "infection",
        # RU
        "здоровье", "больница", "вакцина", "пандемия", "эпидемия",
        "заболевание", "лечение", "медицина", "врач", "диагноз",
        "онкология", "вирус", "инфекция", "минздрав", "поликлиника",
    ],

    # Космос
    "space": [
        # EN
        "nasa", "spacex", "rocket launch", "satellite orbit", "mars mission",
        "moon landing", "astronaut", "space station", "roscosmos",
        "james webb", "space telescope", "orbital", "spacecraft",
        # RU
        "роскосмос", "ракета", "космонавт", "орбита", "спутник",
        "луна", "марс", "космическая станция", "мкс", "запуск ракеты",
        "космодром", "байконур", "восточный", "космический корабль",
    ],

    # Киберспорт и стриминг
    "esports": [
        # EN
        "esports", "twitch", "streamer", "pro player", "prize pool",
        "gaming tournament", "esports team", "live stream", "youtube gaming",
        "dota 2 tournament", "cs2 major", "valorant champions",
        # RU
        "киберспорт", "стример", "стрим", "трансляция", "призовой фонд",
        "про геймер", "киберспортивный", "турнир по играм",
    ],

    # Игровые консоли
    "console": [
        # EN
        "playstation", "xbox", "nintendo switch", "ps5", "ps4",
        "xbox series x", "xbox series s", "game console", "controller",
        # RU
        "консоль", "приставка", "плейстейшн", "иксбокс", "нинтендо",
        "геймпад", "игровая консоль",
    ],

    # Мобильные технологии
    "mobile": [
        # EN
        "smartphone", "android", "ios", "iphone", "google play",
        "app store", "mobile app", "5g phone", "mobile game",
        # RU
        "смартфон", "мобильный телефон", "мобильное приложение",
        "айфон", "андроид", "планшет", "мобильный интернет",
        "сотовый", "мобильная связь",
    ],

    # Стартапы и инвестиции
    "startup": [
        # EN
        "startup", "venture capital", "ipo", "series a", "series b",
        "funding round", "unicorn company", "angel investor", "accelerator",
        # RU
        "стартап", "венчурный", "инвестиции", "раунд финансирования",
        "инвестор", "привлечение средств", "акселератор", "посевной раунд",
    ],

    # Срочные новости
    "breaking": [
        # EN
        "breaking", "just in", "developing story", "urgent", "alert",
        "breaking news", "live updates", "emergency",
        # RU
        "срочно", "молния", "срочная новость", "экстренное",
        "только что", "оперативно", "чрезвычайная ситуация",
    ],
}


def preprocess_text(text: str) -> str:
    """Приводит текст к нижнему регистру и удаляет лишние символы."""
    text = text.lower()
    text = re.sub(r"[^a-zа-яё0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_tags(title: str, description: str = "") -> list[str]:
    """
    Извлекает теги из заголовка и описания новости.
    Работает с русскими и английскими источниками.
    """
    # Объединяем заголовок и описание, НЕ удаляем символы —
    # нам важны составные фразы типа "машинное обучение"
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
            max_features=10000,
            sublinear_tf=True,
            stop_words='english'
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            class_weight="balanced"
        )),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ МОДЕЛИ")
    print("=" * 60)
    print(f"Accuracy: {accuracy * 100:.1f}%")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nМатрица ошибок:")
    print(confusion_matrix(y_test, y_pred))

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

    # 1. СИЛЬНЫЕ МАРКЕРЫ ДЛЯ ИГР
    GAMING_KEYWORDS = [
        "esports", "gameplay", "twitch", "streamer", "video game", "fortnite", "minecraft", "nintendo", "xbox", "playstation",
        "игры", "видеоигры", "геймер", "киберспорт", "геймплей", "стрим", "релиз игры", "steam", "epic games", "gamedev",
        "dota", "valorant", "overwatch", "league of legends", "counter strike", "speedrun", "battle pass", "free to play"
    ]
    if any(kw in text for kw in GAMING_KEYWORDS):
        return "gaming"

    # 2. ЖЕСТКИЕ ТЕХНОЛОГИЧЕСКИЕ МАРКЕРЫ
    TECH_KEYWORDS = [
        "artificial intelligence", "machine learning", "neural", "quantum", "cybersecurity", "software", "hardware",
        "технологии", "искусственный интеллект", "нейросеть", "микросхема", "процессор", "разработка", "программирование",
        "ит", "it", "смартфон", "гаджет", "робот", "автоматизация", "облачные", "безопасность", "сервер", "ии", "ai",
        "тестирование", "test", "база данных", "бэкенд", "фронтенд", "python", "javascript", "linux", "git", "github"
    ]
    if any(kw in text for kw in TECH_KEYWORDS):
        return "technology"

    # 3. МАРКЕРЫ ПОЛИТИКИ
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
        # Русскоязычные тесты для новых источников
        {
            "title": "Искусственный интеллект от Яндекс получил новые возможности",
            "description": "YandexGPT обновлён до версии 3.0 с улучшенным пониманием контекста."
        },
        {
            "title": "ЦСКА обыграл Спартак в дерби со счётом 2:1",
            "description": "Московский матч прошёл при аншлаге на стадионе Лужники."
        },
        {
            "title": "Президент подписал закон об электронных госуслугах",
            "description": "Документ вступит в силу с 1 января следующего года."
        },
        {
            "title": "Роскосмос запустил новый спутник связи на орбиту",
            "description": "Ракета Союз стартовала с космодрома Байконур."
        },
    ]

    logger.info("\n=== Классификация демо-статей ===")
    results = classify_batch(demo_articles, trained_model)
    for r in results:
        tags_str = ", ".join(f"#{t}" for t in r['tags']) if r['tags'] else "—"
        print(f"[{r['category'].upper():^12}] {r['title']}")
        print(f"               Теги: {tags_str}\n")