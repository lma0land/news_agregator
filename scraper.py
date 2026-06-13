import feedparser
import re
import ssl
from classifier import classify_article, extract_tags, load_model
from database import insert_article

# Отключаем строгую проверку SSL, чтобы старые сертификаты питона не дропали запросы
if hasattr(ssl, '_create_unverified_context'):
    ssl._create_default_https_context = ssl._create_unverified_context

def clean_html(text):
    if not text:
        return ""
    # Удаляем HTML-теги, спецсимволы и остатки скриптов
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"&[a-z0-9#]+;", " ", text)
    return text.strip()

# 🌍 Сверхстабильные и открытые международные/российские RSS-ленты
sources = {
    # 🏛️ ПОЛИТИКА (Открытые и подробные ленты новостей)
    "Интерфакс - Политика": "https://www.interfax.ru/rss.asp", 
    "Euronews Ближний Восток": "https://ru.euronews.com/rss?format=single&level=theme&name=news",
    
    # ⚽ СПОРТ (Стабильные и незащищенные от роботов фиды)
    "Euronews Спорт": "https://ru.euronews.com/rss?level=theme&name=sport",
    "Спорт-Экспресс": "https://www.sport-express.ru/services/materials/news/se/",
    
    # 💻 ТЕХНОЛОГИИ И ИГРЫ
    "Habr IT Новости": "https://habr.com/ru/rss/articles/?fl=ru",
    "Habr Программирование": "https://habr.com/ru/rss/hub/programming/articles/?fl=ru"
}

def scrape_news():
    news = []
    
    # Загружаем модель один раз для экономии ресурсов
    model = load_model()
    if model is None:
        print("⚠️ Модель не найдена, классификация пойдет по ключевым словам.")

    for source, url in sources.items():
        print(f"\n📡 Запрос к источнику: {source}...")
        
        # Маскируемся под стандартный браузер Firefox
        feed = feedparser.parse(url, agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:110.0) Gecko/20100101 Firefox/110.0')

        if not feed.entries:
            print(f"❌ Источник {source} пуст или заблокирован.")
            continue

        inserted_count = 0
        duplicate_count = 0

        for article in feed.entries[:25]:  # Берем по 25 свежих новостей
            title = article.get("title", "").strip()
            raw_description = article.get("summary", "") or article.get("description", "") or ""
            link = article.get("link", "")
            published = article.get("published", "") or article.get("pubDate", "")

            if not title:
                continue

            description = clean_html(raw_description)
            
            # Если описание слишком короткое или пустое, дублируем заголовок для классификатора
            if len(description) < 10:
                description = title

            # Прогоняем через наш гибридный классификатор
            category = classify_article(title, description, model=model)
            tags = ",".join(extract_tags(title, description))

            try:
                # Пытаемся вставить в БД
                insert_article(title, link, description, published, source, category, tags)
                inserted_count += 1
            except Exception:
                # Если сработал UNIQUE constraint (дубликат по ссылке/заголовку)
                duplicate_count += 1

            news.append({
                "title": title, "link": link, "description": description,
                "published": published, "source": source, "category": category, "tags": tags
            })
            
        print(f"✅ Результаты по {source}: Добавлено новых: {inserted_count} | Пропущено дубликатов: {duplicate_count}")

    return news

if __name__ == "__main__":
    print("🚀 Старт очищенного скрапера...")
    scrape_news()
    print("\n🏁 Сбор завершен!")