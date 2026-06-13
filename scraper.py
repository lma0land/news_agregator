import feedparser
from classifier import classify_article, extract_tags
from database import insert_article

sources = {
    "Reuters": "https://feeds.reuters.com/reuters/topNews",
    "Habr": "https://habr.com/ru/rss/articles/?fl=ru",
    "Rambler Sport": "https://sport.rambler.ru/rss/",
    "BBC Technology": "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "BBC Sport": "https://feeds.bbci.co.uk/sport/rss.xml",
    "Habr Programming": "https://habr.com/ru/rss/hub/programming/articles/?fl=ru"
}


def scrape_news():
    news = []

    for source, url in sources.items():
        feed = feedparser.parse(url)

        for article in feed.entries[:30]:  # 100 слишком тяжело → 30 норм

            title = article.get("title", "")
            description = article.get("summary", "")
            link = article.get("link", "")
            published = article.get("published", "")

            category = classify_article(title, description)
            tags = ",".join(extract_tags(title, description))

            insert_article(
                title,
                link,
                description,
                published,
                source,
                category,
                tags
            )

            news.append({
                "title": title,
                "link": link,
                "description": description,
                "published": published,
                "source": source,
                "category": category,
                "tags": tags
            })

    return news