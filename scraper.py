import feedparser
from classifier import classify_article, extract_tags
from database import insert_article
import re


def clean_html(text):
    return re.sub(r"<.*?>", "", text)


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

        for article in feed.entries[:30]:

            title = article.get("title", "")
            raw_description = article.get("summary", "")
            link = article.get("link", "")
            published = article.get("published", "")

            # ✅ очистка HTML
            description = clean_html(raw_description)

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


if __name__ == "__main__":
    scrape_news()