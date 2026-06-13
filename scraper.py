# scraper.py

import feedparser

sources = {

    "Habr": "https://habr.com/ru/rss/articles/?fl=ru",
    "Rambler Sport": "https://sport.rambler.ru/rss/",
    "BBC Politics": "https://feeds.bbci.co.uk/news/politics/rss.xml"
}


def get_news():

    news = []

    for source, url in sources.items():

        feed = feedparser.parse(url)

        for article in feed.entries[:100]:

            news.append({
                "title": article.get("title", ""),
                "link": article.get("link", ""),
                "description": article.get("summary", ""),
                "published": article.get("published", ""),
                "source": source
            })

    return news