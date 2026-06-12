# scraper.py

import feedparser

sources = {
    "Reuters": "https://feeds.reuters.com/reuters/topNews",
    "Habr": "https://habr.com/ru/rss/articles/?fl=ru",
    
    "Rambler Sport": "https://sport.rambler.ru/rss/",
    
    "BBC Technology": "https://feeds.bbci.co.uk/news/technology/rss.xml",
    
    "BBC Sport": "https://feeds.bbci.co.uk/sport/rss.xml",
    
    "Habr Programming": "https://habr.com/ru/rss/hub/programming/articles/?fl=ru"

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