# scraper.py

import feedparser

sources = {
    "BBC": "http://feeds.bbci.co.uk/news/rss.xml",
    "Reuters": "https://feeds.reuters.com/reuters/topNews",
    "CNN": "http://rss.cnn.com/rss/edition.rss"
}

def get_news():

    news = []

    for source, url in sources.items():

        feed = feedparser.parse(url)

        for article in feed.entries[:10]:

            news.append({
                "title": article.get("title", ""),
                "link": article.get("link", ""),
                "description": article.get("summary", ""),
                "published": article.get("published", ""),
                "source": source
            })

    return news