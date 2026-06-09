# main.py

import sqlite3
from scraper import get_news
from database import create_database

create_database()

conn = sqlite3.connect("news.db")
cursor = conn.cursor()

articles = get_news()

for article in articles:

    try:
        cursor.execute("""
        INSERT INTO articles
        (title, link, description, published, source)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            article["title"],
            article["link"],
            article["description"],
            article["published"],
            article["source"]
        ))

    except sqlite3.IntegrityError:
        pass

conn.commit()

cursor.execute("SELECT * FROM articles")

rows = cursor.fetchall()

for row in rows:
    print(row)

conn.close()