import sqlite3
from scraper import get_news
from database import create_database
# Добавляем импорт функций классификации
from classifier import classify_article, extract_tags

create_database()

conn = sqlite3.connect("news.db")
cursor = conn.cursor()

articles = get_news()

for article in articles:
    # 1. Получаем категорию и теги перед вставкой
    title = article["title"]
    desc = article["description"]
    
    category = classify_article(title, desc)
    tags = ",".join(extract_tags(title, desc))

    try:
        # 2. Добавляем category и tags в SQL-запрос
        cursor.execute("""
        INSERT INTO articles
        (title, link, description, published, source, category, tags)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            title,
            article["link"],
            desc,
            article["published"],
            article["source"],
            category,
            tags
        ))

    except sqlite3.IntegrityError:
        pass

conn.commit()

cursor.execute("SELECT * FROM articles")
rows = cursor.fetchall()
for row in rows:
    print(row)

conn.close()