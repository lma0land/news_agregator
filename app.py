from flask import Flask, render_template, request, jsonify
import sqlite3

app = Flask(__name__)
DB_NAME = "news.db"


def get_connection():
    return sqlite3.connect(DB_NAME)


# 1. Твой существующий роут для обычного сайта (возвращает HTML)
@app.route("/")
def home():
    search = request.args.get("search", "")
    category = request.args.get("category", "")

    conn = get_connection()
    cursor = conn.cursor()

    query = """
        SELECT title, source, link, description, published, category, tags
        FROM articles
        WHERE 1=1
    """
    params = []

    if search:
        query += " AND title LIKE ?"
        params.append(f"%{search}%")

    if category:
        query += " AND category = ?"
        params.append(category)

    query += " ORDER BY id DESC"

    cursor.execute(query, params)
    articles = cursor.fetchall()
    conn.close()

    return render_template("index.html", articles=articles)


# 2. НОВЫЙ ЭНДПОИНТ: REST API для внешних клиентов (возвращает JSON)
@app.route("/api/news")
def get_news_api():
    # Согласно ТЗ: q — поиск, cat — категория
    search = request.args.get("q", "")
    category = request.args.get("cat", "")

    conn = get_connection()
    cursor = conn.cursor()

    # Запрашиваем данные из базы
    query = """
        SELECT title, source, link, description, published, category, tags
        FROM articles
        WHERE 1=1
    """
    params = []

    if search:
        query += " AND title LIKE ?"
        params.append(f"%{search}%")

    if category:
        query += " AND category = ?"
        params.append(category)

    query += " ORDER BY id DESC"

    cursor.execute(query, params)
    raw_articles = cursor.fetchall()
    conn.close()

    # Конвертируем кортежи из БД в список словарей (чтобы сформировать красивый JSON)
    articles_json = []
    for row in raw_articles:
        articles_json.append({
            "title": row[0],
            "source": row[1],
            "link": row[2],
            "description": row[3],
            "published": row[4],
            "category": row[5],
            "tags": row[6]
        })

    # Возвращаем JSON-ответ
    return jsonify(articles_json)


if __name__ == "__main__":
    app.run(debug=True)