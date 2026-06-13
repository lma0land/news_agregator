from flask import Flask, request
import sqlite3

app = Flask(__name__)

DB_NAME = "news.db"


def get_connection():
    return sqlite3.connect(DB_NAME)


@app.route("/")
def home():

    category = request.args.get("category", "")
    search = request.args.get("search", "")

    conn = get_connection()
    cursor = conn.cursor()

    query = """
    SELECT title, source, category, link
    FROM articles
    WHERE 1=1
    """

    params = []

    if category:
        query += " AND category = ?"
        params.append(category)

    if search:
        query += " AND title LIKE ?"
        params.append(f"%{search}%")

    query += " ORDER BY id DESC"

    cursor.execute(query, params)

    articles = cursor.fetchall()

    conn.close()

    html = """
    <h1>Новостной агрегатор</h1>

    <form>
        <input name="search" placeholder="Поиск">
        <button>Найти</button>
    </form>

    <hr>
    """

    for article in articles:

        title = article[0]
        source = article[1]
        category = article[2]
        link = article[3]

        html += f"""
        <div>
            <h3>{title}</h3>
            <p>{source} | {category}</p>
            <a href="{link}" target="_blank">Открыть</a>
            <hr>
        </div>
        """

    return html


if __name__ == "__main__":
    app.run(debug=True)