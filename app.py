from flask import Flask, render_template, request
import sqlite3

app = Flask(__name__)
DB_NAME = "news.db"


def get_connection():
    return sqlite3.connect(DB_NAME)


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


if __name__ == "__main__":
    app.run(debug=True)