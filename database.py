import sqlite3

def create_database():
    conn = sqlite3.connect("news.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS articles(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT UNIQUE,
        link TEXT,
        description TEXT,
        published TEXT,
        source TEXT,
        category TEXT,
        tags TEXT
    )
    """)

    conn.commit()
    conn.close()


def insert_article(title, link, description, published, source, category, tags):
    conn = sqlite3.connect("news.db")
    cursor = conn.cursor()

    try:
        cursor.execute("""
        INSERT INTO articles (title, link, description, published, source, category, tags)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (title, link, description, published, source, category, tags))

        conn.commit()

    except sqlite3.IntegrityError:
        print("Дубликат новости")

    conn.close()


def get_articles():
    conn = sqlite3.connect("news.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM articles ORDER BY id DESC")
    data = cursor.fetchall()

    conn.close()
    return data


create_database()