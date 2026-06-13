# database.py

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

create_database()