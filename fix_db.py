# fix_db.py
import sqlite3
import os
from classifier import classify_article, load_model

def force_fix_database():
    # Проверяем, где лежит скрипт, и принудительно берем эту папку за основу
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "news.db")
    
    print(f"Попытка подключиться к БД по пути: {db_path}")
    
    if not os.path.exists(db_path) or os.path.getsize(db_path) == 0:
        print("⚠️ Предупреждение: Файл news.db в текущей папке пуст или отсутствует.")
        print("Ищем news.db в соседних директориях...")
        # Проверим на один уровень выше
        parent_db = os.path.join(os.path.dirname(base_dir), "news.db")
        if os.path.exists(parent_db) and os.path.getsize(parent_db) > 0:
            db_path = parent_db
            print(f"🎯 Найдена реальная база данных уровнем выше: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    model = load_model()
    
    # Проверяем структуру таблиц в выбранном файле
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [t[0] for t in cursor.fetchall()]
        print(f"📦 Таблицы, найденные в этом файле БД: {tables}")
        
        if 'articles' not in tables:
            print("❌ Ошибка: В этой базе данных нет таблицы 'articles'.")
            print("Убедись, что запускаешь скрипт из той же папки, где лежит заполненный news.db скрапера!")
            conn.close()
            return
            
        cursor.execute("PRAGMA table_info(articles)")
        columns = [col[1] for col in cursor.fetchall()]
        print(f"📊 Структура таблицы 'articles': {columns}")
        
    except Exception as e:
        print(f"❌ Ошибка при чтении структуры: {e}")
        conn.close()
        return

    cursor.execute("SELECT id, title, description, category FROM articles")
    rows = cursor.fetchall()
    
    print(f"\n🔍 Всего в базе найдено новостей: {len(rows)}")
    
    updated_count = 0
    politics_count = 0
    sport_count = 0
    tech_count = 0
    gaming_count = 0
    
    for row in rows:
        row_id, title, description, old_cat = row
        title_str = str(title) if title else ""
        desc_str = str(description) if description else ""
        
        new_cat = classify_article(title_str, desc_str, model).strip().lower()
        
        if new_cat == "politics": politics_count += 1
        elif new_cat == "sport": sport_count += 1
        elif new_cat == "technology": tech_count += 1
        elif new_cat == "gaming": gaming_count += 1
        
        cursor.execute("UPDATE articles SET category = ? WHERE id = ?", (new_cat, row_id))
        updated_count += 1

    conn.commit()
    conn.close()
    
    print("\n📈 --- ИТОГИ КЛАССИФИКАЦИИ ---")
    print(f"🏛️ Политика (politics): {politics_count} шт.")
    print(f"⚽ Спорт (sport): {sport_count} шт.")
    print(f"💻 Технологии (technology): {tech_count} шт.")
    print(f"🎮 Игры (gaming): {gaming_count} шт.")
    print(f"✅ Всего успешно обновлено строк в БД: {updated_count}\n")

if __name__ == "__main__":
    force_fix_database()