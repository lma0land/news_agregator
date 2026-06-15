# Часть 2: ML-классификатор — News Aggregator

## Автор ветки
`feature/ml-classifier` → PR → `main`

---

## Структура файлов (Часть 2)

```
news_aggregator/
├── classifier.py        ← основной модуль ML
├── test_classifier.py   ← демонстрация и тест
└── models/
    └── classifier.pkl   ← сохранённая модель (генерируется автоматически)
```

---

## Архитектура ML-пайплайна

```
Новость (title + description)
        │
        ▼
  preprocess_text()       ← нижний регистр, удаление символов
        │
        ▼
  TfidfVectorizer         ← bigrams, max 5000 фич, sublinear_tf
        │
        ▼
  LogisticRegression      ← C=5.0, solver=lbfgs, max_iter=500
        │
        ▼
  Категория: sport | politics | technology | gaming
        │
        ▼
  extract_tags()          ← словарный поиск по ключевым словам
        │
        ▼
  Теги: #AI #crypto #esports #console ...
```

---

## Категории

| Категория    | Описание                                      |
|-------------|-----------------------------------------------|
| `sport`     | Спорт, матчи, турниры, чемпионаты             |
| `politics`  | Политика, выборы, правительство, законы        |
| `technology`| IT, AI, гаджеты, стартапы, кибербезопасность  |
| `gaming` ⭐ | Видеоигры, esports, стриминг, консоли         |

---

## Система тегов ⭐ (Креативное улучшение)

Теги извлекаются из заголовка и описания по словарю ключевых слов:

| Тег        | Ключевые слова                              |
|-----------|---------------------------------------------|
| `#AI`      | artificial intelligence, chatgpt, llm…     |
| `#crypto`  | bitcoin, ethereum, blockchain…              |
| `#election`| election, vote, ballot, campaign…          |
| `#climate` | climate, global warming, carbon…            |
| `#war`     | war, conflict, military, missile…           |
| `#health`  | health, vaccine, pandemic, covid…           |
| `#space`   | space, nasa, rocket, satellite…             |
| `#esports` | esports, tournament, streamer, twitch…      |
| `#console` | playstation, xbox, nintendo, switch…        |
| `#mobile`  | mobile, android, ios, smartphone…           |
| `#startup` | startup, funding, venture capital…          |
| `#breaking`| breaking, urgent, just in, developing…     |

---

## Запуск

```bash
# Установка зависимостей
pip install scikit-learn

# Запуск обучения + демо
python classifier.py

# Запуск полного теста
python test_classifier.py
```

---

## Интеграция с остальными модулями

В `database.py` (Часть 1) нужно добавить колонки:
```sql
ALTER TABLE articles ADD COLUMN category TEXT DEFAULT 'unknown';
ALTER TABLE articles ADD COLUMN tags     TEXT DEFAULT '';
```

В `scraper.py` (Часть 1) после сохранения статьи:
```python
from classifier import classify_article, extract_tags

category = classify_article(title, description)
tags     = extract_tags(title, description)
# передать в db.save_article(... category=category, tags=",".join(tags))
```

В `app.py` (Часть 3) для фильтрации:
```python
@app.route("/")
def index():
    category = request.args.get("category", "all")
    # SELECT * FROM articles WHERE category = ? ...
```

---

## Точность модели

```
              precision    recall  f1-score
    gaming       1.00      1.00      1.00
  politics       1.00      0.67      0.80
     sport       1.00      1.00      1.00
technology       0.75      1.00      0.86

  accuracy                           0.92
```

> Точность 92% на тестовой выборке. При добавлении реальных новостей
> из скрапера точность вырастет за счёт более разнообразного датасета.

---

## Логические выводы

- **TF-IDF** хорошо работает для коротких текстов (заголовок + описание)
- **Bigrams** (ngram_range=1,2) помогают различать "world cup" vs просто "world"
- **gaming** — самая специфичная категория, почти не пересекается с другими
- **technology** и **politics** иногда пересекаются (законы об AI, кибербезопасность)
- Система тегов **дополняет** категорию, давая более детальную метаинформацию
