"""
test_classifier.py — Демонстрация работы ML-классификатора.

Запуск:  python test_classifier.py
"""

from classifier import train_model, classify_batch, extract_tags

def main():
    print("=" * 60)
    print("  ОБУЧЕНИЕ ML-МОДЕЛИ")
    print("=" * 60)
    model = train_model()

    test_cases = [
        # GAMING
        {"title": "Elden Ring DLC Shadow of the Erdtree review",
         "description": "FromSoftware expands the hit RPG with massive new content"},
        {"title": "Fortnite Chapter 5 Season 3 battle pass rewards",
         "description": "Epic Games reveals new skins and gameplay mechanics"},
        {"title": "Steam Summer Sale 2024 best deals indie games",
         "description": "Thousands of PC games discounted up to 90 percent"},

        # POLITICS
        {"title": "UN Security Council votes on new sanctions resolution",
         "description": "Permanent members debate emergency ceasefire measures"},
        {"title": "Prime Minister announces snap general election date",
         "description": "Voters head to polls amid economic uncertainty"},

        # SPORT
        {"title": "Novak Djokovic wins record 25th Grand Slam title",
         "description": "Serbian champion beats rival in five-set final"},
        {"title": "Premier League transfer window biggest signings 2024",
         "description": "Top clubs spend record fees on star players"},

        # TECHNOLOGY
        {"title": "OpenAI releases GPT-5 with multimodal capabilities",
         "description": "New AI model achieves human-level reasoning benchmarks"},
        {"title": "NVIDIA RTX 5090 GPU benchmark tests released",
         "description": "New graphics card doubles performance of previous generation"},
    ]

    print("\n" + "=" * 60)
    print("  РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ")
    print("=" * 60)

    results = classify_batch(test_cases, model)
    category_counts: dict[str, int] = {}

    for r in results:
        cat = r["category"]
        tags = r["tags"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
        tag_str = ", ".join(f"#{t}" for t in tags) if tags else "—"
        print(f"\n  [{cat.upper():^12}]  {r['title']}")
        print(f"  {'Теги:':>14}  {tag_str}")

    print("\n" + "=" * 60)
    print("  СТАТИСТИКА ПО КАТЕГОРИЯМ")
    print("=" * 60)
    for cat, count in sorted(category_counts.items()):
        bar = "█" * count
        print(f"  {cat:<12}  {bar}  ({count})")

    print("\n" + "=" * 60)
    print("  ТЕСТ ФУНКЦИИ ТЕГОВ")
    print("=" * 60)
    tag_tests = [
        ("Bitcoin price hits new all-time high as ETF approval nears",
         "Cryptocurrency market surges amid institutional buying"),
        ("NASA Artemis moon mission launch date confirmed",
         "Astronauts prepare for lunar orbit insertion"),
        ("Breaking: Massive earthquake strikes coastal region",
         "Emergency services respond to developing situation"),
    ]
    for title, desc in tag_tests:
        tags = extract_tags(title, desc)
        print(f"\n  Заголовок: {title[:55]}...")
        print(f"  Теги:      {tags}")

    print("\n✅ Тест завершён успешно!\n")


if __name__ == "__main__":
    main()
