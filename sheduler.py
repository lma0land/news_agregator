from apscheduler.schedulers.background import BackgroundScheduler
from scraper import scrape_news
import time


def start_scheduler():
    scheduler = BackgroundScheduler()

    # каждые 30 минут
    scheduler.add_job(func=run_scraper, trigger="interval", minutes=30)

    scheduler.start()

    print("[SCHEDULER] Запущен. Обновление каждые 30 минут.")

    try:
        while True:
            time.sleep(2)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()


def run_scraper():
    print("[SCHEDULER] Запуск обновления новостей...")
    scrape_news()
    print("[SCHEDULER] Обновление завершено.")