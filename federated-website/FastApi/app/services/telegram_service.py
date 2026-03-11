import os
import requests
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
SUPPORT_CHAT_ID = os.getenv("SUPPORT_CHAT_ID")


def send_telegram_message(text: str) -> dict:
    if not BOT_TOKEN:
        raise ValueError("BOT_TOKEN is missing in .env")

    if not SUPPORT_CHAT_ID:
        raise ValueError("SUPPORT_CHAT_ID is missing in .env")

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    payload = {
        "chat_id": SUPPORT_CHAT_ID,
        "text": text,
    }

    response = requests.post(url, json=payload, timeout=10)
    response.raise_for_status()
    return response.json()


