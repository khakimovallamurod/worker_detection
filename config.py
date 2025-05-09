from telegram import Bot
from dotenv import load_dotenv
import os

load_dotenv()
def get_token():
    token = os.getenv("TOKEN")
    if token is None:
        raise ValueError("TELEGRAM_TOKEN environment variable not set")
    return token

def get_chat_id():
    chat_id = os.getenv("CHAT_ID")
    if chat_id is None:
        raise ValueError("CHAT_ID environment variable not set")
    return chat_id
