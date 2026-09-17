import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "proem_ai_rag")
    DB_USER = os.getenv("DB_USER", "anusmacbook")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


settings = Settings()