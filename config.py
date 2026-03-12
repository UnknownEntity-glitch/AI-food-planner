import os
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "8356367584:AAF5hc0ueznNw2Kp8feA0AM98nAN-_4-EqY")
DATA_DIR = os.getenv("DATA_DIR", "data")
RECIPES_ZIP = os.getenv("RECIPES_ZIP", "data/recipes.zip")
USE_AGENT = os.getenv("USE_AGENT", "true").lower() == "true"
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")