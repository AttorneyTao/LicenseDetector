import os
import yaml
from dotenv import load_dotenv
load_dotenv()
GEMINI_CONFIG = {
    "api_key": os.getenv("GEMINI_API_KEY"),
    "model": "gemini-2.5-flash-preview-05-20"
}

SCORE_THRESHOLD = 65