import os
import yaml

GEMINI_CONFIG = {
    "api_key": os.getenv("GEMINI_API_KEY"),
    "model": "gemini-2.5-flash-preview-05-20"
}