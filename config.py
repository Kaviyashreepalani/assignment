import os

EMAIL = os.getenv("EMAIL")
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")

AIPIPE_BASE = "https://aipipe.org/openai/v1"
TEXT_MODEL = "gpt-4o-mini"
VISION_MODEL = "gpt-4o"
EMBED_MODEL = "text-embedding-3-small"