import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Configuration for Google Generative AI
API_KEYS = [
    key.strip() for key in os.getenv("GOOGLE_GENAI_API_KEYS", "").split(",") if key.strip()
]

def configure_genai(api_key_index=0):
    """Configure Google Generative AI with API key from .env"""
    if not API_KEYS:
        raise ValueError("No Google Generative AI API keys found in .env file (GOOGLE_GENAI_API_KEYS)")
    genai.configure(api_key=API_KEYS[api_key_index])

# Model configurations
EMBEDDING_MODEL = 'models/embedding-001'
GENERATION_MODEL = 'models/gemini-1.5-flash-latest'

# Chunking parameters
MAX_CHUNK_WORDS = 200

# Retrieval parameters
DEFAULT_TOP_K = 5