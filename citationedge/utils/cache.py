import os
import pickle
import hashlib
from datetime import datetime, timedelta
from citationedge.constants.config import *

def load_cache():
    """Load cached Semantic Scholar results."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'rb') as f:
                cache_data = pickle.load(f)
                current_time = datetime.now()
                cleaned_cache = {}
                for key, (data, timestamp) in cache_data.items():
                    if current_time - timestamp < timedelta(days=CACHE_EXPIRY_DAYS):
                        cleaned_cache[key] = (data, timestamp)
                return cleaned_cache
        except Exception as e:
            print(f"Cache load error: {e}")
            return {}
    return {}

def save_cache(cache_data):
    """Save cache to file."""
    try:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(cache_data, f)
    except Exception as e:
        print(f"Cache save error: {e}")

def generate_cache_key(text: str) -> str:
    """Generate cache key from text."""
    return hashlib.md5(text.encode()).hexdigest()
