# utilitaire de retry pour les appels Groq API
# gere automatiquement les erreurs 429 

import time
import logging
from groq import RateLimitError

logger = logging.getLogger(__name__)

MAX_RETRIES = 6      
BASE_DELAY  = 5.0   


def groq_call_with_retry(fn, *args, **kwargs):

    delay = BASE_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except RateLimitError as e:
            if attempt == MAX_RETRIES:
                raise
            print(f"  [RateLimit] tentative {attempt}/{MAX_RETRIES} — "
                  f"attente {delay:.0f}s avant retry...")
            time.sleep(delay)
            delay *= 2  
