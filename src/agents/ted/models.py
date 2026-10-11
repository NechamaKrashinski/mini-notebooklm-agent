from langchain.chat_models import init_chat_model
from functools import lru_cache
MODELS = {
    "planning": {"model": "claude-haiku-4-5", "temperature": 0.7},
    "writing": {"model": "claude-sonnet-4-6", "temperature": 0.7},
    "critique": {"model": "claude-haiku-4-5", "temperature": 0.5},
}

@lru_cache
def get_model(role:str):
    return init_chat_model(
        **MODELS[role],
        timeout=30,max_tokens=1000,
        max_retries=6
        )
