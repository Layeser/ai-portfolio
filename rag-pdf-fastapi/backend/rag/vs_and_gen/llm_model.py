from sentence_transformers import SentenceTransformer
from openai import OpenAI

from backend.core.config import settings


def llm_model():
    model = SentenceTransformer(settings.MODEL_NAME)
    return model