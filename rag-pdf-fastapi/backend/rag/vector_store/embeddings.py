from sentence_transformers import SentenceTransformer
from backend.core.config import settings

def get_embedding_model(chunks):
    model = SentenceTransformer(settings.MODEL_NAME)
    return model.encode(chunks)