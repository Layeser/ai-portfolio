import faiss
import pickle
import numpy as np
from backend.rag.vector_store.gather_data import load_data
from backend.rag.vector_store.chunking import chunk_text
from backend.rag.vector_store.embeddings import get_embedding_model

from backend.core.config import settings

def build_index():
    try:
        text = load_data()
        chunks = chunk_text(text)
        embeddings = get_embedding_model(chunks)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings))

        faiss.write_index(index, settings.INDEX_PATH)

        with open(settings.CHUNKS_PATH, "wb") as f:
            pickle.dump(chunks, f)

        return len(chunks)

    except Exception as e:
        raise RuntimeError(f"Index build failed: {e}")