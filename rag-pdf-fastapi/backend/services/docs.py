from backend.rag.build_vs import build_index

class DocsService:
    def __init__(self):
        pass

    def ingest_docs(self):
        chunks_created = build_index()
        return chunks_created