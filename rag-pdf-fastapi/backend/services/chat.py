from backend.rag.generator import generate_answer


class ChatService:
    def __init__(self):
        pass

    def ask(self, query):
        result = generate_answer(query)
        return result