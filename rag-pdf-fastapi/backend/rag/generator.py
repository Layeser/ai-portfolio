from backend.rag.vs_and_gen.retriver import retrieve
from backend.rag.vs_and_gen.llm_model import llm_model
from backend.core.config import settings
import faiss
import pickle
from openai import OpenAI



def generate_answer(query):
    # Load index and chunks
    index = faiss.read_index("data/guinee.index")
    with open("data/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    # Load LLM model
    model = llm_model()
    client = OpenAI(
        base_url=settings.URL,
        api_key=settings.API_KEY
    )

    # Retrieve relevant chunks
    relevant_chunks = retrieve(query, index, chunks, model)

    context = "\n\n".join(relevant_chunks)
    

    messages = [
        {
            "role": "system",
            "content": "Tu es un assistant expert sur l'histoire de la Guinée. Répond uniquement avec le contexte fourni."
        },
        {
            "role": "user",
            "content": f"""
    Contexte:
    {context}

    Question:
    {query}
    """
        }
    ]

    response = client.chat.completions.create(
        model = settings.CHAT_MODEL,
        messages=messages
    )
    return {
        "answer": response.choices[0].message.content,
        "context": context
    }
