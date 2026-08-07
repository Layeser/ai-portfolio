from sentence_transformers import SentenceTransformer
import faiss
import pickle
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# 1. Load model + index
model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("guinee.index")

with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# 2. LLM client (OpenRouter ou OpenAI)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("API_KEY")
)

def retrieve(query, k=3):
    query_emb = model.encode([query])
    distances, indices = index.search(query_emb, k)
    return [chunks[i] for i in indices[0]]

while True:
    query = input("\nQuestion: ")

    # 1. retrieval
    docs = retrieve(query)

    context = "\n\n".join(docs)

    # 2. prompt
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

    # 3. LLM call
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    print("\n🤖 Answer:")
    print(response.choices[0].message.content)