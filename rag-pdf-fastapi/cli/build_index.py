from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# 1. Load PDF
pdf_path = "R-0076.pdf"
reader = PdfReader(pdf_path)

text = ""
for page in reader.pages:
    text += page.extract_text() + "\n"

# 2. Chunking simple
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

chunks = chunk_text(text)

print(f"Nb chunks: {len(chunks)}")

# 3. Embeddings model
model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)

# 4. FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))

# 5. Save
faiss.write_index(index, "guinee.index")

with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("Index créé ✔")