def retrieve(query, index, chunks, model, k=3):
    query_emb = model.encode([query])
    distances, indices = index.search(query_emb, k)
    return [chunks[i] for i in indices[0]]