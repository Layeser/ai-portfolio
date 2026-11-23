FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src HF_HOME=/models TOKENIZERS_PARALLELISM=false
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt ./
COPY src ./src
COPY corpus ./corpus
COPY eval ./eval
RUN pip install --no-cache-dir -r requirements.txt
RUN useradd --create-home app && mkdir -p /models /app/index && chown -R app /app /models
USER app
EXPOSE 8000
# Embedding / reranker weights are downloaded on first start into the /models volume,
# and the index is built if missing (python -m ragassist.retrieval).
CMD ["sh", "-c", "test -f index/chunks.jsonl || python -m ragassist.retrieval; uvicorn ragassist.api:app --host 0.0.0.0 --port 8000"]
