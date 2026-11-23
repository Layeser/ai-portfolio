FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt tree.json ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.txt
RUN useradd --create-home app && chown -R app /app
USER app
EXPOSE 8000
# Trained models are mounted at /app/artifacts (not baked into the image).
CMD ["uvicorn", "prodcat.api:app", "--host", "0.0.0.0", "--port", "8000"]
