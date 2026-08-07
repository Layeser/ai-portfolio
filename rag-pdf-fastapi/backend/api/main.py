from fastapi import FastAPI

from backend.api.routes.chat import router as chat_router
from backend.api.routes.docs import router as docs_router
from backend.api.routes.health import router as health_router


app = FastAPI(
    title="RAG Shop Assistant API",
    version="1.0.0",
    description="API for RAG-based Shop Assistant",
)

app.include_router(health_router, prefix="/api")
app.include_router(docs_router, prefix="/api/docs")
app.include_router(chat_router, prefix="/api/chat")