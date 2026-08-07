from fastapi import APIRouter, HTTPException
from backend.schema.chat import ChatRequest, ChatResponse
from backend.services.chat import ChatService

router = APIRouter()
service = ChatService()

@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        result = service.ask(request.query)
        return ChatResponse(answer=result["answer"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))