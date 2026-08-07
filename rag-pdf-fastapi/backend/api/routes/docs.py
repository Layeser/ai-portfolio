from fastapi import APIRouter, HTTPException
from backend.schema.docs import IngestResponse
from backend.services.docs import DocsService


router = APIRouter()
service = DocsService()

@router.post("/ingest", response_model=IngestResponse)
def ingest():
    try:
        chunks_created = service.ingest_docs()
        return IngestResponse(status="success", chunks_created=chunks_created)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
