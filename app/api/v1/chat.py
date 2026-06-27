from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_rag_service
from app.db.session import get_db
from app.rag.pipeline import ConversationalRAGService
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/query", response_model=ChatResponse)
async def query(
    payload: ChatRequest,
    session: AsyncSession = Depends(get_db),
    rag_service: ConversationalRAGService = Depends(get_rag_service),
) -> ChatResponse:
    return await rag_service.answer(session, payload.session_id, payload.message)
