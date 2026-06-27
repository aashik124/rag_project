from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_ingestion_service
from app.core.config import settings
from app.db.session import get_db
from app.schemas.document import ChunkingStrategy, DocumentIngestResponse
from app.services.ingestion import DocumentIngestionService
from app.services.text_extractor import UnsupportedFileTypeError

router = APIRouter()


@router.post("/ingest", response_model=DocumentIngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_document(
    file: Annotated[UploadFile, File(...)],
    chunking_strategy: Annotated[ChunkingStrategy, Form(...)] = ChunkingStrategy.recursive,
    session: AsyncSession = Depends(get_db),
    ingestion_service: DocumentIngestionService = Depends(get_ingestion_service),
) -> DocumentIngestResponse:
    try:
        document = await ingestion_service.ingest(session, file, chunking_strategy)
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return DocumentIngestResponse(
        document_id=document.id,
        filename=document.filename,
        chunking_strategy=chunking_strategy,
        chunk_count=document.chunk_count,
        vector_store=settings.vector_store,
    )
