from uuid import uuid4

from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document, DocumentChunk
from app.rag.vector_store import VectorRecord, VectorStore
from app.schemas.document import ChunkingStrategy
from app.services.chunking import get_chunker
from app.services.embeddings import EmbeddingService
from app.services.text_extractor import extract_text_from_upload


class DocumentIngestionService:
    def __init__(self, embeddings: EmbeddingService, vector_store: VectorStore) -> None:
        self.embeddings = embeddings
        self.vector_store = vector_store

    async def ingest(self, session: AsyncSession, file: UploadFile, strategy: ChunkingStrategy) -> Document:
        text = await extract_text_from_upload(file)
        chunks = get_chunker(strategy).chunk(text)
        if not chunks:
            raise ValueError("The uploaded document did not contain extractable text.")

        document = Document(
            filename=file.filename or "uploaded-document",
            content_type=file.content_type or "application/octet-stream",
            chunking_strategy=strategy.value,
            chunk_count=len(chunks),
        )
        session.add(document)
        await session.flush()

        vectors = await self.embeddings.embed_texts([chunk.text for chunk in chunks])
        records: list[VectorRecord] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            vector_id = str(uuid4())
            session.add(
                DocumentChunk(
                    document_id=document.id,
                    vector_id=vector_id,
                    chunk_index=chunk.index,
                    text=chunk.text,
                )
            )
            records.append(
                VectorRecord(
                    id=vector_id,
                    vector=vector,
                    text=chunk.text,
                    metadata={
                        "document_id": document.id,
                        "filename": document.filename,
                        "chunk_index": chunk.index,
                    },
                )
            )

        await self.vector_store.upsert(records)
        await session.commit()
        await session.refresh(document)
        return document
